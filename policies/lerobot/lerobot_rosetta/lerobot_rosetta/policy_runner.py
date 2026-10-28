#!/usr/bin/env python
# Copyright 2025 Isaac Blankenau
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LeRobot PolicyRunner.

Wraps LeRobot's async inference stack. It launches the gRPC policy-server
subprocess eagerly in setup() (node configure time) so the model is loaded
before the first goal arrives, then per goal drives a RobotClient whose control
loop pulls observations from and pushes actions to the injected frame I/O
(the node-hosted TopicBridge).
LeRobot-specific parameters are declared and read here so the hosting node
stays framework-agnostic.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import fields
from typing import Optional

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient

# Submodule import, not the package root: pulling lerobot_robot_rosetta/__init__
# would import the full Robot stack just to get the config dataclass.
from lerobot_robot_rosetta.config_rosetta import RosettaConfig
from lerobot_robot_rosetta.rosetta import GATE_SLICE_SEC
from rosetta.contract.schema import Contract
from rosetta.frames.protocols import FrameIO
from rosetta.policies import RunnerFeedback, RunnerResult

SERVER_STARTUP_TIMEOUT_SEC = 120.0  # default; covers model preload (see param)
SERVER_STARTUP_POLL_SEC = 0.5
SERVER_STOP_TIMEOUT_SEC = 5.0
THREAD_JOIN_TIMEOUT_SEC = 2.0


class LeRobotPolicyRunner:
    """PolicyRunner backed by LeRobot's RobotClient."""

    def __init__(self) -> None:
        self._host = None
        self._contract: Optional[Contract] = None
        self._client: Optional[RobotClient] = None
        self._server_process: Optional[subprocess.Popen] = None
        self._server_log = None  # file handle for the server subprocess output
        self._server_log_path: str = ""

    # -------------------- Lifecycle --------------------

    def setup(self, host, contract: Contract) -> None:
        """Declare LeRobot inference settings on the host and store the contract."""
        self._host = host
        self._contract = contract

        # Fail the lifecycle configure early: the live LeRobot stack cannot
        # represent >1 numeric observation key or >1 action key (see
        # ensure_live_observation_layout).
        from lerobot_robot_rosetta.rosetta import ensure_live_observation_layout
        from rosetta.contract.specs import iter_action_specs, iter_observation_specs

        ensure_live_observation_layout(list(iter_observation_specs(contract)), list(iter_action_specs(contract)))

        _p = host.declare
        _p("pretrained_name_or_path", "", "Path or HF repo ID of trained policy", True)
        _p("server_address", "127.0.0.1:8080", "Policy server address (host:port)", True)
        _p("policy_type", "act", "Policy architecture (act, diffusion, ...)", True)
        _p(
            "policy_device",
            "cuda",
            "Device for policy inference (cuda, xpu, mps, cpu). The server resolves it: "
            "a backend its host lacks falls back to cpu",
            True,
        )
        _p("actions_per_chunk", 50, "Number of actions to request per chunk")
        _p("chunk_size_threshold", 0.5, "Queue threshold ratio to request new chunk (0.0-1.0)")
        _p("aggregate_fn_name", "weighted_average", "Action aggregation function")
        _p("launch_local_server", True, "Launch local policy server subprocess", True)
        _p("obs_similarity_atol", 1.0, "L2 tolerance for obs similarity (-1.0 disables)")
        _p("is_classifier", False, "Use reward section as action output", True)
        _p(
            "server_startup_timeout_sec",
            SERVER_STARTUP_TIMEOUT_SEC,
            "Max seconds to wait for the policy server to come up. Covers the "
            "model preload; raise it if the first launch must download weights.",
            True,
        )

        # Warm start: pay subprocess spawn + torch/CUDA init + model preload at
        # configure time so the first goal costs the same as any other. A
        # failure here propagates and fails the lifecycle configure transition.
        if host.get("launch_local_server"):
            self._start_policy_server()

    def teardown(self) -> None:
        """Stop the policy server subprocess if running."""
        self._stop_policy_server()

    # -------------------- Execution --------------------

    def run(
        self,
        frames: FrameIO,
        *,
        task: str,
        stop_event: threading.Event,
    ) -> RunnerResult:
        """Run policy inference until completion or stop_event is set."""
        # Self-heal: the server is started eagerly in setup(); this respawns it
        # (with the same preload args — params are read-only) if it died.
        if self._host.get("launch_local_server"):
            self._start_policy_server()

        # The host's contract, not contract_path: the host may have resolved
        # it from the checkpoint when the parameter is empty.
        robot_config = RosettaConfig.from_contract(
            self._contract,
            id="rosetta",
            is_classifier=self._host.get("is_classifier"),
        )
        robot_config._external_bridge = frames  # Inject the pre-built frame I/O
        # Under sim time the adapter's tick gates pace on the node clock; the
        # goal's stop_event releases a gate blocked by a paused sim clock.
        robot_config._stop_event = stop_event

        config = self._build_client_config(robot_config, task)
        client = RobotClient(config)
        self._client = client

        if not client.start():
            self._client = None
            return RunnerResult(False, "Failed to connect to policy server")

        receiver = threading.Thread(target=client.receive_actions, daemon=True)
        receiver.start()

        # Bridge the cooperative stop_event to LeRobot's shutdown_event.
        watcher_stop = threading.Event()
        watcher = threading.Thread(target=self._watch_stop, args=(stop_event, client, watcher_stop), daemon=True)
        watcher.start()

        try:
            client.control_loop(task=task)
        finally:
            client.stop()
            receiver.join(timeout=THREAD_JOIN_TIMEOUT_SEC)
            watcher_stop.set()
            watcher.join(timeout=THREAD_JOIN_TIMEOUT_SEC)

        self._client = None
        return RunnerResult(True, "Stopped" if stop_event.is_set() else "Completed")

    def request_stop(self) -> None:
        """Interrupt an in-progress control loop.

        Invariant: every stop path must set run()'s goal stop_event before (or
        with) this call. shutdown_event only exits the control loop at its
        while-check; under a paused sim the loop can be blocked inside a
        sim-paced robot call that only the stop_event releases.
        """
        # Local ref: run()'s finally can clear self._client from another thread.
        client = self._client
        if client is not None:
            client.shutdown_event.set()

    def feedback(self) -> RunnerFeedback:
        """Snapshot of action-queue depth and published-action count."""
        client = self._client
        if client is None:
            return RunnerFeedback(0, 0)
        with client.action_queue_lock:
            depth = client.action_queue.qsize()
        with client.latest_action_lock:
            published = max(0, client.latest_action)
        return RunnerFeedback(depth, published)

    # -------------------- Internals --------------------

    @staticmethod
    def _watch_stop(
        stop_event: threading.Event,
        client: RobotClient,
        watcher_stop: threading.Event,
    ) -> None:
        """Set the client's shutdown_event once the cooperative stop is requested.

        Polls at 10ms: once stop_event releases the adapter's sim-pacing gates,
        the control loop free-runs at the (lifted) wall fps until
        shutdown_event stops it, so this poll bounds that window.
        """
        while not watcher_stop.wait(0.01):
            if stop_event.is_set():
                client.shutdown_event.set()
                return

    def _build_client_config(self, robot_config: RosettaConfig, task: str) -> RobotClientConfig:
        """Build RobotClientConfig from ROS2 parameters.

        Under sim time the robot adapter's tick gates pace the control loop on
        the node clock, so the client fps is only a wall-clock ceiling that
        must never bind before the gate itself runs out of resolution. The
        gate resolves at most one tick per GATE_SLICE_SEC, so that resolution
        rate IS the ceiling: faster gains nothing, slower throttles fast sims.
        On wall clock the contract fps paces the loop exactly as before.
        """
        host = self._host
        contract_fps = robot_config.fps
        # A ROS 2 host already declares use_sim_time (TimeSource.attach_node); other hosts get the default.
        host.declare("use_sim_time", False, "Pace on the host's simulated clock")
        use_sim_time = bool(host.get("use_sim_time"))
        control_loop_fps = max(contract_fps, int(1 / GATE_SLICE_SEC)) if use_sim_time else contract_fps
        if use_sim_time:
            host.logger.info(
                f"Sim time active: pacing on the sim clock at contract fps={contract_fps}Hz; "
                f"client fps ceiling lifted to {control_loop_fps}Hz (FPSTracker logs the ceiling)."
            )

        config_kwargs = {
            "robot": robot_config,
            "server_address": host.get("server_address"),
            "policy_type": host.get("policy_type"),
            "pretrained_name_or_path": host.get("pretrained_name_or_path"),
            "policy_device": host.get("policy_device"),
            "task": task,
            "fps": control_loop_fps,
            "actions_per_chunk": host.get("actions_per_chunk"),
            "chunk_size_threshold": host.get("chunk_size_threshold"),
            "aggregate_fn_name": host.get("aggregate_fn_name"),
        }

        # obs_similarity_atol is optional in some LeRobot versions. Gate on availability.
        atol_param = host.get("obs_similarity_atol")
        atol = None if atol_param < 0 else atol_param
        supported = {f.name for f in fields(RobotClientConfig)}
        if "obs_similarity_atol" in supported:
            config_kwargs["obs_similarity_atol"] = atol
        elif atol_param != 1.0:
            host.logger.warning("obs_similarity_atol not supported in this LeRobot version; ignoring.")

        return RobotClientConfig(**config_kwargs)

    def _start_policy_server(self) -> None:
        """Launch the local policy server subprocess (idempotent while running)."""
        if self._server_process is not None and self._server_process.poll() is None:
            return  # already running

        cfg = self._host
        server_address = cfg.get("server_address")
        host, port = server_address.split(":")

        if cfg.get("is_classifier"):
            module = "lerobot_rosetta.classifier_server"
        else:
            module = "lerobot_rosetta.policy_server"

        cmd = [sys.executable, "-m", module, f"--host={host}", f"--port={port}"]
        pretrained = cfg.get("pretrained_name_or_path")
        if pretrained:
            cmd += [
                f"--policy-type={cfg.get('policy_type')}",
                f"--pretrained-name-or-path={pretrained}",
                f"--policy-device={cfg.get('policy_device')}",
            ]
        else:
            cfg.logger.warning(
                "pretrained_name_or_path is empty; the policy server will load "
                "the model on the first goal instead of at startup"
            )
        timeout = cfg.get("server_startup_timeout_sec")

        # Capture stdout+stderr to a log file so startup failures are visible
        # (a pipe could deadlock if nobody drains it while the server runs).
        log = tempfile.NamedTemporaryFile(prefix="rosetta_policy_server_", suffix=".log", delete=False)
        self._server_log = log
        self._server_log_path = log.name
        cfg.logger.info(
            f"Launching {module} on {host}:{port} "
            f"(preload={pretrained or 'none'}, timeout={timeout}s, log: {log.name})..."
        )
        self._server_process = subprocess.Popen(cmd, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._server_process.poll() is not None:
                raise RuntimeError(
                    f"Policy server exited with code {self._server_process.returncode}. "
                    f"Last output:\n{self._read_log_tail()}"
                )
            try:
                with socket.create_connection((host, int(port)), timeout=1.0):
                    cfg.logger.info(f"Policy server ready on {host}:{port}")
                    return
            except (ConnectionRefusedError, socket.timeout, OSError):
                time.sleep(SERVER_STARTUP_POLL_SEC)

        raise RuntimeError(f"Policy server failed to start within {timeout}s. Last output:\n{self._read_log_tail()}")

    def _read_log_tail(self, max_bytes: int = 4000) -> str:
        """Tail of the server log, for diagnostics."""
        try:
            with open(self._server_log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - max_bytes))
                return f.read().decode("utf-8", "replace")
        except OSError:
            return "(no server log available)"

    def _stop_policy_server(self) -> None:
        """Terminate the policy server process if running."""
        proc = self._server_process
        if proc is None or proc.poll() is not None:
            self._close_server_log()
            self._server_process = None
            return
        if self._host is not None:
            self._host.logger.info("Stopping local policy server...")
        proc.terminate()
        try:
            proc.wait(timeout=SERVER_STOP_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=SERVER_STOP_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                if self._host is not None:
                    self._host.logger.error("Policy server did not exit after kill().")
        self._close_server_log()
        self._server_process = None

    def _close_server_log(self) -> None:
        if self._server_log is not None:
            try:
                self._server_log.close()
            except OSError:
                pass
            self._server_log = None
