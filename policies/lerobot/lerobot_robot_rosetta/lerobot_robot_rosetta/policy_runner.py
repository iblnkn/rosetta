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

Wraps LeRobot's async inference stack. It lazily launches the gRPC policy_server
subprocess, then drives a RobotClient whose control loop pulls observations from
and pushes actions to the injected TopicBridge. LeRobot-specific parameters are
declared and read here so the hosting node stays framework-agnostic.
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
from rcl_interfaces.msg import ParameterDescriptor

from rosetta.backends.protocols import RunnerFeedback, RunnerResult
from rosetta.core.contract import Contract
from rosetta.ros2.topic_bridge import TopicBridge

from .config_rosetta import RosettaConfig

SERVER_STARTUP_TIMEOUT_SEC = 30.0
SERVER_STARTUP_POLL_SEC = 0.5
SERVER_STOP_TIMEOUT_SEC = 5.0
THREAD_JOIN_TIMEOUT_SEC = 2.0


class LeRobotPolicyRunner:
    """PolicyRunner backed by LeRobot's RobotClient."""

    def __init__(self) -> None:
        self._node = None
        self._contract: Optional[Contract] = None
        self._contract_path: str = ''
        self._client: Optional[RobotClient] = None
        self._server_process: Optional[subprocess.Popen] = None
        self._server_log = None  # file handle for the server subprocess output
        self._server_log_path: str = ''

    # -------------------- Lifecycle --------------------

    def setup(self, node, contract: Contract) -> None:
        """Declare LeRobot inference parameters on node and store the contract."""
        self._node = node
        self._contract = contract
        # contract_path is declared by the node and shared by all backends. Read it.
        self._contract_path = node.get_parameter('contract_path').value

        declare = node.declare_parameter

        def _p(name, default, desc, read_only=False):
            if not node.has_parameter(name):
                declare(
                    name,
                    default,
                    ParameterDescriptor(description=desc, read_only=read_only),
                )

        _p('pretrained_name_or_path', '', 'Path or HF repo ID of trained policy', True)
        _p('server_address', '127.0.0.1:8080', 'Policy server address (host:port)', True)
        _p('policy_type', 'act', 'Policy architecture (act, diffusion, ...)', True)
        _p('policy_device', 'cuda', 'Device for policy inference (cuda, cpu)', True)
        _p('actions_per_chunk', 50, 'Number of actions to request per chunk')
        _p('chunk_size_threshold', 0.5, 'Queue threshold ratio to request new chunk (0.0-1.0)')
        _p('aggregate_fn_name', 'weighted_average', 'Action aggregation function')
        _p('launch_local_server', True, 'Launch local policy server subprocess', True)
        _p('obs_similarity_atol', 1.0, 'L2 tolerance for obs similarity (-1.0 disables)')
        _p('is_classifier', False, 'Use reward section as action output', True)
        _p('sim_time_multiplier', 1.0, 'fps multiplier for slow sims (contract_fps * mult)')

    def teardown(self) -> None:
        """Stop the policy server subprocess if running."""
        self._stop_policy_server()

    # -------------------- Execution --------------------

    def run(
        self,
        bridge: TopicBridge,
        *,
        task: str,
        stop_event: threading.Event,
    ) -> RunnerResult:
        """Run policy inference until completion or stop_event is set."""
        if self._node.get_parameter('launch_local_server').value:
            self._start_policy_server()

        robot_config = RosettaConfig(
            id='rosetta',
            config_path=self._contract_path,
            is_classifier=self._node.get_parameter('is_classifier').value,
        )
        robot_config._external_bridge = bridge  # Inject pre-built bridge

        config = self._build_client_config(robot_config, task)
        client = RobotClient(config)
        self._client = client

        if not client.start():
            self._client = None
            return RunnerResult(False, 'Failed to connect to policy server')

        receiver = threading.Thread(target=client.receive_actions, daemon=True)
        receiver.start()

        # Bridge the cooperative stop_event to LeRobot's shutdown_event.
        watcher_stop = threading.Event()
        watcher = threading.Thread(
            target=self._watch_stop, args=(stop_event, client, watcher_stop), daemon=True
        )
        watcher.start()

        try:
            client.control_loop(task=task)
        finally:
            client.stop()
            receiver.join(timeout=THREAD_JOIN_TIMEOUT_SEC)
            watcher_stop.set()
            watcher.join(timeout=THREAD_JOIN_TIMEOUT_SEC)

        cancelled = stop_event.is_set()
        self._client = None
        return RunnerResult(not cancelled, 'Cancelled' if cancelled else 'Completed')

    def request_stop(self) -> None:
        """Interrupt an in-progress control loop."""
        # Local ref: run()'s finally can clear self._client from another thread.
        client = self._client
        if client is not None:
            client.shutdown_event.set()

    def feedback(self) -> RunnerFeedback:
        """Snapshot of action-queue depth and published-action count."""
        client = self._client
        if client is None:
            return RunnerFeedback(0, 0, 'idle')
        with client.action_queue_lock:
            depth = client.action_queue.qsize()
        with client.latest_action_lock:
            published = max(0, client.latest_action)
        return RunnerFeedback(depth, published, 'executing')

    # -------------------- Internals --------------------

    @staticmethod
    def _watch_stop(
        stop_event: threading.Event,
        client: RobotClient,
        watcher_stop: threading.Event,
    ) -> None:
        """Set the client's shutdown_event once the cooperative stop is requested."""
        while not watcher_stop.wait(0.1):
            if stop_event.is_set():
                client.shutdown_event.set()
                return

    def _build_client_config(
        self, robot_config: RosettaConfig, task: str
    ) -> RobotClientConfig:
        """Build RobotClientConfig from ROS2 parameters, with sim-time fps scaling."""
        node = self._node
        contract_fps = robot_config.fps
        sim_multiplier = node.get_parameter('sim_time_multiplier').value
        control_loop_fps = int(contract_fps * sim_multiplier)
        if sim_multiplier != 1.0:
            node.get_logger().info(
                f'Applied sim_time_multiplier={sim_multiplier:.2f}: '
                f'contract fps={contract_fps}Hz -> control loop fps={control_loop_fps}Hz'
            )

        config_kwargs = {
            'robot': robot_config,
            'server_address': node.get_parameter('server_address').value,
            'policy_type': node.get_parameter('policy_type').value,
            'pretrained_name_or_path': node.get_parameter('pretrained_name_or_path').value,
            'policy_device': node.get_parameter('policy_device').value,
            'task': task,
            'fps': control_loop_fps,
            'actions_per_chunk': node.get_parameter('actions_per_chunk').value,
            'chunk_size_threshold': node.get_parameter('chunk_size_threshold').value,
            'aggregate_fn_name': node.get_parameter('aggregate_fn_name').value,
        }

        # obs_similarity_atol is optional in some LeRobot versions. Gate on availability.
        atol_param = node.get_parameter('obs_similarity_atol').value
        atol = None if atol_param < 0 else atol_param
        supported = {f.name for f in fields(RobotClientConfig)}
        if 'obs_similarity_atol' in supported:
            config_kwargs['obs_similarity_atol'] = atol
        elif atol_param != 1.0:
            node.get_logger().warning(
                'obs_similarity_atol not supported in this LeRobot version; ignoring.'
            )

        return RobotClientConfig(**config_kwargs)

    def _start_policy_server(self) -> None:
        """Launch the local policy server subprocess (idempotent while running)."""
        if self._server_process is not None and self._server_process.poll() is None:
            return  # already running

        node = self._node
        server_address = node.get_parameter('server_address').value
        host, port = server_address.split(':')

        if node.get_parameter('is_classifier').value:
            module = 'lerobot_robot_rosetta.classifier_server'
        else:
            module = 'lerobot.async_inference.policy_server'

        # Capture stdout+stderr to a log file so startup failures are visible
        # (a pipe could deadlock if nobody drains it while the server runs).
        log = tempfile.NamedTemporaryFile(
            prefix='rosetta_policy_server_', suffix='.log', delete=False
        )
        self._server_log = log
        self._server_log_path = log.name
        node.get_logger().info(
            f'Launching {module} on {host}:{port} (log: {log.name})...'
        )
        cmd = [sys.executable, '-m', module, f'--host={host}', f'--port={port}']
        self._server_process = subprocess.Popen(
            cmd, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT
        )

        start_time = time.time()
        while time.time() - start_time < SERVER_STARTUP_TIMEOUT_SEC:
            if self._server_process.poll() is not None:
                raise RuntimeError(
                    f'Policy server exited with code {self._server_process.returncode}. '
                    f'Last output:\n{self._read_log_tail()}'
                )
            try:
                with socket.create_connection((host, int(port)), timeout=1.0):
                    node.get_logger().info(f'Policy server ready on {host}:{port}')
                    return
            except (ConnectionRefusedError, socket.timeout, OSError):
                time.sleep(SERVER_STARTUP_POLL_SEC)

        raise RuntimeError(
            f'Policy server failed to start within {SERVER_STARTUP_TIMEOUT_SEC}s. '
            f'Last output:\n{self._read_log_tail()}'
        )

    def _read_log_tail(self, max_bytes: int = 4000) -> str:
        """Tail of the server log, for diagnostics."""
        try:
            with open(self._server_log_path, 'rb') as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - max_bytes))
                return f.read().decode('utf-8', 'replace')
        except OSError:
            return '(no server log available)'

    def _stop_policy_server(self) -> None:
        """Terminate the policy server process if running."""
        proc = self._server_process
        if proc is None or proc.poll() is not None:
            self._close_server_log()
            self._server_process = None
            return
        if self._node is not None:
            self._node.get_logger().info('Stopping local policy server...')
        proc.terminate()
        try:
            proc.wait(timeout=SERVER_STOP_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=SERVER_STOP_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                if self._node is not None:
                    self._node.get_logger().error('Policy server did not exit after kill().')
        self._close_server_log()
        self._server_process = None

    def _close_server_log(self) -> None:
        if self._server_log is not None:
            try:
                self._server_log.close()
            except OSError:
                pass
            self._server_log = None
