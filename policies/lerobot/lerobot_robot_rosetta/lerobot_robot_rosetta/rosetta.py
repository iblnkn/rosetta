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
Rosetta: LeRobot Robot adapter over a live robot's frames.

This module presents a :class:`~rosetta.frames.protocols.FrameIO` as a LeRobot
Robot. It translates between the frame dict ({contract_key: np.ndarray | str})
and LeRobot's flattened observation/action shape: images by short key, state
and action as individual namespaced floats. It imports no robot middleware.

Two modes:
    - Standalone: LeRobot hosts the run (its CLIs), so connect() opens the
      robot through the contract's robot interface
      (``RobotInterface.connect``) and disconnect() closes it.
    - Injected: the robot side hosts the run and hands in its FrameIO (via
      config._external_bridge), which this adapter never closes. Used by
      policy_runner_node so launch topic remappings apply.
"""

from __future__ import annotations

import logging
import time
from functools import cached_property
from typing import Optional

import numpy as np
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from rosetta.frames.layout import FrameLayout
from rosetta.frames.protocols import Connection, FrameIO, RobotClock
from rosetta.robots.registry import load_robot_interface

from .config_rosetta import RosettaConfig

# Internal timing constants
WARMUP_TIMEOUT_SEC = 5.0
WARMUP_POLL_SEC = 0.02
# Wall-time poll slice for sim-paced waits. Bounds stop/Ctrl-C latency; also
# the release resolution, so sim-time jitter grows with the real-time factor
# (2ms wall = 2ms x RTF sim -- noticeable above RTF ~15-20 at 30 fps).
GATE_SLICE_SEC = 0.002


class _TickGate:
    """Pace one calling loop at the contract period on the robot's simulated clock.

    No-op unless the clock is simulated. Slice-polls wall time so a stop
    request (or Ctrl-C on the main thread) releases a wait blocked by a paused
    sim, where a clock-native sleep (rclpy's Clock.sleep_until) could not be
    woken by our stop event. Not thread-safe: one gate per calling loop.
    """

    def __init__(self, clock: RobotClock, period_ns: int, stop_event=None):
        self._clock = clock
        self._period_ns = int(period_ns)
        self._stop_event = stop_event
        self._last_ns: Optional[int] = None

    def reset(self) -> None:
        self._last_ns = None

    def wait(self) -> None:
        clock = self._clock
        if not clock.simulated:
            return
        now = clock.now_ns()
        if self._last_ns is None:
            self._last_ns = now  # first gated call releases immediately
            return
        target = self._last_ns + self._period_ns
        while now < target:
            if now < self._last_ns:  # sim clock jumped backwards: re-anchor
                self._last_ns = now
                return
            if self._stop_event is not None:
                if self._stop_event.wait(GATE_SLICE_SEC):
                    return  # released by stop; anchor unchanged
            else:
                time.sleep(GATE_SLICE_SEC)
            now = clock.now_ns()
        # Exact cadence when on time; re-anchor (no burst) after a stall of a
        # period or more (pause, inference stall at a chunk boundary).
        self._last_ns = target if now - target < self._period_ns else now


def ensure_live_observation_layout(observation_specs, action_specs=()) -> None:
    """Reject contracts the live LeRobot stack silently mangles.

    lerobot's hw_to_dataset_features collapses ALL numeric observation leaves
    into a single ``observation.state`` feature (and all action leaves into
    one ``action``), while rosetta's offline datasets keep one feature per
    contract key (the layout LeRobot policies actually expect —
    ``observation.environment_state`` is a distinct policy input). A policy
    trained on the ported dataset would therefore receive missing, misshaped,
    or misrouted inputs live. Fail loudly at connect/setup instead. Counting
    KEYS (not specs) keeps multi-topic aggregation under one key working.
    """
    obs_layout = FrameLayout(list(observation_specs))
    act_layout = FrameLayout(list(action_specs)) if action_specs else None
    numeric_keys = [k for k in obs_layout.keys if obs_layout[k].category == "numeric"]
    action_keys = list(act_layout.keys) if act_layout else []
    problems = []
    if len(numeric_keys) > 1:
        problems.append(f"{len(numeric_keys)} numeric observation keys ({numeric_keys})")
    if len(action_keys) > 1:
        problems.append(f"{len(action_keys)} action keys ({action_keys})")

    # Select-less numeric streams have no selector names, and the live stack
    # derives its motor/feature names from selector names — so such a stream
    # is silently dropped live while the offline dataset keeps it.
    selectless = [
        f"'{sl.spec.key}' (topic {sl.spec.source.channel.topic})"
        for layout in (obs_layout, act_layout)
        if layout is not None
        for k in layout.keys
        if layout[k].category == "numeric"
        for sl in layout[k].slices
        if not sl.spec.names
    ]
    if selectless:
        problems.append(f"numeric streams without select: ({', '.join(selectless)})")
    if problems:
        raise ValueError(
            f"Contract declares {' and '.join(problems)}, which the live "
            f"LeRobot stack silently mangles: it collapses all numeric "
            f"observations into a single 'observation.state' feature and all "
            f"actions into a single 'action' feature (lerobot "
            f"hw_to_dataset_features), and it derives motor names from "
            f"selector names, dropping select-less numeric streams entirely — "
            f"while rosetta datasets keep every key. A policy trained on the "
            f"ported dataset would receive missing, misrouted, or misshaped "
            f"inputs live. Merge numeric streams under one key per role "
            f"(values concatenate in declaration order) and give every "
            f"numeric stream a select:, or deploy with a framework adapter "
            f"that preserves per-key layout."
        )


class Rosetta(Robot):
    """LeRobot Robot over a live robot's frames (see the module docstring for the two modes)."""

    config_class = RosettaConfig
    name = "rosetta"

    def __init__(self, config: RosettaConfig):
        super().__init__(config)
        self._config: RosettaConfig = config

        # Injected mode: the robot side's FrameIO, which this adapter never closes.
        self._external_bridge: Optional[FrameIO] = getattr(config, "_external_bridge", None)
        # The live robot while connected; in standalone mode also the Connection we close.
        self._frames: Optional[FrameIO] = None
        self._connection: Optional[Connection] = None

        # Sim pacing: created in connect(), no-ops unless the clock is simulated
        self._obs_gate: Optional[_TickGate] = None
        self._act_gate: Optional[_TickGate] = None

    # -------------------- LeRobot <-> frame-dict adapters --------------------
    #
    # All name/shape derivations come from the same FrameLayout the offline
    # dataset writer uses (build_lerobot_features), so live and ported
    # features agree by construction rather than by test.

    @cached_property
    def _obs_layout(self) -> FrameLayout:
        return FrameLayout(list(self.config.observation_specs))

    @cached_property
    def _act_layout(self) -> FrameLayout:
        return FrameLayout(list(self.config.action_specs))

    @cached_property
    def _obs_vector_names(self) -> dict[str, list[str]]:
        """Numeric observation key -> ordered namespaced selector names."""
        feats = self._obs_layout.lerobot_features()
        return {
            key: list(feats[key]["names"] or [])
            for key in self._obs_layout.keys
            if self._obs_layout[key].category == "numeric"
        }

    @cached_property
    def _obs_image_keys(self) -> dict[str, str]:
        """Image observation full key -> LeRobot short key."""
        return {
            key: key.removeprefix("observation.images.")
            for key in self._obs_layout.keys
            if self._obs_layout[key].category == "image"
        }

    @cached_property
    def _act_vector_names(self) -> dict[str, list[str]]:
        """Action key -> ordered namespaced selector names."""
        feats = self._act_layout.lerobot_features()
        return {
            key: list(feats[key]["names"] or [])
            for key in self._act_layout.keys
            if self._act_layout[key].category == "numeric"
        }

    def _flatten_observation(self, frame: dict) -> RobotObservation:
        """Frame dict -> LeRobot observation (images by short key, state as floats)."""
        obs: RobotObservation = {}
        for key, names in self._obs_vector_names.items():
            vec = np.asarray(frame[key]).flatten()
            for i, name in enumerate(names):
                obs[name] = float(vec[i])
        for key, short in self._obs_image_keys.items():
            obs[short] = frame[key]  # (H, W, C) uint8 passthrough
        return obs

    def _unflatten_action(self, action: RobotAction) -> dict:
        """LeRobot action (namespaced floats) -> frame dict (combined per-key vectors)."""
        action_frame: dict = {}
        for key, names in self._act_vector_names.items():
            action_frame[key] = np.array([action[name] for name in names], dtype=np.float32)
        return action_frame

    # -------------------- Features --------------------

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Feature spec: individual state values as float, images as (H, W, C) tuples.

        Image shapes come from FrameLayout.lerobot_features(), which declares the
        DECODED shape (h, w, 3) — decoders always emit 3-channel RGB, so
        declaring the source encoding's channel count (1 for mono8, 4 for
        rgba8) mismatched every delivered frame.
        """
        features: dict[str, type | tuple] = {}
        feats = self._obs_layout.lerobot_features()
        for key, short in self._obs_image_keys.items():
            features[short] = tuple(feats[key]["shape"])
        for names in self._obs_vector_names.values():
            for name in names:
                features[name] = float
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Feature spec: individual action values as float."""
        return {name: float for names in self._act_vector_names.values() for name in names}

    # -------------------- Connection state --------------------

    @property
    def is_connected(self) -> bool:
        """Returns True when ready to send/receive data."""
        return self._frames is not None

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        # Kept for interface compatibility. The open connection is authoritative.
        del value  # unused

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        """Nothing to configure: connect() opens the robot ready to run."""

    def connect(self, calibrate: bool = True) -> None:
        """Attach to the injected FrameIO, or open the robot through the contract's robot interface."""
        del calibrate  # unused; the robot needs no calibration
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Live LeRobot cannot represent >1 numeric observation key (or >1
        # action key); fail loudly here instead of silently recording/serving
        # lumped features.
        ensure_live_observation_layout(self.config.observation_specs, self.config.action_specs)

        if self._external_bridge is not None:
            frames = self._external_bridge
        else:
            interface = load_robot_interface(self.config.contract.robot_interface)
            frames = self._connection = interface.connect(
                observations=self.config.observation_specs,
                actions=self.config.action_specs,
                fps=self.config.fps,
                name=f"rosetta_{self.config.id}",
                use_sim_time=self.config.use_sim_time,
            )
        self._frames = frames
        self._make_gates(frames.clock)
        self._wait_for_warmup(frames)

    def _make_gates(self, clock: RobotClock) -> None:
        """Create the per-loop sim-pacing gates (no-ops under wall clock).

        Two independent gates: LeRobot's control loop skips get_observation
        when the action queue is deep and send_action when it is empty, so a
        shared gate would halve the rate when one iteration does both.
        """
        period_ns = int(1e9 / self.config.fps)
        stop = getattr(self._config, "_stop_event", None)
        self._obs_gate = _TickGate(clock, period_ns, stop)
        self._act_gate = _TickGate(clock, period_ns, stop)
        if clock.simulated:
            logging.info(
                "Pacing observations/actions at %d fps on simulated time (a stalled sim clock will block gated calls).",
                self.config.fps,
            )

    def _wait_for_warmup(self, frames: FrameIO) -> None:
        """Block until every observation stream has delivered a message.

        The bag porter skips warmup ticks, so ported datasets never contain
        a cold bridge's zero-filled frames; gate the live side on the same
        predicate (frames.warmed_up) so first frames agree. Fail open with a
        warning after the timeout — the missing-stream zero-fill/watchdog
        machinery covers a stream that stays silent.
        """
        deadline = time.monotonic() + WARMUP_TIMEOUT_SEC
        while time.monotonic() < deadline:
            if frames.warmed_up:
                return
            time.sleep(WARMUP_POLL_SEC)
        logging.warning(
            "Observation streams not warmed up after %.1fs; first frames may "
            "contain zero-fill (a ported dataset would have skipped them).",
            WARMUP_TIMEOUT_SEC,
        )

    def disconnect(self) -> None:
        """Stop the robot and release it: close our connection, or detach from the injected FrameIO."""
        if self._frames is None:
            return
        # Let go first: a close that raises has still released what it could
        # (Connection.close), so the robot reads as disconnected either way.
        frames, connection = self._frames, self._connection
        self._frames = self._connection = None
        if connection is not None:
            # Closing sends the safety action and releases the middleware.
            connection.close()
        else:
            # Injected: stop the robot and clear state for the next episode, but
            # never tear down a FrameIO the robot side owns.
            frames.send_safety_action()
            frames.reset_state()

    def reset(self) -> None:
        """Reset internal state tracking (e.g., between episodes)."""
        if self._frames is not None:
            self._frames.reset_state()
        for gate in (self._obs_gate, self._act_gate):
            if gate is not None:
                gate.reset()

    # -------------------- Observation / Action --------------------

    def get_observation(self) -> RobotObservation:
        """
        Get current observations.

        Returns a RobotObservation with state values as individual namespaced
        floats and images by short key. Missing data is zero-filled.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self._obs_gate is not None:
            self._obs_gate.wait()
        return self._flatten_observation(self._frames.sample_frame())

    def send_action(self, action: RobotAction) -> RobotAction:
        """
        Publish an action to the robot.

        action is keyed by namespaced selector names. Returns the values sent.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self._act_gate is not None:
            self._act_gate.wait()
        self._frames.publish_frame(self._unflatten_action(action))
        # Echo back the values sent, as the namespaced floats LeRobot recorded.
        return {name: action[name] for names in self._act_vector_names.values() for name in names}

    @property
    def config(self) -> RosettaConfig:
        return self._config

    @config.setter
    def config(self, value: RosettaConfig) -> None:
        self._config = value
