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
Rosetta: LeRobot Robot adapter over the framework-neutral TopicBridge.

The ROS2 plumbing (subscriptions, lifecycle publishers, watchdog, resampling)
lives in rosetta.robots.ros2.topic_bridge. This module presents that bridge as a LeRobot
Robot. It translates between the frame dict ({contract_key: np.ndarray | str}) and
LeRobot's flattened observation/action shape: images by short key, state and
action as individual namespaced floats.

Two modes:
    - Standalone: creates a BridgeLifecycleNode internally (own node, executor,
      spin thread).
    - Injected: attaches to a pre-built TopicBridge on an external node (via
      config._external_bridge). Used by policy_runner_node so launch topic
      remappings apply.
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
from rclpy.parameter import Parameter

from rosetta.frames.layout import FrameLayout
from rosetta.robots.ros2.node_host import NodeHost
from rosetta.robots.ros2.rclpy_utils import require_transition_success
from rosetta.robots.ros2.rosetta_lifecycle_node import BridgeLifecycleNode
from rosetta.robots.ros2.topic_bridge import TopicBridge

from .config_rosetta import RosettaConfig

# Internal timing constants
WARMUP_TIMEOUT_SEC = 5.0
WARMUP_POLL_SEC = 0.02
# Wall-time poll slice for sim-paced waits. Bounds stop/Ctrl-C latency; also
# the release resolution, so sim-time jitter grows with the real-time factor
# (2ms wall = 2ms x RTF sim -- noticeable above RTF ~15-20 at 30 fps).
GATE_SLICE_SEC = 0.002


class _TickGate:
    """Pace one calling loop at the contract period on the ROS (sim) clock.

    No-op unless the clock reports ROS time active. Slice-polls wall time so
    a stop request (or Ctrl-C on the main thread) releases a wait blocked by
    a paused sim. rclpy Clock.sleep_until is deliberately not used: it leaks
    a strong on_shutdown callback per call and cannot be woken by our stop
    event. Not thread-safe: one gate per calling loop.
    """

    def __init__(self, clock, period_ns: int, stop_event=None):
        self._clock = clock
        self._period_ns = int(period_ns)
        self._stop_event = stop_event
        self._last_ns: Optional[int] = None

    def reset(self) -> None:
        self._last_ns = None

    def wait(self) -> None:
        clock = self._clock
        if clock is None or not getattr(clock, "ros_time_is_active", False):
            return
        now = clock.now().nanoseconds
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
            now = clock.now().nanoseconds
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
    """LeRobot Robot that adapts a framework-neutral TopicBridge.

    Two modes:
        - Standalone: creates an internal BridgeLifecycleNode with its own
          executor and spin thread. Used when launched independently.
        - Injected: attaches to a pre-built TopicBridge on an external node (via
          config._external_bridge). Used by policy_runner_node so launch topic
          remappings apply to observation/action topics.
    """

    config_class = RosettaConfig
    name = "rosetta"

    def __init__(self, config: RosettaConfig):
        super().__init__(config)
        self._config: RosettaConfig = config

        # Standalone mode resources (unused in injected mode)
        self._host = NodeHost()

        # Injected mode: pre-built bridge from external node
        self._external_bridge: Optional[TopicBridge] = getattr(config, "_external_bridge", None)
        self._bridge: Optional[TopicBridge] = None

        # Sim pacing: created in connect(), no-ops unless ROS time is active
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
        if self._external_bridge is not None:
            return self._bridge is not None
        if self._node is None:
            return False
        return self._node.is_active

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        # Kept for interface compatibility. Lifecycle state is authoritative.
        del value  # unused

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        """Trigger lifecycle configure transition (standalone mode only)."""
        if self._external_bridge is not None:
            return  # bridge already configured by external node
        if self._node is None:
            self._create_node()
        require_transition_success(self._node.trigger_configure(), "configure")

    def connect(self, calibrate: bool = True) -> None:
        """Configure (if needed) and activate the lifecycle node."""
        del calibrate  # unused; ROS2 robot needs no calibration
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Live LeRobot cannot represent >1 numeric observation key (or >1
        # action key); fail loudly here instead of silently recording/serving
        # lumped features.
        ensure_live_observation_layout(self.config.observation_specs, self.config.action_specs)

        if self._external_bridge is not None:
            # Injected mode: bridge already set up and activated by external node
            self._bridge = self._external_bridge
            self._make_gates(getattr(self._bridge, "clock", None))
            self._wait_for_warmup(self._bridge)
            return

        # Standalone mode
        if self._node is None:
            self._create_node()

        # Auto-configure if unconfigured (no publishers/subscriptions yet).
        # trigger_* returns the transition's result rather than raising, so a
        # failed configure would otherwise leave connect() reporting success
        # against a node with no publishers -- surfacing later as a warmup
        # timeout that names the wrong cause.
        if not self._node.is_configured:
            require_transition_success(self._node.trigger_configure(), "configure")

        require_transition_success(self._node.trigger_activate(), "activate")
        self._make_gates(self._node.get_clock())
        self._wait_for_warmup(self._node.bridge)

    def _make_gates(self, clock) -> None:
        """Create the per-loop sim-pacing gates (no-ops under wall clock).

        Two independent gates: LeRobot's control loop skips get_observation
        when the action queue is deep and send_action when it is empty, so a
        shared gate would halve the rate when one iteration does both.
        """
        period_ns = int(1e9 / self.config.fps)
        stop = getattr(self._config, "_stop_event", None)
        self._obs_gate = _TickGate(clock, period_ns, stop)
        self._act_gate = _TickGate(clock, period_ns, stop)
        if getattr(clock, "ros_time_is_active", False):
            logging.info(
                "Pacing observations/actions at %d fps on ROS sim time (a silent /clock will block gated calls).",
                self.config.fps,
            )

    def _wait_for_warmup(self, bridge: TopicBridge) -> None:
        """Block until every observation stream has delivered a message.

        The bag porter skips warmup ticks, so ported datasets never contain
        a cold bridge's zero-filled frames; gate the live side on the same
        predicate (bridge.warmed_up) so first frames agree. Fail open with a
        warning after the timeout — the missing-stream zero-fill/watchdog
        machinery covers a stream that stays silent.
        """
        deadline = time.monotonic() + WARMUP_TIMEOUT_SEC
        while time.monotonic() < deadline:
            if bridge.warmed_up:
                return
            time.sleep(WARMUP_POLL_SEC)
        logging.warning(
            "Observation streams not warmed up after %.1fs; first frames may "
            "contain zero-fill (a ported dataset would have skipped them).",
            WARMUP_TIMEOUT_SEC,
        )

    def _create_node(self) -> None:
        """Create the lifecycle node and start the spin thread (NodeHost)."""
        overrides = [Parameter("use_sim_time", Parameter.Type.BOOL, True)] if self.config.use_sim_time else None
        self._host.start(
            lambda ctx: BridgeLifecycleNode(
                f"rosetta_{self.config.id}",
                self.config.observation_specs,
                self.config.action_specs,
                self.config.fps,
                context=ctx,
                parameter_overrides=overrides,
            )
        )

    @property
    def _node(self) -> Optional[BridgeLifecycleNode]:
        return self._host.node

    def disconnect(self) -> None:
        """Deactivate and cleanup."""
        if self._external_bridge is not None:
            # Injected mode: detach from bridge, reset state for next episode.
            # Do NOT tear down. The bridge is owned by the external node.
            if self._bridge is not None:
                # Send safety action before disconnecting to stop robot movement
                self._bridge.send_safety_action()
                self._bridge.reset_state()
                self._bridge = None
            return

        # Standalone mode
        if self._node is None:
            return

        if self._node.is_active:
            self._node.trigger_deactivate()

        if self._node.is_configured:
            self._node.trigger_cleanup()

        self._host.stop()

    def reset(self) -> None:
        """Reset internal state tracking (e.g., between episodes)."""
        if self._bridge is not None:
            self._bridge.reset_state()
        elif self._node is not None:
            self._node.bridge.reset_state()
        for gate in (self._obs_gate, self._act_gate):
            if gate is not None:
                gate.reset()

    # -------------------- Observation / Action --------------------

    def get_observation(self) -> RobotObservation:
        """
        Get current observations.

        Returns a RobotObservation with state values as individual namespaced
        floats and images by short key. Missing data is zero-filled by the bridge.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self._obs_gate is not None:
            self._obs_gate.wait()
        src = self._bridge if self._bridge is not None else self._node.bridge
        return self._flatten_observation(src.sample_frame())

    def send_action(self, action: RobotAction) -> RobotAction:
        """
        Send action to ROS2 topics.

        action is keyed by namespaced selector names. Returns the values sent.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self._act_gate is not None:
            self._act_gate.wait()
        src = self._bridge if self._bridge is not None else self._node.bridge
        src.publish_frame(self._unflatten_action(action))
        # Echo back the values sent, as the namespaced floats LeRobot recorded.
        return {name: action[name] for names in self._act_vector_names.values() for name in names}

    @property
    def config(self) -> RosettaConfig:
        return self._config

    @config.setter
    def config(self, value: RosettaConfig) -> None:
        self._config = value
