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
Rosetta: LeRobot Robot adapter over the backend-neutral TopicBridge.

The ROS2 plumbing (subscriptions, lifecycle publishers, watchdog, resampling)
lives in rosetta.ros2.topic_bridge. This module presents that bridge as a LeRobot
Robot. It translates between the frame dict ({contract_key: np.ndarray | str}) and
LeRobot's flattened observation/action shape: images by short key, state and
action as individual namespaced floats.

Two modes:
    - Standalone: creates a RosettaLifecycleNode internally (own node, executor,
      spin thread).
    - Injected: attaches to a pre-built TopicBridge on an external node (via
      config._external_bridge). Used by rosetta_client_node so launch topic
      remappings apply.
"""

from __future__ import annotations

import threading
from functools import cached_property
from typing import Optional

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor

from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from rosetta.core.contract_utils import get_namespaced_names
from rosetta.ros2.rosetta_lifecycle_node import RosettaLifecycleNode
from rosetta.ros2.topic_bridge import TopicBridge

from .config_rosetta import RosettaConfig

# Internal timing constants
SPIN_TIMEOUT_SEC = 0.01
THREAD_JOIN_TIMEOUT_SEC = 1.0


class Rosetta(Robot):
    """LeRobot Robot that adapts a backend-neutral TopicBridge.

    Two modes:
        - Standalone: creates an internal RosettaLifecycleNode with its own
          executor and spin thread. Used when launched independently.
        - Injected: attaches to a pre-built TopicBridge on an external node (via
          config._external_bridge). Used by rosetta_client_node so launch topic
          remappings apply to observation/action topics.
    """

    config_class = RosettaConfig
    name = "rosetta"

    def __init__(self, config: RosettaConfig):
        super().__init__(config)
        self._config: RosettaConfig = config

        # Standalone mode resources (None in injected mode)
        self._node: Optional[RosettaLifecycleNode] = None
        self._executor: Optional[SingleThreadedExecutor] = None
        self._spin_thread: Optional[threading.Thread] = None
        self._owns_rclpy = False

        # Injected mode: pre-built bridge from external node
        self._external_bridge: Optional[TopicBridge] = getattr(
            config, "_external_bridge", None
        )
        self._bridge: Optional[TopicBridge] = None

    # -------------------- LeRobot <-> frame-dict adapters --------------------

    @cached_property
    def _obs_vector_names(self) -> dict[str, list[str]]:
        """Non-image observation key -> ordered namespaced selector names."""
        m: dict[str, list[str]] = {}
        for spec in self.config.observation_specs:
            if spec.is_image:
                continue
            m.setdefault(spec.key, []).extend(get_namespaced_names(spec))
        return m

    @cached_property
    def _obs_image_keys(self) -> dict[str, str]:
        """Image observation full key -> LeRobot short key."""
        return {
            spec.key: spec.key.removeprefix("observation.images.")
            for spec in self.config.observation_specs
            if spec.is_image
        }

    @cached_property
    def _act_vector_names(self) -> dict[str, list[str]]:
        """Action key -> ordered namespaced selector names."""
        m: dict[str, list[str]] = {}
        for spec in self.config.action_specs:
            m.setdefault(spec.key, []).extend(get_namespaced_names(spec))
        return m

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
        """Feature spec: individual state values as float, images as (H, W, C) tuples."""
        features: dict[str, type | tuple] = {}
        for spec in self.config.observation_specs:
            if spec.is_image:
                key = spec.key.removeprefix("observation.images.")
                h, w = spec.image_resize
                features[key] = (h, w, spec.image_channels)
            else:
                for name in get_namespaced_names(spec):
                    features[name] = float
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Feature spec: individual action values as float."""
        features: dict[str, type] = {}
        for spec in self.config.action_specs:
            for name in get_namespaced_names(spec):
                features[name] = float
        return features

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
        self._node.trigger_configure()

    def connect(self, calibrate: bool = True) -> None:
        """Configure (if needed) and activate the lifecycle node."""
        del calibrate  # unused; ROS2 robot needs no calibration
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        if self._external_bridge is not None:
            # Injected mode: bridge already set up and activated by external node
            self._bridge = self._external_bridge
            return

        # Standalone mode
        if self._node is None:
            self._create_node()

        # Auto-configure if unconfigured (no publishers/subscriptions yet)
        if not self._node.is_configured:
            self._node.trigger_configure()

        self._node.trigger_activate()

    def _create_node(self) -> None:
        """Create the lifecycle node and start the spin thread."""
        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True

        self._node = RosettaLifecycleNode(
            f"rosetta_{self.config.id}",
            self.config.observation_specs,
            self.config.action_specs,
            self.config.fps,
        )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)

        # Start spin thread before lifecycle transitions (needed for service calls)
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

    def _spin_loop(self) -> None:
        """Spin the executor until node is destroyed."""
        while self._executor is not None and self._node is not None:
            try:
                self._executor.spin_once(timeout_sec=SPIN_TIMEOUT_SEC)
            except Exception:
                if self._node is None:
                    break
                raise

    def _destroy_node(self) -> None:
        """Clean up node, executor, and spin thread."""
        node = self._node
        self._node = None  # signal spin loop to stop

        if self._spin_thread is not None:
            self._spin_thread.join(timeout=THREAD_JOIN_TIMEOUT_SEC)
            self._spin_thread = None

        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

        if node is not None:
            node.destroy_node()

        if self._owns_rclpy:
            rclpy.try_shutdown()
            self._owns_rclpy = False

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

        self._destroy_node()

    def reset(self) -> None:
        """Reset internal state tracking (e.g., between episodes)."""
        if self._bridge is not None:
            self._bridge.reset_state()
        elif self._node is not None:
            self._node.reset_state()

    # -------------------- Observation / Action --------------------

    def get_observation(self) -> RobotObservation:
        """
        Get current observations.

        Returns a RobotObservation with state values as individual namespaced
        floats and images by short key. Missing data is zero-filled by the bridge.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        src = self._bridge if self._bridge is not None else self._node
        return self._flatten_observation(src.sample_frame())

    def send_action(self, action: RobotAction) -> RobotAction:
        """
        Send action to ROS2 topics.

        action is keyed by namespaced selector names. Returns the values sent.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        src = self._bridge if self._bridge is not None else self._node
        src.publish_frame(self._unflatten_action(action))
        # Echo back the values sent, as the namespaced floats LeRobot recorded.
        return {
            name: action[name]
            for names in self._act_vector_names.values()
            for name in names
        }

    @property
    def config(self) -> RosettaConfig:
        return self._config

    @config.setter
    def config(self, value: RosettaConfig) -> None:
        self._config = value
