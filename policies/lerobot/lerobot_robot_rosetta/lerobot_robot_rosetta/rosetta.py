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
Rosetta: LeRobot Robot implementation for ROS2 with lifecycle support.

Bridges ROS2 topics to LeRobot's Robot interface for recording and inference.
Uses ROS 2 lifecycle nodes for managed state transitions.

Usage:
    from rosetta import Rosetta, RosettaConfig

    config = RosettaConfig(config_path="contract.yaml")
    robot = Rosetta(config)
    robot.connect()

    obs = robot.get_observation()  # Returns zeros if sensor data missing
    robot.send_action({"position.joint1": 0.1, "position.joint2": 0.2})

    robot.disconnect()

Architecture:
    _TopicBridge: Manages observation subscriptions, lifecycle action publishers,
        watchdog timer, and data buffering on a given LifecycleNode.
    _RosettaLifecycleNode: Thin lifecycle wrapper around _TopicBridge for
        standalone mode (creates its own node + executor + spin thread).
    Rosetta: LeRobot Robot interface with two modes:
        - Standalone: creates _RosettaLifecycleNode internally (current behavior)
        - Injected: attaches to a pre-built _TopicBridge on an external node
          (used by RosettaClientNode for launch remapping support)
"""

from __future__ import annotations

import threading
from functools import cached_property, partial
from typing import Any, Optional

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import Node
from rclpy.lifecycle import Publisher
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.timer import Timer
from rosidl_runtime_py.utilities import get_message

from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_rosetta import RosettaConfig
from rosetta.common.contract import ObservationStreamSpec
from rosetta.common.converters import decode_value, encode_value
from rosetta.common.contract_utils import (
    StreamBuffer,
    zeros_for_spec,
    get_namespaced_names,
)
from rosetta.common.ros2_utils import get_message_timestamp_ns, qos_profile_from_dict
from rosetta.common import decoders as _decoders  # noqa: F401 - registers decoders
from rosetta.common import encoders as _encoders  # noqa: F401 - registers encoders

# Ensure side-effect imports are not removed by optimizers
del _decoders, _encoders

# Internal timing constants
SPIN_TIMEOUT_SEC = 0.01
THREAD_JOIN_TIMEOUT_SEC = 1.0


class _TopicBridge:
    """Manages observation subscriptions, action publishers, and watchdog on a LifecycleNode.

    This is a plain Python object (not a Node). It creates ROS2 entities
    (subscriptions, lifecycle publishers, timers) on a host LifecycleNode via
    ``setup()``, and destroys them via ``teardown()``.

    Lifecycle publishers are activated/deactivated atomically by the host node's
    lifecycle transitions (``super().on_activate()`` / ``super().on_deactivate()``).
    The publisher's ``is_activated`` property gates publishing automatically.
    """

    def __init__(self, config: RosettaConfig):
        self._config = config

        # Created in setup(), cleared in teardown()
        self._obs_buffers: dict[str, tuple[ObservationStreamSpec, StreamBuffer]] = {}
        self._act_publishers: dict[str, tuple[Any, Publisher]] = {}
        self._subscriptions: list[Any] = []
        self._watchdog_timer: Optional[Timer] = None

        # Stream state tracking
        self._missing_streams: set[str] = set()
        self._header_warned: set[str] = set()

        # Safety state
        self._last_action_ns: Optional[int] = None
        self._last_sent: dict[str, np.ndarray] = {}

        # Reference to the host node (set in setup, cleared in teardown)
        self._node: Optional[Any] = None

    def setup(self, node) -> None:
        """Create subscriptions, lifecycle publishers, and watchdog on the given node.

        Subscriptions start buffering immediately. Lifecycle publishers are
        created in the inactive state and will be enabled when the host node
        transitions to active via ``super().on_activate()``.

        Args:
            node: A ``rclpy.lifecycle.Node`` (LifecycleNode).
        """
        self._node = node

        # Create subscriptions (start buffering immediately)
        for spec in self._config.observation_specs:
            buffer = StreamBuffer.from_spec(spec)
            self._obs_buffers[spec.topic] = (spec, buffer)
            callback = partial(self._on_observation, spec=spec, buffer=buffer)
            sub = node.create_subscription(
                get_message(spec.msg_type),
                spec.topic,
                callback,
                qos_profile_from_dict(spec.qos) or 10,
            )
            self._subscriptions.append(sub)

        # Create lifecycle publishers (disabled until host node activates)
        for spec in self._config.action_specs:
            pub = node.create_lifecycle_publisher(
                get_message(spec.msg_type),
                spec.topic,
                qos_profile_from_dict(spec.qos) or 10,
            )
            self._act_publishers[spec.topic] = (spec, pub)

        # Create watchdog timer
        if self._should_use_watchdog():
            period_sec = 2.0 / self._config.fps
            self._watchdog_timer = node.create_timer(period_sec, self._on_watchdog)

        node.get_logger().info(
            f"TopicBridge: {len(self._config.observation_specs)} obs, "
            f"{len(self._config.action_specs)} act @ {self._config.fps}Hz"
        )

    def teardown(self) -> None:
        """Destroy all ROS2 resources on the host node."""
        node = self._node
        if node is None:
            return

        if self._watchdog_timer is not None:
            node.destroy_timer(self._watchdog_timer)
            self._watchdog_timer = None

        for sub in self._subscriptions:
            node.destroy_subscription(sub)
        self._subscriptions.clear()

        for _, pub in self._act_publishers.values():
            if pub is not None:
                node.destroy_publisher(pub)
        self._act_publishers.clear()

        self._obs_buffers.clear()
        self._missing_streams.clear()
        self._header_warned.clear()
        self._last_sent.clear()
        self._last_action_ns = None
        self._node = None

    def send_safety_action(self) -> None:
        """Publish safety action (zeros or hold) per spec's safety_behavior.

        Only publishes on activated lifecycle publishers.
        """
        if self._node is None:
            return
        stamp_ns = self._node.get_clock().now().nanoseconds
        for topic, (spec, pub) in self._act_publishers.items():
            if pub is None or not pub.is_activated:
                continue
            if spec.safety_behavior == "none":
                continue
            if spec.safety_behavior == "hold" and topic in self._last_sent:
                arr = self._last_sent[topic]
            else:
                arr = np.zeros(len(spec.names), dtype=np.float32)
            msg = encode_value(spec, arr, stamp_ns)
            pub.publish(msg)

    def reset_state(self) -> None:
        """Reset internal state tracking (e.g., between episodes)."""
        self._missing_streams.clear()
        self._header_warned.clear()
        self._last_action_ns = None

    # -------------------- Properties --------------------

    @property
    def is_active(self) -> bool:
        """Check if lifecycle publishers are activated."""
        for _, pub in self._act_publishers.values():
            return pub.is_activated
        return False

    @property
    def is_configured(self) -> bool:
        """Check if setup() has been called (resources exist)."""
        return bool(self._act_publishers) or bool(self._subscriptions)

    # -------------------- Observation / Action --------------------

    def sample_observation(self) -> RobotObservation:
        """Sample current observations from buffers."""
        obs = {}
        now_ns = self._node.get_clock().now().nanoseconds

        for spec, buffer in self._obs_buffers.values():
            data = buffer.sample(now_ns)
            self._log_stream_state(spec.key, data is None)

            if spec.is_image:
                key = spec.key.removeprefix("observation.images.")
                if data is not None:
                    obs[key] = data  # (H, W, C) uint8 from decode_value()
                else:
                    obs[key] = zeros_for_spec(spec)
            else:
                names = get_namespaced_names(spec)
                for i, name in enumerate(names):
                    obs[name] = float(data[i]) if data is not None else 0.0

        return obs

    def publish_action(self, action: RobotAction) -> RobotAction:
        """Publish action to ROS2 topics via lifecycle publishers."""
        sent = {}
        stamp_ns = self._node.get_clock().now().nanoseconds

        for topic, (spec, pub) in self._act_publishers.items():
            if pub is None:
                continue

            names = get_namespaced_names(spec)
            arr = np.array([action[name] for name in names], dtype=np.float32)
            msg = encode_value(spec, arr, stamp_ns)

            # Lifecycle publisher handles active/inactive state automatically
            pub.publish(msg)
            self._last_sent[topic] = arr

            for name in names:
                sent[name] = action[name]

        self._last_action_ns = stamp_ns
        return sent

    # -------------------- Private --------------------

    def _should_use_watchdog(self) -> bool:
        """Check if watchdog should be enabled."""
        if not self._act_publishers:
            return False
        return not all(
            spec.safety_behavior == "none" for spec, _ in self._act_publishers.values()
        )

    def _on_watchdog(self) -> None:
        """Check if actions have stopped and send safety action if needed."""
        if not self.is_active:
            return
        if self._last_action_ns is None:
            return

        now_ns = self._node.get_clock().now().nanoseconds
        timeout_ns = int(2e9 / self._config.fps)  # 2 frame periods

        if now_ns - self._last_action_ns > timeout_ns:
            self._node.get_logger().warning("Action timeout - sending safety action")
            self.send_safety_action()
            self._last_action_ns = None

    def _on_observation(
        self, msg, spec: ObservationStreamSpec, buffer: StreamBuffer
    ) -> None:
        """Handle incoming observation message: extract timestamp, decode, and buffer."""
        fallback_ns = self._node.get_clock().now().nanoseconds
        ts_ns = get_message_timestamp_ns(msg, spec, fallback_ns)

        # Log warning on first header fallback per stream
        if spec.stamp_src == "header" and ts_ns == fallback_ns:
            if spec.key not in self._header_warned:
                self._node.get_logger().warning(
                    f"Header stamp unavailable for '{spec.key}', using receive time"
                )
                self._header_warned.add(spec.key)

        buffer.push(ts_ns, decode_value(msg, spec))

    def _log_stream_state(self, key: str, is_missing: bool) -> None:
        """Log on state transitions only (missing <-> recovered)."""
        was_missing = key in self._missing_streams
        if is_missing and not was_missing:
            self._node.get_logger().warning(f"Stream '{key}' missing - using zeros")
            self._missing_streams.add(key)
        elif not is_missing and was_missing:
            self._node.get_logger().info(f"Stream '{key}' recovered")
            self._missing_streams.discard(key)


class _RosettaLifecycleNode(Node):
    """Thin lifecycle wrapper around _TopicBridge for standalone mode."""

    def __init__(self, node_name: str, config: RosettaConfig, **kwargs):
        super().__init__(node_name, **kwargs)
        self._bridge = _TopicBridge(config)

    def on_configure(self, _state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_configure() is called.")
        self._bridge.setup(self)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_activate() is called.")
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_deactivate() is called.")
        self._bridge.send_safety_action()
        self._bridge._last_action_ns = None
        return super().on_deactivate(state)

    def on_cleanup(self, _state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_cleanup() is called.")
        self._bridge.teardown()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, _state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_shutdown() is called.")
        self._bridge.teardown()
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().error(f"Error occurred in state: {state.label}")
        try:
            self._bridge.teardown()
        except Exception as e:
            self.get_logger().error(f"Error during cleanup: {e}")
        return TransitionCallbackReturn.SUCCESS

    @property
    def is_active(self) -> bool:
        return self._bridge.is_active

    @property
    def is_configured(self) -> bool:
        return self._bridge.is_configured

    def sample_observation(self) -> RobotObservation:
        return self._bridge.sample_observation()

    def publish_action(self, action: RobotAction) -> RobotAction:
        return self._bridge.publish_action(action)

    def reset_state(self) -> None:
        self._bridge.reset_state()


class Rosetta(Robot):
    """LeRobot Robot that bridges to ROS2 topics with lifecycle support.

    Supports two modes:
        - Standalone: creates an internal _RosettaLifecycleNode with its own
          executor and spin thread. Used when launched independently.
        - Injected: attaches to a pre-built _TopicBridge on an external node
          (via config._external_bridge). Used by RosettaClientNode so that
          launch topic remappings apply to observation/action topics.
    """

    config_class = RosettaConfig
    name = "rosetta"

    def __init__(self, config: RosettaConfig):
        super().__init__(config)
        self._config: RosettaConfig = config

        # Standalone mode resources (None in injected mode)
        self._node: Optional[_RosettaLifecycleNode] = None
        self._executor: Optional[SingleThreadedExecutor] = None
        self._spin_thread: Optional[threading.Thread] = None
        self._owns_rclpy = False

        # Injected mode: pre-built bridge from external node
        self._external_bridge: Optional[_TopicBridge] = getattr(
            config, "_external_bridge", None
        )
        self._bridge: Optional[_TopicBridge] = None

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
        # Setter maintained for interface compatibility but lifecycle state is authoritative
        del value  # Unused - lifecycle state is authoritative

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        """Trigger lifecycle configure transition (standalone mode only)."""
        if self._external_bridge is not None:
            return  # Bridge already configured by external node
        if self._node is None:
            self._create_node()
        self._node.trigger_configure()

    def connect(self, calibrate: bool = True) -> None:
        """Configure (if needed) and activate the lifecycle node."""
        del calibrate  # Unused - ROS2 robot doesn't require calibration
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        if self._external_bridge is not None:
            # Injected mode: bridge already setup + activated by external node
            self._bridge = self._external_bridge
            return

        # Standalone mode
        if self._node is None:
            self._create_node()

        # Auto-configure if unconfigured (no publishers/subscriptions yet)
        if not self._node.is_configured:
            self._node.trigger_configure()

        # Activate
        self._node.trigger_activate()

    def _create_node(self) -> None:
        """Create the lifecycle node and start the spin thread."""
        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True

        self._node = _RosettaLifecycleNode(
            f"rosetta_{self.config.id}", self._config,
            # Ignore global CLI args (e.g. __node:= remap from the launch system)
            # so this internal node keeps its own name and services separate from
            # the launch-managed lifecycle node. Side-effect: --log-level from the
            # launch CLI won't apply to this node (it uses the default level).
            use_global_arguments=False,
        )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)

        # Start spin thread BEFORE lifecycle transitions (required for service calls)
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
        self._node = None  # Signal spin loop to stop

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
            # Do NOT teardown — bridge is owned by the external node.
            if self._bridge is not None:
                self._bridge.reset_state()
                self._bridge = None
            return

        # Standalone mode
        if self._node is None:
            return

        # Deactivate if active
        if self._node.is_active:
            self._node.trigger_deactivate()

        # Cleanup if configured (inactive state)
        if self._node.is_configured:
            self._node.trigger_cleanup()

        self._destroy_node()

    def reset(self) -> None:
        """Reset internal state tracking (e.g., between episodes)."""
        if self._bridge is not None:
            self._bridge.reset_state()
        elif self._node is not None:
            self._node.reset_state()

    def get_observation(self) -> RobotObservation:
        """
        Get current observations.

        Returns:
            RobotObservation with state values as individual floats (namespaced), images by short key.
            Missing data is replaced with zeros (logged on state transition).
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self._bridge is not None:
            return self._bridge.sample_observation()
        return self._node.sample_observation()

    def send_action(self, action: RobotAction) -> RobotAction:
        """
        Send action to ROS2 topics.

        Args:
            action: RobotAction dict with individual values keyed by namespaced selector names.

        Returns:
            RobotAction dict of sent values.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self._bridge is not None:
            return self._bridge.publish_action(action)
        return self._node.publish_action(action)

    @property
    def config(self) -> RosettaConfig:
        return self._config

    @config.setter
    def config(self, value: RosettaConfig) -> None:
        self._config = value
