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
RosettaTeleop: LeRobot Teleoperator implementation for ROS2 with lifecycle support.

Bridges ROS2 topics to LeRobot's Teleoperator interface for teleoperation.
Uses ROS 2 lifecycle nodes for managed state transitions.

Lifecycle states:
    - Unconfigured: Node exists, no subscriptions/publishers
    - Inactive: Subscriptions active (buffering), publishers disabled
    - Active: Processing inputs, sending feedback
"""

from __future__ import annotations

import threading
from functools import partial
from typing import Any, Optional

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import Node
from rclpy.lifecycle import Publisher
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn
from rosidl_runtime_py.utilities import get_message

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents

from .config_rosetta_teleop import RosettaTeleopConfig
from rosetta.common.converters import decode_value, encode_value
from rosetta.common.contract_utils import (
    StreamBuffer,
    get_namespaced_names,
)
from rosetta.common.ros2_utils import dot_get, get_message_timestamp_ns, qos_profile_from_dict
from rosetta.common import decoders as _decoders  # noqa: F401 - registrers decoders
from rosetta.common import encoders as _encoders  # noqa: F401 - registrers encoders

# Ensure side-effect imports are not removed by optimizers
del _decoders, _encoders

# Internal timing constants
SPIN_TIMEOUT_SEC = 0.01
THREAD_JOIN_TIMEOUT_SEC = 1.0

EVENT_NAME_TO_ENUM = {
    "is_intervention": TeleopEvents.IS_INTERVENTION,
    "success": TeleopEvents.SUCCESS,
    "terminate_episode": TeleopEvents.TERMINATE_EPISODE,
    "rerecord_episode": TeleopEvents.RERECORD_EPISODE,
    "failure": TeleopEvents.FAILURE,
}


class _RosettaTeleopLifecycleNode(Node):
    """Internal lifecycle node for RosettaTeleop."""

    def __init__(self, node_name: str, config: RosettaTeleopConfig, **kwargs):
        self._config = config

        # These will be created in on_configure, None/empty indicates unconfigured
        # Following the demo pattern: check publisher existence and is_activated
        self._input_buffers: dict[str, tuple[Any, StreamBuffer]] = {}
        self._feedback_publishers: dict[str, tuple[Any, Publisher]] = {}
        self._subscriptions: list[Any] = []

        # Events state
        self._events_lock = threading.Lock()
        self._events_state: dict[TeleopEvents, bool] = {
            TeleopEvents.IS_INTERVENTION: False,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.RERECORD_EPISODE: False,
        }

        super().__init__(node_name, **kwargs)

    def on_configure(self, _state: State) -> TransitionCallbackReturn:
        """
        Configure the node, after a configuring transition is requested.

        Creates subscriptions for inputs and events (start buffering immediately)
        and lifecycle publishers for feedback (disabled until activate).
        """
        self.get_logger().info("on_configure() is called.")

        # Create regular subscriptions for inputs (start buffering immediately)
        for spec in self._config.input_specs:
            buffer = StreamBuffer.from_spec(spec)
            self._input_buffers[spec.topic] = (spec, buffer)
            sub = self.create_subscription(
                get_message(spec.msg_type),
                spec.topic,
                partial(self._on_input, spec=spec, buffer=buffer),
                qos_profile_from_dict(spec.qos) or 10,
            )
            self._subscriptions.append(sub)

        # Create regular subscription for events
        events_spec = self._config.events_spec
        if events_spec:
            sub = self.create_subscription(
                get_message(events_spec.msg_type),
                events_spec.topic,
                partial(self._on_events, spec=events_spec),
                qos_profile_from_dict(events_spec.qos) or 10,
            )
            self._subscriptions.append(sub)

        # Create lifecycle publishers for feedback (disabled until activate)
        for spec in self._config.feedback_specs:
            pub = self.create_lifecycle_publisher(
                get_message(spec.msg_type),
                spec.topic,
                qos_profile_from_dict(spec.qos) or 10,
            )
            self._feedback_publishers[spec.topic] = (spec, pub)

        self.get_logger().info(
            f"Configured: {len(self._config.input_specs)} inputs, "
            f"{len(self._config.feedback_specs)} feedback"
        )
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """
        Activate the node.

        The default LifecycleNode callback transitions LifecyclePublisher entities
        from inactive to enabled. We must call super().on_activate().
        """
        self.get_logger().info("on_activate() is called.")
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """
        Deactivate the node.

        Publishers are disabled by super().on_deactivate().
        """
        self.get_logger().info("on_deactivate() is called.")
        return super().on_deactivate(state)

    def on_cleanup(self, _state: State) -> TransitionCallbackReturn:
        """
        Cleanup the node, after a cleaning-up transition is requested.

        Destroys all subscriptions and publishers.
        """
        self.get_logger().info("on_cleanup() is called.")

        # Destroy subscriptions
        for sub in self._subscriptions:
            self.destroy_subscription(sub)
        self._subscriptions.clear()

        # Destroy publishers
        for _, pub in self._feedback_publishers.values():
            if pub is not None:
                self.destroy_publisher(pub)
        self._feedback_publishers.clear()

        # Clear buffers and state
        self._input_buffers.clear()
        with self._events_lock:
            for key in self._events_state:
                self._events_state[key] = False

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, _state: State) -> TransitionCallbackReturn:
        """
        Shutdown the node, after a shutting-down transition is requested.
        """
        self.get_logger().info("on_shutdown() is called.")

        for sub in self._subscriptions:
            self.destroy_subscription(sub)
        self._subscriptions.clear()

        for _, pub in self._feedback_publishers.values():
            if pub is not None:
                self.destroy_publisher(pub)
        self._feedback_publishers.clear()

        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state: State) -> TransitionCallbackReturn:
        """
        Handle errors by cleaning up resources.
        """
        self.get_logger().error(f"Error occurred in state: {state.label}")

        try:
            for sub in self._subscriptions:
                self.destroy_subscription(sub)
            self._subscriptions.clear()

            for _, pub in self._feedback_publishers.values():
                if pub is not None:
                    self.destroy_publisher(pub)
            self._feedback_publishers.clear()

            self._input_buffers.clear()
            with self._events_lock:
                for key in self._events_state:
                    self._events_state[key] = False
        except Exception as e:
            self.get_logger().error(f"Error during cleanup: {e}")

        return TransitionCallbackReturn.SUCCESS

    def _on_input(self, msg, spec, buffer) -> None:
        """Handle incoming input message."""
        fallback_ns = self.get_clock().now().nanoseconds
        ts_ns = get_message_timestamp_ns(msg, spec, fallback_ns)
        buffer.push(ts_ns, decode_value(msg, spec))

    def _on_events(self, msg, spec) -> None:
        """Handle incoming events message."""
        with self._events_lock:
            for event_name, selector in spec.mappings.items():
                if event_name not in EVENT_NAME_TO_ENUM:
                    continue
                try:
                    value = dot_get(msg, selector)
                    self._events_state[EVENT_NAME_TO_ENUM[event_name]] = bool(value)
                except (AttributeError, IndexError, ValueError):
                    pass

    def sample_action(self) -> dict[str, Any]:
        """Sample current action from input buffers."""
        action = {}
        now_ns = self.get_clock().now().nanoseconds

        for spec, buffer in self._input_buffers.values():
            data = buffer.sample(now_ns)
            if data is None:
                continue
            for i, name in enumerate(get_namespaced_names(spec)):
                action[name] = float(data[i])

        return action

    def get_events(self) -> dict[TeleopEvents, bool]:
        """Get current events state."""
        with self._events_lock:
            return self._events_state.copy()

    def publish_feedback(self, feedback: dict[str, Any]) -> None:
        """Publish feedback via lifecycle publishers."""
        stamp_ns = self.get_clock().now().nanoseconds

        for spec, pub in self._feedback_publishers.values():
            if pub is None:
                continue

            names = get_namespaced_names(spec)
            if not all(name in feedback for name in names):
                continue
            arr = np.array([feedback[name] for name in names], dtype=np.float32)
            msg = encode_value(spec, arr, stamp_ns)

            # Lifecycle publisher handles active/inactive state automatically
            pub.publish(msg)

    @property
    def is_active(self) -> bool:
        """
        Check if the node is in active state.

        Uses the lifecycle publisher's is_activated property (public API).
        This follows the pattern from the official ROS 2 lifecycle demo.
        """
        # Check any lifecycle publisher - they all transition together
        for _, pub in self._feedback_publishers.values():
            return pub.is_activated
        # No publishers configured - check subscriptions exist (configured but no feedback)
        return False

    @property
    def is_configured(self) -> bool:
        """
        Check if the node has been configured.

        Publishers/subscriptions are created in on_configure and cleared in on_cleanup.
        """
        # Publishers/subscriptions created in on_configure, cleared in on_cleanup
        return bool(self._feedback_publishers) or bool(self._subscriptions)


class RosettaTeleop(Teleoperator):
    """LeRobot Teleoperator that bridges to ROS2 topics with lifecycle support."""

    config_class = RosettaTeleopConfig
    name = "rosetta_teleop"

    def __init__(self, config: RosettaTeleopConfig):
        super().__init__(config)
        self.config = config
        self._calibrated = True

        self._node: Optional[_RosettaTeleopLifecycleNode] = None
        self._executor: Optional[SingleThreadedExecutor] = None
        self._spin_thread: Optional[threading.Thread] = None
        self._owns_rclpy = False

    @property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for spec in self.config.input_specs:
            for name in get_namespaced_names(spec):
                features[name] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for spec in self.config.feedback_specs:
            for name in get_namespaced_names(spec):
                features[name] = float
        return features

    @property
    def is_connected(self) -> bool:
        """Returns True only when lifecycle state is ACTIVE."""
        if self._node is None:
            return False
        return self._node.is_active

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        """Trigger lifecycle configure transition."""
        if self._node is None:
            self._create_node()
        self._node.trigger_configure()

    def connect(self, calibrate: bool = True) -> None:
        """Configure (if needed) and activate the lifecycle node."""
        del calibrate  # Unused - ROS2 teleop doesn't require calibration
        if self.is_connected:
            return

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

        self._node = _RosettaTeleopLifecycleNode(f"rosetta_teleop_{self.id}", self.config)
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

    def get_action(self) -> dict[str, Any]:
        """Get current action from input buffers."""
        if not self.is_connected:
            return {}
        return self._node.sample_action()

    def get_teleop_events(self) -> dict[TeleopEvents, bool]:
        """Get current teleop events state."""
        if self._node is None:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }
        return self._node.get_events()

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Send feedback to ROS2 topics."""
        if not self.is_connected:
            return
        self._node.publish_feedback(feedback)

    def disconnect(self) -> None:
        """Deactivate and cleanup the lifecycle node."""
        if self._node is None:
            return

        # Deactivate if active
        if self._node.is_active:
            self._node.trigger_deactivate()

        # Cleanup if configured (inactive state)
        if self._node.is_configured:
            self._node.trigger_cleanup()

        self._destroy_node()
