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
RosettaTeleop: LeRobot Teleoperator that adapts a framework-neutral TopicBridge.

Teleop inputs are observation streams and feedback are action streams of the
same TopicBridge machinery the robot adapter uses — shared message ingest,
lifecycle publishers, and teardown come from the bridge. Only the events
subscription and the name-flattening layer live here. Feedback specs always
carry safety_behavior='none' (enforced at contract parse), so the bridge's
watchdog stays disabled.

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
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from rclpy.lifecycle import Node, State, TransitionCallbackReturn
from rosetta.frames.layout import FrameLayout, get_namespaced_names
from rosetta.robots.ros2.node_host import NodeHost
from rosetta.robots.ros2.ros2_utils import (
    LIFECYCLE_CONFIGURED_LABELS,
    dot_get,
    lifecycle_state_label,
    qos_profile_from_dict,
)
from rosetta.robots.ros2.topic_bridge import TopicBridge
from rosidl_runtime_py.utilities import get_message

from .config_rosetta_teleop import RosettaTeleopConfig

EVENT_NAME_TO_ENUM = {
    "is_intervention": TeleopEvents.IS_INTERVENTION,
    "success": TeleopEvents.SUCCESS,
    "terminate_episode": TeleopEvents.TERMINATE_EPISODE,
    "rerecord_episode": TeleopEvents.RERECORD_EPISODE,
    "failure": TeleopEvents.FAILURE,
}

# One source of truth for the event key set. (The old per-site dicts dropped
# FAILURE, so the returned key set mutated after the first failure message.)
DEFAULT_EVENTS: dict[TeleopEvents, bool] = dict.fromkeys(EVENT_NAME_TO_ENUM.values(), False)


class _RosettaTeleopLifecycleNode(Node):
    """Lifecycle node hosting a TopicBridge for teleop inputs and feedback."""

    def __init__(self, node_name: str, config: RosettaTeleopConfig, **kwargs):
        self._config = config
        self.bridge = TopicBridge(config.input_specs, config.feedback_specs, config.fps)
        self._feedback_layout = FrameLayout(config.feedback_specs)
        self._events_sub = None

        self._events_lock = threading.Lock()
        self._events_state: dict[TeleopEvents, bool] = dict(DEFAULT_EVENTS)

        super().__init__(node_name, **kwargs)

    # -------------------- lifecycle --------------------

    def on_configure(self, _state: State) -> TransitionCallbackReturn:
        """Set up the bridge plus the events subscription.

        Bridge setup covers input subscriptions and feedback lifecycle
        publishers.
        """
        self.bridge.setup(self)

        events_spec = self._config.events_spec
        if events_spec:
            self._events_sub = self.create_subscription(
                get_message(events_spec.channel.type),
                events_spec.channel.topic,
                partial(self._on_events, spec=events_spec),
                qos_profile_from_dict(events_spec.channel.qos) or 10,
            )

        self.get_logger().info(
            f"Configured: {len(self._config.input_specs)} inputs, {len(self._config.feedback_specs)} feedback"
        )
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        # super() enables the bridge's lifecycle publishers.
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        # super() disables the bridge's lifecycle publishers.
        return super().on_deactivate(state)

    def _teardown(self) -> None:
        """Destroy everything on_configure created; reset event state."""
        self.bridge.teardown()
        if self._events_sub is not None:
            self.destroy_subscription(self._events_sub)
            self._events_sub = None
        with self._events_lock:
            self._events_state = dict(DEFAULT_EVENTS)

    def on_cleanup(self, _state: State) -> TransitionCallbackReturn:
        self._teardown()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, _state: State) -> TransitionCallbackReturn:
        self._teardown()
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().error(f"Error occurred in state: {state.label}")
        try:
            self._teardown()
        except Exception as e:
            self.get_logger().error(f"Error during cleanup: {e}")
        return TransitionCallbackReturn.SUCCESS

    # -------------------- teleop surface --------------------

    def _on_events(self, msg, spec) -> None:
        """Handle incoming events message."""
        with self._events_lock:
            for event_name, selector in spec.select.items():
                if event_name not in EVENT_NAME_TO_ENUM:
                    continue
                try:
                    value = dot_get(msg, selector)
                    self._events_state[EVENT_NAME_TO_ENUM[event_name]] = bool(value)
                except (AttributeError, IndexError, ValueError):
                    pass

    def sample_action(self) -> dict[str, Any]:
        """Flatten per-spec input samples to {namespaced_name: float}.

        Streams with no data yet are omitted (not zero-filled): a fabricated
        zero teleop action could command motion.
        """
        action: dict[str, Any] = {}
        for spec, value in zip(self._config.input_specs, self.bridge.sample_values(), strict=False):
            if value is None:
                continue
            for i, name in enumerate(get_namespaced_names(spec)):
                action[name] = float(value[i])
        return action

    def get_events(self) -> dict[TeleopEvents, bool]:
        """Get current events state."""
        with self._events_lock:
            return self._events_state.copy()

    def publish_feedback(self, feedback: dict[str, Any]) -> None:
        """Regroup {namespaced_name: value} into key vectors and publish.

        Publishes only when every feedback name is present — publish_frame
        needs the complete per-key layout.
        """
        frame: dict[str, Any] = {}
        for key in self._feedback_layout.keys:
            names = [name for sl in self._feedback_layout[key].slices for name in get_namespaced_names(sl.spec)]
            if not all(name in feedback for name in names):
                return
            frame[key] = np.array([feedback[name] for name in names], dtype=np.float64)
        if frame:
            self.bridge.publish_frame(frame)

    @property
    def is_active(self) -> bool:
        """True when the lifecycle state machine is in 'active'.

        Reads the authoritative state machine (shared helper), NOT publisher
        is_activated flags: an input-only teleop (feedback: []) has no
        publishers, and a publisher-based check would report such a node as
        never active — making the teleop unusable.
        """
        return lifecycle_state_label(self) == "active"

    @property
    def is_configured(self) -> bool:
        """True once configured (inactive/active/transitioning)."""
        return lifecycle_state_label(self) in LIFECYCLE_CONFIGURED_LABELS


class RosettaTeleop(Teleoperator):
    """LeRobot Teleoperator that bridges to ROS2 topics with lifecycle support."""

    config_class = RosettaTeleopConfig
    name = "rosetta_teleop"

    def __init__(self, config: RosettaTeleopConfig):
        super().__init__(config)
        self.config = config
        self._calibrated = True
        self._host = NodeHost()

    @property
    def _node(self) -> Optional[_RosettaTeleopLifecycleNode]:
        return self._host.node

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
        return self._node is not None and self._node.is_active

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        """Trigger lifecycle configure transition."""
        self._start_node().trigger_configure()

    def connect(self, calibrate: bool = True) -> None:
        """Configure (if needed) and activate the lifecycle node.

        Raises DeviceAlreadyConnectedError when already connected — the same
        convention as the robot adapter (and lerobot's own devices).
        """
        del calibrate  # Unused - ROS2 teleop doesn't require calibration
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        node = self._start_node()
        if not node.is_configured:
            node.trigger_configure()
        node.trigger_activate()

    def _start_node(self) -> _RosettaTeleopLifecycleNode:
        return self._host.start(
            lambda ctx: _RosettaTeleopLifecycleNode(f"rosetta_teleop_{self.id}", self.config, context=ctx)
        )

    def get_action(self) -> dict[str, Any]:
        """Get current action from input buffers."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self._node.sample_action()

    def get_teleop_events(self) -> dict[TeleopEvents, bool]:
        """Get current teleop events state."""
        if self._node is None:
            return dict(DEFAULT_EVENTS)
        return self._node.get_events()

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Send feedback to ROS2 topics."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self._node.publish_feedback(feedback)

    def disconnect(self) -> None:
        """Deactivate and cleanup the lifecycle node."""
        node = self._node
        if node is None:
            return

        if node.is_active:
            node.trigger_deactivate()
        if node.is_configured:
            node.trigger_cleanup()

        self._host.stop()
