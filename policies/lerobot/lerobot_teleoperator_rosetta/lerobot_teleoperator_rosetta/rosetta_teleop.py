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

"""LeRobot Teleoperator backed by the same TopicBridge the robot adapter uses.

Teleop inputs are observation streams. Feedback are action streams. Message
ingest, lifecycle publishers, and teardown all come from the bridge. Only two
things are teleop-specific and live here: the events subscription and the
flattening of per-key vectors to the {name: float} shape lerobot expects.

Feedback channels cannot declare a safety behavior (rejected at contract load),
so the bridge watchdog never arms.

Lifecycle states:
    Unconfigured  node exists, no subscriptions or publishers
    Inactive      subscriptions buffering, publishers disabled
    Active        inputs sampled, feedback published
"""

from __future__ import annotations

import threading
from functools import partial
from typing import Any, Optional

import numpy as np
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from rosetta.frames.layout import FrameLayout
from rosetta.robots.ros2.field_access import resolve_indexed
from rosetta.robots.ros2.node_host import NodeHost
from rosetta.robots.ros2.ros2_utils import qos_profile_from_dict, require_transition_success
from rosetta.robots.ros2.rosetta_lifecycle_node import BridgeLifecycleNode
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


class _RosettaTeleopLifecycleNode(BridgeLifecycleNode):
    """BridgeLifecycleNode plus the teleop events subscription.

    The base owns the bridge, the lifecycle callbacks, and the safety send on
    deactivate. That send is a no-op here because feedback carries no safety
    behavior. This class extends ``_setup``/``_teardown`` with the events
    subscription and adds the sample/publish surface lerobot drives.
    """

    def __init__(self, node_name: str, config: RosettaTeleopConfig, **kwargs):
        super().__init__(node_name, config.input_specs, config.feedback_specs, config.fps, **kwargs)
        self._config = config
        self._feedback_layout = FrameLayout(config.feedback_specs)
        self._events_sub = None
        self._events_lock = threading.Lock()
        self._events_state: dict[TeleopEvents, bool] = dict(DEFAULT_EVENTS)

    # -------------------- lifecycle hooks --------------------

    def _setup(self) -> None:
        """Create the bridge entities, then the optional events subscription.

        ``events_spec`` is None when the contract declares no teleop events.
        """
        super()._setup()
        events_spec = self._config.events_spec
        if events_spec:
            self._events_sub = self.create_subscription(
                get_message(events_spec.channel.type),
                events_spec.channel.topic,
                partial(self._on_events, spec=events_spec),
                qos_profile_from_dict(events_spec.channel.qos),
            )
        self.get_logger().info(
            f"Configured: {len(self._config.input_specs)} inputs, {len(self._config.feedback_specs)} feedback"
        )

    def _teardown(self) -> None:
        """Destroy the events subscription and reset state so a reconnect starts clean."""
        super()._teardown()
        if self._events_sub is not None:
            self.destroy_subscription(self._events_sub)
            self._events_sub = None
        with self._events_lock:
            self._events_state = dict(DEFAULT_EVENTS)

    # -------------------- teleop surface --------------------

    def _on_events(self, msg, spec) -> None:
        """Update event state from one events message. Runs on the spin thread.

        Selectors resolve through :func:`resolve_indexed`, so Joy button/axis
        indices (``buttons.0``) work, not only plain fields. An unmapped event
        name is skipped. A missing or malformed field leaves that event at its
        last value instead of killing the subscription.
        """
        with self._events_lock:
            for event_name, selector in spec.select.items():
                if event_name not in EVENT_NAME_TO_ENUM:
                    continue
                try:
                    value = resolve_indexed(msg, selector)
                    self._events_state[EVENT_NAME_TO_ENUM[event_name]] = bool(value)
                except (AttributeError, IndexError, ValueError):
                    pass

    def sample_action(self) -> dict[str, Any]:
        """Flatten per-spec input samples to {namespaced_name: float}.

        A stream with no data yet is omitted, not zero-filled: a fake zero
        teleop value would command motion. ``sample_values`` returns one entry
        per input spec in declaration order, so it lines up with ``input_specs``
        position by position.
        """
        action: dict[str, Any] = {}
        for spec, value in zip(self._config.input_specs, self.bridge.sample_values(), strict=False):
            if value is None:
                continue
            for i, name in enumerate(spec.namespaced_names):
                action[name] = float(value[i])
        return action

    def get_events(self) -> dict[TeleopEvents, bool]:
        """Snapshot event state under the lock. The copy keeps callers off the shared dict."""
        with self._events_lock:
            return self._events_state.copy()

    def publish_feedback(self, feedback: dict[str, Any]) -> None:
        """Regroup {namespaced_name: value} into per-key vectors and publish.

        All or nothing: a key is published only when every one of its names is
        present, because ``publish_frame`` needs the full per-key vector. A
        partial frame would be a malformed action.
        """
        frame: dict[str, Any] = {}
        for key in self._feedback_layout.keys:
            names = [name for sl in self._feedback_layout[key].slices for name in sl.spec.namespaced_names]
            if not all(name in feedback for name in names):
                return
            frame[key] = np.array([feedback[name] for name in names], dtype=np.float64)
        if frame:
            self.bridge.publish_frame(frame)


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
            for name in spec.namespaced_names:
                features[name] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for spec in self.config.feedback_specs:
            for name in spec.namespaced_names:
                features[name] = float
        return features

    @property
    def is_connected(self) -> bool:
        """True only in ACTIVE, so get_action/send_feedback gate on a live bridge."""
        return self._node is not None and self._node.is_active

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        """Configure only: subscriptions buffer while publishers stay disabled."""
        require_transition_success(self._start_node().trigger_configure(), "configure")

    def connect(self, calibrate: bool = True) -> None:
        """Configure if needed, then activate.

        Raises DeviceAlreadyConnectedError when already connected, matching the
        robot adapter and lerobot's own devices.
        """
        del calibrate  # Unused - ROS2 teleop doesn't require calibration
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        node = self._start_node()
        if not node.is_configured:
            require_transition_success(node.trigger_configure(), "configure")
        require_transition_success(node.trigger_activate(), "activate")

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
        """Current event state, or all-False defaults before connect.

        lerobot polls events outside the connected window, so this tolerates a
        missing node instead of raising like get_action and send_feedback.
        """
        if self._node is None:
            return dict(DEFAULT_EVENTS)
        return self._node.get_events()

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Send feedback to ROS2 topics."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self._node.publish_feedback(feedback)

    def disconnect(self) -> None:
        """Deactivate then clean up the lifecycle node.

        The node property raises if the spin thread died. The finally still runs
        host.stop(), so a poisoned host is torn down even as the error
        propagates to the caller.
        """
        try:
            node = self._node
            if node is None:
                return
            if node.is_active:
                node.trigger_deactivate()
            if node.is_configured:
                node.trigger_cleanup()
        finally:
            self._host.stop()
