# Copyright 2026 Isaac Blankenau
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

"""RosConnection: the live robot ``Ros2Interface.connect`` opens for a framework.

A framework that hosts the run itself (LeRobot's CLIs driving a ``Robot`` or
``Teleoperator``) runs in a process rosetta doesn't own. The connection embeds
a lifecycle node there: a :class:`NodeHost` spins it on a private context, the
node's :class:`TopicBridge` serves frames, and an optional subscription follows
the contract's teleop events.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from rclpy.parameter import Parameter
from rosidl_runtime_py.utilities import get_message

from rosetta_ros.field_access import resolve_indexed
from rosetta_ros.node_host import NodeHost
from rosetta_ros.rclpy_utils import qos_profile_from_dict, require_transition_success
from rosetta_ros.rosetta_lifecycle_node import BridgeLifecycleNode
from rosetta_ros.topic_bridge import RosClock

if TYPE_CHECKING:
    from rosetta.contract.schema import TeleopEventMap
    from rosetta.contract.specs import ActionStreamSpec, ObservationStreamSpec

__all__ = [
    "RosConnection",
    "read_events",
]


def read_events(state: Mapping[str, bool], msg: Any, select: Mapping[str, str]) -> dict[str, bool]:
    """Return ``state`` updated by one events message.

    Each event's selector resolves through :func:`resolve_indexed`, so Joy
    indices (``buttons.0``) work as well as plain fields. A missing or
    malformed field leaves that event at its last value instead of killing the
    subscription.
    """
    updated = dict(state)
    for name, selector in select.items():
        try:
            updated[name] = bool(resolve_indexed(msg, selector))
        except (AttributeError, IndexError, ValueError):
            continue
    return updated


class _ConnectionNode(BridgeLifecycleNode):
    """BridgeLifecycleNode plus the optional events subscription."""

    def __init__(
        self,
        node_name: str,
        observations: Sequence[ObservationStreamSpec],
        actions: Sequence[ActionStreamSpec],
        fps: int,
        events: TeleopEventMap | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(node_name, list(observations), list(actions), fps, **kwargs)
        self._event_map = events
        self._events_sub = None
        self._events_lock = threading.Lock()
        self._events: dict[str, bool] = {}

    def setup(self) -> None:
        super().setup()
        if self._event_map is not None:
            channel = self._event_map.channel
            self._events_sub = self.create_subscription(
                get_message(channel.type), channel.topic, self._on_events, qos_profile_from_dict(channel.qos)
            )

    def teardown(self) -> None:
        super().teardown()
        if self._events_sub is not None:
            self.destroy_subscription(self._events_sub)
            self._events_sub = None
        with self._events_lock:
            self._events = {}

    def _on_events(self, msg: Any) -> None:
        # Runs on the spin thread.
        with self._events_lock:
            self._events = read_events(self._events, msg, self._event_map.select)

    def events(self) -> dict[str, bool]:
        with self._events_lock:
            return dict(self._events)


class RosConnection:
    """:class:`~rosetta.frames.protocols.Connection` over an embedded lifecycle node.

    Opening it creates the node, then configures and activates it: the
    subscriptions buffer and the publishers publish. Closing deactivates (which
    sends the safety action), cleans up and shuts the node's context down.
    """

    def __init__(
        self,
        *,
        observations: Sequence[ObservationStreamSpec],
        actions: Sequence[ActionStreamSpec],
        fps: int,
        name: str,
        events: TeleopEventMap | None = None,
        use_sim_time: bool = False,
    ) -> None:
        overrides = [Parameter("use_sim_time", Parameter.Type.BOOL, True)] if use_sim_time else None
        self._host = NodeHost()
        # Kept for close() alone: everything else goes through _live(), but the
        # safety action must still go out after the spin thread has died.
        self._node = node = self._host.start(
            lambda context: _ConnectionNode(
                name, observations, actions, fps, events, context=context, parameter_overrides=overrides
            )
        )
        try:
            # trigger_* returns the transition's result rather than raising.
            require_transition_success(node.trigger_configure(), "configure")
            require_transition_success(node.trigger_activate(), "activate")
        except Exception:
            self.close()
            raise
        self._clock = _LiveClock(self)

    def _live(self) -> _ConnectionNode:
        """The node, while it spins. Raises once its spin thread has died or the connection is closed.

        Every call goes through here rather than a cached node, so a robot
        that is gone raises instead of serving its last frames.
        """
        node = self._host.node  # raises if the spin thread died
        if node is None:
            raise RuntimeError("RosConnection is closed")
        return node

    # FrameIO, served by the node's bridge.

    def sample_frame(self, *, fill_missing: bool = True) -> dict[str, Any]:
        return self._live().bridge.sample_frame(fill_missing=fill_missing)

    @property
    def warmed_up(self) -> bool:
        return self._live().bridge.warmed_up

    @property
    def clock(self) -> _LiveClock:
        return self._clock

    def publish_frame(self, action_frame: dict[str, Any]) -> dict[str, Any]:
        return self._live().bridge.publish_frame(action_frame)

    def send_safety_action(self) -> None:
        self._live().bridge.send_safety_action()

    def reset_state(self) -> None:
        self._live().bridge.reset_state()

    # Connection.

    def events(self) -> dict[str, bool]:
        return self._live().events()

    def close(self) -> None:
        # Deactivating sends the safety action. trigger_* runs on the calling
        # thread, so it works even after the spin thread has died, when the
        # watchdog has stopped and the robot would otherwise keep its last
        # command. The host stops even if a transition raises, so a failed
        # close still releases the spin thread and context.
        node, self._node = self._node, None
        try:
            if node is not None:
                if node.is_active:
                    node.trigger_deactivate()
                if node.is_configured:
                    node.trigger_cleanup()
        finally:
            self._host.stop()


class _LiveClock:
    """The connection's :class:`RosClock`, read through its liveness check.

    Callers keep the clock from connect, so a pacing loop waiting on a sim
    clock that died with the spin thread raises on its next read instead of
    waiting forever.
    """

    __slots__ = ("_connection",)

    def __init__(self, connection: RosConnection) -> None:
        self._connection = connection

    def _clock(self) -> RosClock:
        return self._connection._live().bridge.clock

    @property
    def simulated(self) -> bool:
        return self._clock().simulated

    def now_ns(self) -> int:
        return self._clock().now_ns()
