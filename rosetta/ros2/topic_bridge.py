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
TopicBridge: backend-neutral ROS2 observation/action plumbing.

The online side of the backend-neutral interface. TopicBridge manages
observation subscriptions, lifecycle action publishers, the safety watchdog, and
per-stream resampling buffers on a host rclpy.lifecycle.Node. It uses the same
frame dict the offline bag porter emits:

- sample_frame() -> {contract_key: np.ndarray | str}
- publish_frame() consumes {contract_key: np.ndarray}

No dependency on LeRobot or any policy framework. Framework adapters
(e.g. lerobot_robot_rosetta) convert these frame dicts to their own shapes.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
from rclpy.lifecycle import Publisher
from rclpy.timer import Timer
from rosidl_runtime_py.utilities import get_message

from rosetta.core.contract import ActionStreamSpec, ObservationStreamSpec
from rosetta.core.contract_utils import (
    aggregate_frame,
    slice_action_frame,
    StreamBuffer,
)
from rosetta.core.converters import decode_value, encode_value
from rosetta.ros2.ros2_utils import get_message_timestamp_ns, qos_profile_from_dict

# Register the ROS codecs (import side effect populates the core registry).
from rosetta.ros2 import decoders as _decoders  # noqa: F401
from rosetta.ros2 import encoders as _encoders  # noqa: F401

del _decoders, _encoders

if TYPE_CHECKING:
    pass


class TopicBridge:
    """Manages observation subscriptions, action publishers, and watchdog on a LifecycleNode.

    A plain Python object, not a Node. It creates ROS2 entities (subscriptions,
    lifecycle publishers, timers) on a host LifecycleNode via setup(), and
    destroys them via teardown().

    The host node's lifecycle transitions activate/deactivate the lifecycle
    publishers (super().on_activate() / super().on_deactivate()). The
    publisher's is_activated property gates publishing.

    Args:
        observation_specs: Resolved observation stream specs (subscriptions).
        action_specs: Resolved action stream specs (publishers).
        fps: Contract rate (Hz). Drives watchdog timing in ROS clock (sim-aware).
    """

    def __init__(
        self,
        observation_specs: list[ObservationStreamSpec],
        action_specs: list[ActionStreamSpec],
        fps: int,
    ):
        self._observation_specs = list(observation_specs)
        self._action_specs = list(action_specs)
        self._fps = int(fps)

        # Created in setup(), cleared in teardown()
        self._obs_buffers: dict[str, tuple[ObservationStreamSpec, StreamBuffer]] = {}
        self._act_publishers: dict[str, tuple[ActionStreamSpec, Publisher]] = {}
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
        created inactive and enabled when the host node transitions to active.

        Args:
            node: A rclpy.lifecycle.Node (LifecycleNode).
        """
        self._node = node

        # Create subscriptions (start buffering immediately)
        for spec in self._observation_specs:
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
        for spec in self._action_specs:
            pub = node.create_lifecycle_publisher(
                get_message(spec.msg_type),
                spec.topic,
                qos_profile_from_dict(spec.qos) or 10,
            )
            self._act_publishers[spec.topic] = (spec, pub)

        # Watchdog timer. Uses contract fps because the timer and timeout run on
        # the ROS2 clock, which respects use_sim_time and the /clock topic.
        if self._should_use_watchdog():
            period_sec = 2.0 / self._fps
            self._watchdog_timer = node.create_timer(period_sec, self._on_watchdog)

        node.get_logger().info(
            f"TopicBridge: {len(self._observation_specs)} obs, "
            f"{len(self._action_specs)} act @ {self._fps}Hz"
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
        """Reset internal state tracking, e.g. between episodes.

        Clears episode-specific state without destroying ROS2 resources.
        Called between policy runs in injected mode.
        """
        # Drop stale data from the previous episode.
        for _, buffer in self._obs_buffers.values():
            buffer.reset()

        self._missing_streams.clear()
        self._header_warned.clear()

        self._last_action_ns = None
        self._last_sent.clear()  # Clear cached actions (matters for safety_behavior="hold")

    # -------------------- Properties --------------------

    @property
    def is_active(self) -> bool:
        """Check if the host node is in the active lifecycle state."""
        current_state = self._node._state_machine.current_state
        return current_state[1] == 'active'

    @property
    def is_configured(self) -> bool:
        """Check if the host node has been configured (inactive/active/transitioning)."""
        current_state = self._node._state_machine.current_state
        return current_state[1] in ['inactive', 'active', 'activating', 'deactivating']

    # -------------------- Observation / Action --------------------

    def sample_frame(self) -> dict[str, Any]:
        """Sample all observation buffers into a backend-neutral frame dict.

        Returns {contract_key: np.ndarray | str}, the same shape the bag porter
        emits offline. Missing streams are zero-filled and logged on state
        transition. Specs sharing a key are concatenated in declaration order.
        """
        now_ns = self._node.get_clock().now().nanoseconds

        samples: list[tuple[ObservationStreamSpec, Any]] = []
        for spec, buffer in self._obs_buffers.values():
            data = buffer.sample(now_ns)
            self._log_stream_state(spec.key, data is None)
            samples.append((spec, data))

        return aggregate_frame(samples)

    def publish_frame(self, action_frame: dict[str, Any]) -> dict[str, Any]:
        """Publish a backend-neutral action frame to ROS2 topics.

        action_frame maps each action key to a combined vector. It is sliced per
        publishing stream (by selector count, declaration order) and encoded to
        ROS messages. Returns the frame actually published.
        """
        stamp_ns = self._node.get_clock().now().nanoseconds
        per_topic = slice_action_frame(
            action_frame, [spec for spec, _ in self._act_publishers.values()]
        )

        for topic, (spec, pub) in self._act_publishers.items():
            if pub is None:
                continue
            arr = np.asarray(per_topic[topic], dtype=np.float32)
            msg = encode_value(spec, arr, stamp_ns)

            # Lifecycle publisher handles active/inactive state.
            pub.publish(msg)
            self._last_sent[topic] = arr

        self._last_action_ns = stamp_ns
        return action_frame

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

        # Handle clock resets (sim time going backwards). A last_action
        # timestamp in the future means the clock reset, so clear it.
        if self._last_action_ns > now_ns:
            self._node.get_logger().warning(
                "Clock reset detected (last_action in future), resetting watchdog"
            )
            self._last_action_ns = None
            return

        # Timeout uses contract fps because timestamps are in sim time.
        timeout_ns = int(2e9 / self._fps)  # 2 frame periods

        if now_ns - self._last_action_ns > timeout_ns:
            self._node.get_logger().warning("Action timeout, sending safety action")
            self.send_safety_action()
            self._last_action_ns = None

    def _on_observation(
        self, msg, spec: ObservationStreamSpec, buffer: StreamBuffer
    ) -> None:
        """Handle an incoming observation message: extract timestamp, decode, buffer."""
        fallback_ns = self._node.get_clock().now().nanoseconds
        ts_ns, used_fallback = get_message_timestamp_ns(msg, spec, fallback_ns)

        # Warn once per stream on the first header fallback.
        if spec.stamp_src == "header" and used_fallback:
            if spec.key not in self._header_warned:
                self._node.get_logger().warning(
                    f"Header stamp unavailable for '{spec.key}', using receive time"
                )
                self._header_warned.add(spec.key)

        buffer.push(ts_ns, decode_value(msg, spec))

    def _log_stream_state(self, key: str, is_missing: bool) -> None:
        """Log only on state transitions (missing or recovered)."""
        was_missing = key in self._missing_streams
        if is_missing and not was_missing:
            self._node.get_logger().warning(f"Stream '{key}' missing - using zeros")
            self._missing_streams.add(key)
        elif not is_missing and was_missing:
            self._node.get_logger().info(f"Stream '{key}' recovered")
            self._missing_streams.discard(key)

    # -------------------- Backward-compatible accessors --------------------

    @property
    def observation_specs(self) -> list[ObservationStreamSpec]:
        """Resolved observation specs this bridge serves."""
        return self._observation_specs

    @property
    def action_specs(self) -> list[ActionStreamSpec]:
        """Resolved action specs this bridge serves."""
        return self._action_specs

    @property
    def fps(self) -> int:
        """Contract rate (Hz)."""
        return self._fps
