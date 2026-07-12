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
from typing import Any, Optional

import numpy as np
from rclpy.lifecycle import Publisher
from rclpy.timer import Timer
from rosidl_runtime_py.utilities import get_message

from rosetta.contract.specs import ActionStreamSpec, ObservationStreamSpec
from rosetta.frames.codecs import encode_value
from rosetta.frames.layout import FrameLayout
from rosetta.frames.resample import StreamBuffer

# Register the ROS codecs (import side effect populates the core registry).
from rosetta.robots.ros2 import decoders as _decoders
from rosetta.robots.ros2 import encoders as _encoders
from rosetta.robots.ros2.ingest import StreamIngest
from rosetta.robots.ros2.ros2_utils import (
    LIFECYCLE_CONFIGURED_LABELS,
    lifecycle_state_label,
    qos_profile_from_dict,
)

del _decoders, _encoders


# Watchdog fires after this many frame periods without an action.
WATCHDOG_PERIODS = 2


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

        # Canonical key layouts (validated here; built once, used every tick).
        # Multiple specs may share a key and even a topic, so buffers and
        # publishers are positional lists aligned with the spec lists — never
        # dicts keyed by topic.
        self._obs_layout = FrameLayout(self._observation_specs)
        self._act_layout = FrameLayout(self._action_specs)

        # Created in setup(), cleared in teardown()
        self._obs_buffers: list[tuple[ObservationStreamSpec, StreamBuffer]] = []
        self._act_publishers: list[tuple[ActionStreamSpec, Publisher]] = []
        self._subscriptions: list[Any] = []
        self._watchdog_timer: Optional[Timer] = None

        # Stream state tracking, keyed by spec position (specs may share a
        # key and even a topic, so neither is a unique stream identity —
        # keying by key made one stream's success clear a sibling's flag,
        # flapping the logs).
        self._missing_streams: set[int] = set()
        # Timestamp/decode/push policy shared with the bag porter (parity by
        # construction). Loggers late-bind to the host node, which is only
        # set when messages can actually arrive.
        self._ingest = StreamIngest(
            warn=lambda m: self._node.get_logger().warning(m),
            info=lambda m: self._node.get_logger().info(m),
        )

        # Safety state
        self._last_action_ns: Optional[int] = None
        self._last_sent: list[Optional[np.ndarray]] = []

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

        # Create subscriptions (start buffering immediately). One subscription
        # per spec; several specs may read the same topic with different
        # selectors, each into its own buffer.
        for index, spec in enumerate(self._observation_specs):
            buffer = StreamBuffer.from_spec(spec)
            self._obs_buffers.append((spec, buffer))
            callback = partial(self._on_observation, spec=spec, buffer=buffer, index=index)
            sub = node.create_subscription(
                get_message(spec.source.channel.type),
                spec.source.channel.topic,
                callback,
                qos_profile_from_dict(spec.source.channel.qos) or 10,
            )
            self._subscriptions.append(sub)

        # Create lifecycle publishers (disabled until host node activates)
        for spec in self._action_specs:
            pub = node.create_lifecycle_publisher(
                get_message(spec.source.channel.type),
                spec.source.channel.topic,
                qos_profile_from_dict(spec.source.channel.qos) or 10,
            )
            self._act_publishers.append((spec, pub))
        self._last_sent = [None] * len(self._act_publishers)

        # Watchdog timer. Uses contract fps because the timer and timeout run on
        # the ROS2 clock, which respects use_sim_time and the /clock topic.
        if self._should_use_watchdog():
            period_sec = WATCHDOG_PERIODS / self._fps
            self._watchdog_timer = node.create_timer(period_sec, self._on_watchdog)

        node.get_logger().info(
            f"TopicBridge: {len(self._observation_specs)} obs, {len(self._action_specs)} act @ {self._fps}Hz"
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

        for _, pub in self._act_publishers:
            if pub is not None:
                node.destroy_publisher(pub)
        self._act_publishers.clear()

        self._obs_buffers.clear()
        self._missing_streams.clear()
        self._ingest.reset()
        self._last_sent = []
        self._last_action_ns = None
        self._node = None

    @property
    def warmed_up(self) -> bool:
        """True once every observation stream has delivered at least one message.

        The same predicate the bag porter applies before emitting frames
        (bag_frames skips warmup ticks), so a ported dataset never contains
        the zero-filled frames a cold bridge serves. Adapters should gate
        their first ``sample_frame()`` on this to keep live first frames
        consistent with ported datasets. False before ``setup()``.
        """
        return self._node is not None and all(buffer.last_ts is not None for _, buffer in self._obs_buffers)

    def send_safety_action(self) -> None:
        """Publish safety action (zeros or hold) per spec's safety_behavior.

        Only publishes on activated lifecycle publishers.
        """
        if self._node is None:
            return
        stamp_ns = self._node.get_clock().now().nanoseconds
        for i, (spec, pub) in enumerate(self._act_publishers):
            if pub is None or not pub.is_activated:
                continue
            if spec.source.channel.safety == "none":
                continue
            if spec.source.channel.safety == "hold" and self._last_sent[i] is not None:
                arr = self._last_sent[i]
            else:
                arr = np.zeros(max(len(spec.names), 1), dtype=np.float32)
            msg = encode_value(spec, arr, stamp_ns)
            pub.publish(msg)

    def reset_state(self) -> None:
        """Reset internal state tracking, e.g. between episodes.

        Clears episode-specific state without destroying ROS2 resources.
        Called between policy runs in injected mode.
        """
        # Drop stale data from the previous episode.
        for _, buffer in self._obs_buffers:
            buffer.reset()

        self._missing_streams.clear()
        self._ingest.reset()

        self._last_action_ns = None
        # Clear cached actions (matters for safety_behavior="hold")
        self._last_sent = [None] * len(self._act_publishers)

    # -------------------- Properties --------------------

    @property
    def is_active(self) -> bool:
        """Check if the host node is in the active lifecycle state."""
        return lifecycle_state_label(self._node) == "active"

    @property
    def is_configured(self) -> bool:
        """Check if the host node has been configured (inactive/active/transitioning)."""
        return lifecycle_state_label(self._node) in LIFECYCLE_CONFIGURED_LABELS

    # -------------------- Observation / Action --------------------

    def sample_values(self) -> list[Any]:
        """Sample every observation buffer, in spec order (None = no data yet).

        The pre-assembly view of :meth:`sample_frame`, for adapters that need
        per-spec values — e.g. the teleoperator, which omits absent streams
        instead of zero-filling them.
        """
        now_ns = self._node.get_clock().now().nanoseconds

        values: list[Any] = []
        for index, (spec, buffer) in enumerate(self._obs_buffers):
            data = buffer.sample(now_ns)
            self._log_stream_state(index, spec, data is None)
            values.append(data)
        return values

    def sample_frame(self) -> dict[str, Any]:
        """Sample all observation buffers into a backend-neutral frame dict.

        Returns {contract_key: np.ndarray | str}, the same shape the bag porter
        emits offline. Missing streams are zero-filled and logged on state
        transition. Specs sharing a key are concatenated in declaration order.
        """
        return self._obs_layout.assemble(self.sample_values())

    def publish_frame(self, action_frame: dict[str, Any]) -> dict[str, Any]:
        """Publish a backend-neutral action frame to ROS2 topics.

        action_frame maps each action key to a combined vector. It is sliced per
        publishing stream (by selector count, declaration order) and encoded to
        ROS messages. Returns the input frame for convenience (streams whose
        lifecycle publisher is inactive are not actually published).
        """
        stamp_ns = self._node.get_clock().now().nanoseconds
        # Raises on a key/length mismatch instead of silently truncating.
        per_spec = self._act_layout.split(action_frame)

        for i, (spec, pub) in enumerate(self._act_publishers):
            if pub is None:
                continue
            arr = per_spec[i]
            msg = encode_value(spec, arr, stamp_ns)

            # Lifecycle publisher handles active/inactive state.
            pub.publish(msg)
            self._last_sent[i] = arr

        self._last_action_ns = stamp_ns
        return action_frame

    # -------------------- Private --------------------

    def _should_use_watchdog(self) -> bool:
        """Check if watchdog should be enabled."""
        if not self._action_specs:
            return False
        return not all(spec.source.channel.safety == "none" for spec in self._action_specs)

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
            self._node.get_logger().warning("Clock reset detected (last_action in future), resetting watchdog")
            self._last_action_ns = None
            return

        # Timeout uses contract fps because timestamps are in sim time.
        timeout_ns = int(WATCHDOG_PERIODS * 1e9 / self._fps)

        if now_ns - self._last_action_ns > timeout_ns:
            self._node.get_logger().warning("Action timeout, sending safety action")
            self.send_safety_action()
            self._last_action_ns = None

    def _on_observation(self, msg, spec: ObservationStreamSpec, buffer: StreamBuffer, index: int) -> None:
        """Handle an incoming observation message via the shared ingest policy."""
        self._ingest.ingest(msg, spec, buffer, index, fallback_ns=self._node.get_clock().now().nanoseconds)

    def _log_stream_state(self, index: int, spec: ObservationStreamSpec, is_missing: bool) -> None:
        """Log only on state transitions (missing or recovered)."""
        was_missing = index in self._missing_streams
        if is_missing and not was_missing:
            self._node.get_logger().warning(f"Stream '{spec.key}' ({spec.source.channel.topic}) missing - using zeros")
            self._missing_streams.add(index)
        elif not is_missing and was_missing:
            self._node.get_logger().info(f"Stream '{spec.key}' ({spec.source.channel.topic}) recovered")
            self._missing_streams.discard(index)
