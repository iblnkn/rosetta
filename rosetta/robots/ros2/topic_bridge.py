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
from typing import TYPE_CHECKING, Any

import numpy as np
from rclpy.lifecycle import Publisher
from rclpy.subscription import Subscription
from rclpy.timer import Timer
from rosidl_runtime_py.utilities import get_message

from rosetta.contract.schema import SafetyBehavior
from rosetta.contract.specs import ActionStreamSpec, ObservationStreamSpec
from rosetta.frames.codecs import NonFiniteActionError, encode_value
from rosetta.frames.layout import FrameLayout
from rosetta.frames.resample import StreamBuffer
from rosetta.robots.ros2.ingest import StreamIngest
from rosetta.robots.ros2.ros2_utils import lifecycle_state_label, qos_profile_from_dict

if TYPE_CHECKING:
    from rclpy.lifecycle import LifecycleNode

# Watchdog fires after this many frame periods without an action. The timer
# period equals the timeout, so worst-case detection latency is two timer
# periods after the last action. The watchdog is one-shot: firing (or a
# detected clock reset) disarms it until the next successfully published
# frame. It runs on the host node's executor, so it guards a healthy process
# whose policy stopped commanding -- a wedged process cannot run it; the
# hardware-side backstop is the downstream controller's own command timeout.
WATCHDOG_PERIODS = 2


def _is_latched(spec: ObservationStreamSpec) -> bool:
    """True when the channel declares transient_local (latched) durability."""
    qos = spec.source.channel.qos or {}
    return str(qos.get("durability", "")).lower().strip() == "transient_local"


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
        self._subscriptions: list[Subscription] = []
        self._watchdog_timer: Timer | None = None

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

        # Safety state. Written by publish_frame (policy thread) and read /
        # cleared by the watchdog timer (executor thread) without a lock:
        # attribute access is atomic under the GIL, and the worst-case
        # interleaving at the timeout boundary is one extra or one skipped
        # safety tick, self-correcting on the next published frame.
        self._last_action_ns: int | None = None
        self._last_sent: list[np.ndarray | None] = []

        # Reference to the host node (set in setup, cleared in teardown)
        self._node: LifecycleNode | None = None

    def setup(self, node) -> None:
        """Create subscriptions, lifecycle publishers, and watchdog on the given node.

        Subscriptions start buffering immediately. Lifecycle publishers are
        created inactive and enabled when the host node transitions to active.

        Args:
            node: A rclpy.lifecycle.Node (LifecycleNode).

        """
        if self._node is not None:
            raise RuntimeError("TopicBridge.setup() called twice without teardown()")
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
                qos_profile_from_dict(spec.source.channel.qos),
            )
            self._subscriptions.append(sub)

        # Create lifecycle publishers (disabled until host node activates)
        for spec in self._action_specs:
            pub = node.create_lifecycle_publisher(
                get_message(spec.source.channel.type),
                spec.source.channel.topic,
                qos_profile_from_dict(spec.source.channel.qos),
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
        """Publish safety actions (zeros or hold) per spec's declared ``safety``.

        ``zeros`` are action-space zeros, routed through ``encode_value``'s
        inverse operator pipeline like any command — a declared ``clamp``
        maps them into the safe wire range by design (see
        :class:`SafetyBehavior`). Only publishes on activated lifecycle
        publishers. Two-phase like :meth:`publish_frame` -- every message is
        encoded before any is published, so an encode failure can never
        leave a partial safety frame on hardware. Securing the robot also
        disarms the watchdog: hosts that deactivate mid-goal get identical
        semantics without touching bridge internals. No-op before
        ``setup()``.
        """
        if self._node is None:
            return
        stamp_ns = self._node.get_clock().now().nanoseconds
        to_publish = []
        for i, (spec, pub) in enumerate(self._act_publishers):
            if not pub.is_activated or spec.source.channel.safety == SafetyBehavior.NONE:
                continue
            if spec.source.channel.safety == SafetyBehavior.HOLD and self._last_sent[i] is not None:
                arr = self._last_sent[i]
            else:
                arr = np.zeros(spec.dim)
            to_publish.append((pub, encode_value(arr, spec, stamp_ns)))
        for pub, msg in to_publish:
            pub.publish(msg)
        self._last_action_ns = None

    def reset_state(self) -> None:
        """Reset internal state tracking, e.g. between episodes.

        Clears episode-specific state without destroying ROS2 resources.
        Called between policy runs in injected mode.
        """
        # Drop stale data from the previous episode — except latched
        # (transient_local) streams: their data is not episode-scoped, and
        # DDS redelivers latched samples only to NEW subscriptions, so a
        # cleared buffer on a publish-once topic would never refill.
        for spec, buffer in self._obs_buffers:
            if not _is_latched(spec):
                buffer.reset()

        self._missing_streams.clear()
        self._ingest.reset()

        self._last_action_ns = None
        # Clear cached actions (matters for safety_behavior="hold")
        self._last_sent = [None] * len(self._act_publishers)

    # -------------------- Observation / Action --------------------

    def sample_values(self) -> list[Any]:
        """Sample every observation buffer, in spec order (None = no data yet).

        The pre-assembly view of :meth:`sample_frame`, for adapters that need
        per-spec values — e.g. the teleoperator, which omits absent streams
        instead of zero-filling them.
        """
        # Legal only after setup(): self._node is never None here by contract.
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
        lifecycle publisher is inactive are neither published nor recorded
        as sent).

        Publishing is two-phase -- every spec is encoded before anything is
        published -- so an encode failure can never leave a partial frame on
        hardware. A non-finite command (NaN/Inf from the policy) drops the
        whole frame with a throttled error instead of raising: actions simply
        stop flowing, so the existing watchdog applies each channel's declared
        ``safety`` behavior, and a recovered policy resumes seamlessly.
        ``_last_sent`` (the ``safety: hold`` source) only ever holds
        fully-validated frames.
        """
        # Legal only after setup(): self._node is never None here by contract.
        stamp_ns = self._node.get_clock().now().nanoseconds
        # Raises on a key/length mismatch instead of silently truncating.
        per_spec = self._act_layout.split(action_frame)

        # Phase 1: encode everything. Any raise here leaves nothing published.
        try:
            messages = [encode_value(per_spec[i], spec, stamp_ns) for i, (spec, _) in enumerate(self._act_publishers)]
        except NonFiniteActionError as e:
            self._node.get_logger().error(f"Dropping action frame: {e}", throttle_duration_sec=1.0)
            return action_frame

        # Phase 2: publish, skipping inactive lifecycle publishers explicitly
        # (their publish() is a silent no-op) so the `safety: hold` cache and
        # the watchdog arm only ever reflect commands that reached the wire.
        published = False
        for i, (_, pub) in enumerate(self._act_publishers):
            if not pub.is_activated:
                continue
            pub.publish(messages[i])
            self._last_sent[i] = per_spec[i]
            published = True

        if published:
            self._last_action_ns = stamp_ns
        return action_frame

    # -------------------- Private --------------------

    def _should_use_watchdog(self) -> bool:
        """Check if watchdog should be enabled."""
        if not self._action_specs:
            return False
        return not all(spec.source.channel.safety == SafetyBehavior.NONE for spec in self._action_specs)

    def _on_watchdog(self) -> None:
        """Check if actions have stopped and send safety action if needed."""
        if lifecycle_state_label(self._node) != "active":
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
            self.send_safety_action()  # also disarms until the next published frame

    def _on_observation(self, msg, spec: ObservationStreamSpec, buffer: StreamBuffer, index: int) -> None:
        """Handle an incoming observation message via the shared ingest policy."""
        self._ingest.ingest(msg, spec, buffer, index, receive_ns=self._node.get_clock().now().nanoseconds)

    def _log_stream_state(self, index: int, spec: ObservationStreamSpec, is_missing: bool) -> None:
        """Log only on state transitions (missing or recovered)."""
        was_missing = index in self._missing_streams
        if is_missing and not was_missing:
            self._node.get_logger().warning(f"Stream '{spec.key}' ({spec.source.channel.topic}) missing")
            self._missing_streams.add(index)
        elif not is_missing and was_missing:
            self._node.get_logger().info(f"Stream '{spec.key}' ({spec.source.channel.topic}) recovered")
            self._missing_streams.discard(index)
