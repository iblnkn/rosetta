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

"""Backend-neutral ROS2 observation/action plumbing on a host LifecycleNode.

TopicBridge is the online side of the backend-neutral interface. It owns the
observation subscriptions, lifecycle action publishers, per-stream resampling
buffers, and the safety watchdog, all created on a host ``rclpy.lifecycle.Node``.

Ingest, timestamping, and resampling are shared with the offline bag porter, so
a replayed bag and a live robot yield identical frame dicts. That bag/live
parity is why an offline-trained policy behaves the same online. The frame dict
is the seam:

- ``sample_frame()`` returns ``{contract_key: np.ndarray | str}``.
- ``publish_frame()`` consumes ``{contract_key: np.ndarray}``.

Nothing here depends on LeRobot or any policy framework. Framework adapters
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
from rosetta.robots.ros2.rclpy_utils import lifecycle_state_label, qos_profile_from_dict

if TYPE_CHECKING:
    from rclpy.lifecycle import LifecycleNode

# Watchdog fires after this many frame periods without an action.
WATCHDOG_PERIODS = 2


def _is_latched(spec: ObservationStreamSpec) -> bool:
    """True when the channel declares transient_local (latched) durability."""
    qos = spec.source.channel.qos or {}
    return str(qos.get("durability", "")).lower().strip() == "transient_local"


class TopicBridge:
    """Own observation subscriptions, action publishers, and the watchdog on a host node.

    This is not itself a Node. It creates ROS2 entities (subscriptions,
    lifecycle publishers, timers) on a host LifecycleNode in setup() and
    destroys them in teardown(), so its lifetime is bounded by the host's.

    The host node's lifecycle transitions activate and deactivate the lifecycle
    publishers. Each publisher's ``is_activated`` flag gates publishing, so an
    action encoded while inactive is silently dropped rather than sent.

    Args:
        observation_specs: Resolved observation stream specs (subscriptions).
        action_specs: Resolved action stream specs (publishers).
        fps: Contract rate (Hz). Sets the watchdog period on the ROS clock, so
            the timeout tracks sim time under use_sim_time.

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

        # Layouts validate the key structure once. Several specs may share a key
        # and even a topic, so buffers and publishers below are positional lists
        # kept parallel to the spec lists, not keyed by contract key.
        self._obs_layout = FrameLayout(self._observation_specs)
        self._act_layout = FrameLayout(self._action_specs)

        # Populated in setup(), cleared in teardown().
        self._obs_buffers: list[tuple[ObservationStreamSpec, StreamBuffer]] = []
        self._act_publishers: list[tuple[ActionStreamSpec, Publisher]] = []
        self._subscriptions: list[Subscription] = []
        self._watchdog_timer: Timer | None = None

        # Spec positions currently reporting no data, for edge-triggered logging.
        self._missing_streams: set[int] = set()
        # Ingest holds the timestamp/decode/push policy shared with the bag
        # porter, which is what makes bag and live frames identical. The log
        # callbacks read self._node lazily because it is None until setup(), and
        # no message can reach ingest before setup() has run anyway.
        self._ingest = StreamIngest(
            warn=lambda m: self._node.get_logger().warning(m),
            info=lambda m: self._node.get_logger().info(m),
        )

        # Watchdog state shared across threads. publish_frame writes both fields
        # on the policy thread; the watchdog timer reads and clears them on the
        # executor thread. No lock: each is a single attribute assignment, and a
        # stale read only delays or advances one safety action by one tick.
        self._last_action_ns: int | None = None
        self._last_sent: list[np.ndarray | None] = []

        # Host node, set in setup() and cleared in teardown().
        self._node: LifecycleNode | None = None

    def setup(self, node) -> None:
        """Create subscriptions, lifecycle publishers, and watchdog on the given node.

        Subscriptions start buffering immediately, before activation, so data is
        already flowing when the policy begins. Lifecycle publishers are created
        inactive and only publish once the host node transitions to active.

        Args:
            node: The host rclpy.lifecycle.Node (LifecycleNode).

        Raises:
            RuntimeError: If called a second time without an intervening
                teardown(), which would leak the first set of ROS entities.

        """
        if self._node is not None:
            raise RuntimeError("TopicBridge.setup() called twice without teardown()")
        self._node = node

        # One subscription per spec. Several specs may read the same topic with
        # different selectors; each gets its own callback and buffer.
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

        # Lifecycle publishers stay inactive until the host node activates.
        for spec in self._action_specs:
            pub = node.create_lifecycle_publisher(
                get_message(spec.source.channel.type),
                spec.source.channel.topic,
                qos_profile_from_dict(spec.source.channel.qos),
            )
            self._act_publishers.append((spec, pub))
        self._last_sent = [None] * len(self._act_publishers)

        # The period comes from contract fps because the timer and the timeout
        # check both run on the ROS clock, which honors use_sim_time and /clock.
        # A bag played at any wall speed still times out on its own timeline.
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
        """True once every observation stream has delivered at least one message."""
        return self._node is not None and all(buffer.last_ts is not None for _, buffer in self._obs_buffers)

    def send_safety_action(self) -> None:
        """Publish each active stream's declared safety command, then disarm.

        Zeros for ``safety: zero`` (also the fallback for ``hold`` with nothing
        cached yet), the last sent vector for ``safety: hold``, nothing for
        ``safety: none``. Inactive publishers are skipped. Clearing
        ``_last_action_ns`` disarms the watchdog until a real frame arms it
        again, so this fires once per stall, not every tick.

        Runs on the executor thread (called from the watchdog).
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
        """Clear episode-specific state between policy runs, keeping ROS entities.

        Unlike teardown(), this leaves subscriptions and publishers in place so
        the next episode starts without recreating them.
        """
        # Latched (transient_local) streams keep their value: nothing will
        # republish it, so dropping it would leave the stream permanently empty.
        for spec, buffer in self._obs_buffers:
            if not _is_latched(spec):
                buffer.reset()

        self._missing_streams.clear()
        self._ingest.reset()

        self._last_action_ns = None
        # Drop the hold cache too, so a new episode never holds a stale command.
        self._last_sent = [None] * len(self._act_publishers)

    # -------------------- Observation / Action --------------------

    def sample_values(self) -> list[Any]:
        """Sample every observation buffer in spec order, with None for no data yet.

        The pre-assembly view of :meth:`sample_frame`, for adapters that need
        per-spec values. The teleoperator uses it to omit absent streams instead
        of zero-filling them.

        Returns:
            One entry per observation spec, in declaration order.

        """
        # Valid only after setup(): self._node is non-None here by contract.
        now_ns = self._node.get_clock().now().nanoseconds

        values: list[Any] = []
        for index, (spec, buffer) in enumerate(self._obs_buffers):
            data = buffer.sample(now_ns)
            self._log_stream_state(index, spec, data is None)
            values.append(data)
        return values

    def sample_frame(self) -> dict[str, Any]:
        """Sample all observation buffers into a backend-neutral frame dict.

        Returns:
            ``{contract_key: np.ndarray | str}``, the same shape the bag porter
            emits offline. Missing streams are zero-filled so the frame keeps a
            stable shape, and logged only on a missing/recovered transition.
            Specs sharing a key are concatenated in declaration order.

        """
        return self._obs_layout.assemble(self.sample_values())

    def publish_frame(self, action_frame: dict[str, Any]) -> dict[str, Any]:
        """Encode and publish a backend-neutral action frame to ROS2 topics.

        The frame maps each action key to a combined vector. Each vector is
        sliced per publishing stream (by selector count, declaration order) and
        encoded to a ROS message.

        Encode and publish are two separate phases: every spec is encoded first,
        and only if all succeed does anything reach the wire. So an encode
        failure never leaves a partial frame on hardware. A non-finite command
        (NaN or Inf from the policy) drops the whole frame with a throttled
        error rather than raising. Actions simply stop flowing, the watchdog then
        applies each channel's declared ``safety`` behavior, and a recovered
        policy resumes with no extra handshaking. ``_last_sent`` (the source for
        ``safety: hold``) therefore only ever holds fully validated frames.

        Args:
            action_frame: ``{contract_key: np.ndarray}`` for this tick.

        Returns:
            The same ``action_frame`` object, unchanged, for call chaining.
            Streams whose lifecycle publisher is inactive are neither published
            nor recorded as sent.

        Raises:
            The split call raises on a key or length mismatch rather than
            silently truncating; a NonFiniteActionError is caught here.

        """
        # Valid only after setup(): self._node is non-None here by contract.
        stamp_ns = self._node.get_clock().now().nanoseconds
        per_spec = self._act_layout.split(action_frame)

        # Phase 1: encode everything. A raise here leaves nothing published.
        try:
            messages = [encode_value(per_spec[i], spec, stamp_ns) for i, (spec, _) in enumerate(self._act_publishers)]
        except NonFiniteActionError as e:
            self._node.get_logger().error(f"Dropping action frame: {e}", throttle_duration_sec=1.0)
            return action_frame

        # Phase 2: publish. Inactive publishers are skipped explicitly because
        # their publish() is a silent no-op. That keeps the hold cache and the
        # watchdog arm reflecting only commands that actually reached the wire.
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
        """True unless there is nothing to guard: no action specs, or all ``safety: none``."""
        if not self._action_specs:
            return False
        return not all(spec.source.channel.safety == SafetyBehavior.NONE for spec in self._action_specs)

    def _on_watchdog(self) -> None:
        """Send a safety action if the action stream has stalled.

        Runs on the executor thread. No-op unless the host node is active and at
        least one frame has armed the watchdog since the last stall or reset.
        """
        if lifecycle_state_label(self._node) != "active":
            return
        if self._last_action_ns is None:
            return

        now_ns = self._node.get_clock().now().nanoseconds

        # A last_action timestamp in the future means the sim clock jumped
        # backwards (bag loop or reset). Disarm rather than misjudge the gap.
        if self._last_action_ns > now_ns:
            self._node.get_logger().warning("Clock reset detected (last_action in future), resetting watchdog")
            self._last_action_ns = None
            return

        # Timeout is derived from contract fps because the compared timestamps
        # are ROS clock (sim) time, the same base as _last_action_ns.
        timeout_ns = int(WATCHDOG_PERIODS * 1e9 / self._fps)

        if now_ns - self._last_action_ns > timeout_ns:
            self._node.get_logger().warning("Action timeout, sending safety action")
            self.send_safety_action()  # disarms until the next published frame

    def _on_observation(self, msg, spec: ObservationStreamSpec, buffer: StreamBuffer, index: int) -> None:
        """Push one incoming message into its buffer via the shared ingest policy."""
        self._ingest.ingest(msg, spec, buffer, index, receive_ns=self._node.get_clock().now().nanoseconds)

    def _log_stream_state(self, index: int, spec: ObservationStreamSpec, is_missing: bool) -> None:
        """Log edge-triggered, only when a stream goes missing or recovers."""
        was_missing = index in self._missing_streams
        if is_missing and not was_missing:
            self._node.get_logger().warning(f"Stream '{spec.key}' ({spec.source.channel.topic}) missing")
            self._missing_streams.add(index)
        elif not is_missing and was_missing:
            self._node.get_logger().info(f"Stream '{spec.key}' ({spec.source.channel.topic}) recovered")
            self._missing_streams.discard(index)
