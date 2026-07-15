#!/usr/bin/env python3
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
EpisodeRecorderNode: Stream-to-bag recorder with action control.

Records ROS2 messages directly to rosbag2 as they arrive. Topics come from
a contract file. The node exposes a RecordEpisode action for start/stop control.

Usage:
    ros2 run rosetta episode_recorder_node --ros-args \
        -p contract_path:=/path/to/contract.yaml

    ros2 action send_goal /record_episode \
        rosetta_interfaces/action/RecordEpisode "{prompt: 'pick up cube'}" --feedback
"""

from __future__ import annotations

import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

import rosbag2_py
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionServer, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import serialize_message
from rosetta_interfaces.action import RecordEpisode
from rosetta_interfaces.srv import StartRecording
from rosidl_runtime_py.utilities import get_message
from std_srvs.srv import Trigger

from rosetta.contract.schema import load_contract
from rosetta.contract.specs import iter_specs
from rosetta.robots.ros2.bag_metadata import (
    BAG_CONTRACT_KEY,
    BAG_PROMPT_KEY,
    update_custom_data,
)
from rosetta.robots.ros2.nodes.node_utils import (
    extract_qos_numeric_values,
    finish_goal,
    positive_rate_descriptor,
    qos_to_rosbag2,
    spin_lifecycle_node,
)
from rosetta.robots.ros2.ros2_utils import qos_profile_from_dict
from rosetta.robots.ros2.rosetta_lifecycle_node import RosettaLifecycleNode

# ---------- Constants ----------

# Metadata file retry settings (internal implementation detail)
METADATA_RETRY_COUNT = 10
METADATA_RETRY_DELAY_SEC = 0.1
# Safety ceiling on the widened QoS depth for per-episode latched
# subscriptions, regardless of how many publishers are discovered
# (see _latched_sub_depth).
MAX_LATCHED_SUB_DEPTH = 64

# Regex patterns for topics excluded from auto-recording by default.
# Uses the same regex convention as ``ros2 bag record --exclude``.
_DEFAULT_BLACKLIST = (
    r"^/rosout$",
    r"^/parameter_events$",
)


def adapt_publisher_qos(publishers_info, *, warn=None) -> QoSProfile:
    """
    Subscription QoS for a discovered topic, adapted across ALL its publishers.

    Mirrors ``ros2 bag record``: request RELIABLE only if every publisher
    offers it, TRANSIENT_LOCAL only if every publisher offers it — otherwise
    an incompatible publisher's messages would silently never be delivered.
    ``warn`` (a callable taking a message) is invoked when the publishers
    disagree, since the adapted profile is then a compromise worth noticing.
    """
    if not publishers_info:
        return qos_profile_from_dict(None)
    profiles = [p.qos_profile for p in publishers_info]
    reliable = all(p.reliability == ReliabilityPolicy.RELIABLE for p in profiles)
    latched = all(p.durability == DurabilityPolicy.TRANSIENT_LOCAL for p in profiles)
    if warn is not None and len({(p.reliability, p.durability) for p in profiles}) > 1:
        warn("publishers offer mixed QoS; subscribing adapted profile")
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=max(10, *(p.depth for p in profiles)),
        reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL if latched else DurabilityPolicy.VOLATILE,
    )


class EpisodeRecorderNode(RosettaLifecycleNode):
    """
    Stream-to-bag episode recorder with lifecycle and action interface.

    Design:
    - Non-latched contract topics get persistent subscriptions (created at
      configure) so an episode captures them from its very first instant.
    - Latched (TRANSIENT_LOCAL) contract topics and all auto-discovered
      topics get per-episode subscriptions, created after the bag writer is
      live: the RMW redelivers each publisher's latched sample to the fresh
      subscription, which is how pre-episode data (e.g. /tf_static) lands in
      every bag — the same mechanism ``ros2 bag record`` relies on.
    - The writer IS the recording flag: subscription callbacks write whenever
      a writer is open; "a recording is in flight" is the busy guard.
    """

    def __init__(self, **kwargs):
        """Initialize the episode recorder node and declare parameters.

        ``**kwargs`` pass through to the base (like the sibling nodes), so
        tests can construct with ``parameter_overrides=[...]``.
        """
        super().__init__("episode_recorder", **kwargs)

        # Parameters with descriptors for introspection (ros2 param describe)
        self.declare_parameter(
            "contract_path",
            "",
            ParameterDescriptor(description="Path to contract YAML file", read_only=True),
        )
        self.declare_parameter(
            "bag_base_dir",
            "datasets/bags",
            ParameterDescriptor(
                description="Base directory for bag storage; relative paths "
                "resolve against the current working directory, like "
                "`ros2 bag record`",
                read_only=True,
            ),
        )
        self.declare_parameter(
            "storage_id",
            "mcap",
            ParameterDescriptor(description="Bag storage format (mcap, sqlite3)", read_only=True),
        )
        self.declare_parameter(
            "exclude_topics",
            [""],
            ParameterDescriptor(
                description="Regex patterns for topics to exclude from auto-recording "
                "(same syntax as ros2 bag record --exclude)",
                read_only=True,
            ),
        )
        self.declare_parameter(
            "default_max_duration",
            0.0,
            ParameterDescriptor(
                description="Maximum recording duration in seconds (0 or negative = record until stopped, no limit). "
                "Read at each recording start."
            ),
        )
        self.declare_parameter(
            "feedback_rate_hz",
            2.0,
            positive_rate_descriptor("Rate for publishing action feedback. Read at each recording start."),
        )
        self.declare_parameter(
            "record_all",
            True,
            ParameterDescriptor(
                description="Record every topic on the graph (like `ros2 bag record -a`), not just contract topics",
                read_only=True,
            ),
        )
        self.declare_parameter(
            "embed_contract",
            True,
            ParameterDescriptor(
                description="Embed the contract YAML (text + hash) into the bag's metadata.yaml custom_data. "
                "The full file text, comments included, is copied into every bag",
                read_only=True,
            ),
        )

        # Initialize state variables (resources created in lifecycle callbacks)
        self._contract = None
        self._contract_text: str = ""
        self._bag_base: Path | None = None
        self._storage_id: str | None = None
        self._topics: list[tuple[str, str, QoSProfile]] = []  # (topic, type, qos)
        self._discovered_topics: list[tuple[str, str, QoSProfile]] = []
        self._exclude_regex: re.Pattern | None = None
        self._subs: dict[str, Any] = {}  # persistent subs (non-latched contract topics)
        self._episode_subs: dict[str, Any] = {}  # per-episode subs (latched contract + discovered)
        self._action_server: ActionServer | None = None
        self._cancel_service = None
        self._start_service = None
        self._delete_last_bag_service = None

        # Recording state. The writer gates the subscription write path (its
        # existence IS "we are writing"); the base's work gate (busy guard +
        # per-claim stop event) is claimed at accept time — before the writer
        # opens — so concurrent starts (service+service or service+action)
        # can't both proceed during the writer-open window. _cancel_requested
        # lets the cancel service ask _record to finish an action goal
        # without touching the goal handle.
        self._writer: rosbag2_py.SequentialWriter | None = None
        self._writer_lock = threading.Lock()
        self._messages_written = 0
        self._topic_msg_counts: dict[str, int] = {}
        self._cancel_requested = False
        self._cbg = ReentrantCallbackGroup()
        self._last_bag_dir: Path | None = None

        self.get_logger().info("Node created (unconfigured)")

    # -------------------- Lifecycle hooks --------------------

    def _setup(self) -> None:
        """Load the contract, create subscriptions, action server, and services.

        Every failure raises: the base logs the traceback and routes through
        on_error, which tears down whatever was partially built here.
        """
        contract_path = self.get_parameter("contract_path").value
        if not contract_path:
            raise ValueError("contract_path parameter required")

        self._contract = load_contract(contract_path)

        if self.get_parameter("embed_contract").value:
            self._contract_text = Path(contract_path).read_text()
        else:
            self._contract_text = ""

        self._bag_base = Path(self.get_parameter("bag_base_dir").value).expanduser()
        self._bag_base.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(f"Bag output directory: {self._bag_base.resolve()}")
        self._storage_id = self.get_parameter("storage_id").value

        # Build exclude regex: merge default patterns and user-specified
        # patterns into a single compiled pattern — same convention as
        # ``ros2 bag record --exclude``.
        user_patterns = tuple(p for p in self.get_parameter("exclude_topics").value if p)
        self._exclude_regex = re.compile("|".join(_DEFAULT_BLACKLIST + user_patterns))

        # Build topic list from contract
        self._topics = self._build_topic_list()

        # Persistent subscriptions for non-latched contract topics only
        # (callbacks no-op while no writer is open). Latched topics get
        # per-episode subscriptions in _open_writer so the RMW redelivers
        # their latched samples into each new bag.
        for topic, type_str, qos in self._topics:
            if qos.durability != DurabilityPolicy.TRANSIENT_LOCAL:
                self._subs[topic] = self._create_sub(topic, type_str, qos)

        # Create action server
        self._action_server = ActionServer(
            self,
            RecordEpisode,
            "record_episode",
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
            callback_group=self._cbg,
        )

        # Service to allow external callers to cancel an active recording
        # Useful for users who can't (or don't want to) interact with the
        # action protocol directly.
        self._cancel_service = self.create_service(
            Trigger,
            "~/cancel_recording",
            self._on_cancel_service,
            callback_group=self._cbg,
        )

        # Service to start recording without using the ROS2 action protocol.
        # Useful for Foxglove extensions and other clients that cannot call
        # the hidden _action/* services.
        self._start_service = self.create_service(
            StartRecording,
            "~/start_recording",
            self._on_start_service,
            callback_group=self._cbg,
        )

        # Service to delete the most recently completed bag directory.
        self._delete_last_bag_service = self.create_service(
            Trigger,
            "~/delete_last_bag",
            self._on_delete_last_bag_service,
            callback_group=self._cbg,
        )

        self.get_logger().info(f"Configured: robot_type={self._contract.robot_type}, topics={len(self._topics)}")

    def _teardown(self) -> None:
        """Destroy everything _setup() created. Tolerates partial state.

        Closes any open writer first: on_shutdown reaches here even when a
        wedged recording outlived the base's bounded wait, and the bag must
        still be finalized. Leaked services would stack up across
        cleanup->configure cycles.
        """
        self._close_writer()

        for sub in self._subs.values():
            self.destroy_subscription(sub)
        self._subs.clear()

        if self._action_server is not None:
            self._action_server.destroy()
            self._action_server = None

        for attr in ("_cancel_service", "_start_service", "_delete_last_bag_service"):
            service = getattr(self, attr)
            if service is not None:
                self.destroy_service(service)
                setattr(self, attr, None)

        self._contract = None
        self._contract_text = ""
        self._topics = []

    # -------------------- Topic and subscription management --------------------

    def _build_topic_list(self) -> list[tuple[str, str, QoSProfile]]:
        """
        Extract topics from contract.

        Includes:
        - Observation and action topics (from iter_specs)
        - Task topics
        - Adjunct topics (recorded but not mapped to keys)

        A topic referenced by several contract sections (e.g. one topic read as
        both an observation and a reward) yields ONE entry — duplicates used
        to create two live subscriptions (double-writing every message) and
        register the topic twice with the bag writer. Conflicting type or
        qos declarations for one topic raise ValueError (surfaces as
        a failed on_configure).

        Returns:
            List of (topic, type_str, qos) tuples

        """
        topics: list[tuple[str, str, QoSProfile]] = []

        for spec in iter_specs(self._contract):
            qos = qos_profile_from_dict(spec.source.channel.qos)
            topics.append((spec.source.channel.topic, spec.source.channel.type, qos))

        # Task channels
        for task in self._contract.tasks:
            qos = qos_profile_from_dict(task.channel.qos)
            topics.append((task.channel.topic, task.channel.type, qos))

        # Adjunct channels (record-only, no key mapping)
        for adj in self._contract.adjunct:
            qos = qos_profile_from_dict(adj.qos)
            topics.append((adj.topic, adj.type, qos))

        # If the node runs with simulation time, record /clock so playback
        # can drive sim time. use_sim_time is auto-declared on every rclpy
        # node (TimeSource.attach_node), so the read cannot fail.
        if bool(self.get_parameter("use_sim_time").value):
            topics.append(("/clock", "rosgraph_msgs/msg/Clock", qos_profile_from_dict(None)))

        return self._dedup_topics(topics)

    @staticmethod
    def _latched_sub_depth(configured_depth: int, publisher_count: int) -> int:
        """QoS depth for a per-episode TRANSIENT_LOCAL subscription.

        The contract's configured QoS depth is a wire/DDS matching hint
        (conventionally 1) and is not how many distinct publishers may be
        latched on the topic -- multiple independent nodes can publish the
        same TRANSIENT_LOCAL topic (e.g. robot_state_publisher plus a
        sensor-calibration static_transform_publisher both on /tf_static),
        each redelivering its own latched sample when the subscription is
        created. A depth lower than the live publisher count lets one
        redelivered sample evict another in the RMW queue before the
        executor drains it (verified against rmw_zenoh and cyclonedds).
        """
        return min(max(configured_depth, publisher_count), MAX_LATCHED_SUB_DEPTH)

    @staticmethod
    def _dedup_topics(
        topics: list[tuple[str, str, QoSProfile]],
    ) -> list[tuple[str, str, QoSProfile]]:
        """Collapse duplicate topic entries; reject conflicting declarations."""
        deduped: dict[str, tuple[str, str, QoSProfile]] = {}
        for entry in topics:
            topic, type_str, qos = entry
            prev = deduped.get(topic)
            if prev is None:
                deduped[topic] = entry
                continue
            if prev[1] != type_str:
                raise ValueError(
                    f"Topic '{topic}' appears in multiple contract sections "
                    f"with different types ({prev[1]} vs {type_str})"
                )
            if extract_qos_numeric_values(prev[2]) != extract_qos_numeric_values(qos):
                raise ValueError(
                    f"Topic '{topic}' appears in multiple contract sections with conflicting qos; unify the entries"
                )
            # Identical duplicate: keep the first entry.
        return list(deduped.values())

    def _discover_topics(self) -> list[tuple[str, str, QoSProfile]]:
        """Discover non-contract topics on the ROS2 graph.

        Returns topics not already in self._topics and not matching the
        exclude regex. Topics whose message type can't be imported are
        skipped with a warning — a broken third-party type must not abort
        the episode (contract topics still fail hard at configure).
        """
        known = {t[0] for t in self._topics}
        discovered: list[tuple[str, str, QoSProfile]] = []

        for topic, type_list in self.get_topic_names_and_types():
            if topic in known:
                continue
            if self._exclude_regex and self._exclude_regex.search(topic):
                continue
            type_str = type_list[0]
            if len(type_list) > 1:
                self.get_logger().warning(f"Topic {topic} has multiple types {type_list}; recording {type_str}")
            try:
                get_message(type_str)
            except Exception as e:
                self.get_logger().warning(f"Skipping {topic}: cannot import type {type_str}: {e}")
                continue

            # Subscription QoS adapted across ALL publishers, not copied from
            # the first one — a RELIABLE subscription would silently never
            # match a BEST_EFFORT publisher on the same topic.
            pubs = self.get_publishers_info_by_topic(topic)
            qos = adapt_publisher_qos(pubs, warn=lambda m, t=topic: self.get_logger().warning(f"Topic {t}: {m}"))

            discovered.append((topic, type_str, qos))

        if discovered:
            names = [t[0] for t in discovered]
            self.get_logger().info(f"Auto-discovered {len(discovered)} topics: {names}")

        return discovered

    def _create_sub(self, topic: str, type_str: str, qos: QoSProfile):
        """Create a subscription that writes to the bag whenever a writer is open."""
        msg_cls = get_message(type_str)

        def callback(msg: Any, _topic: str = topic) -> None:
            if self._writer is None:  # unlocked fast path; re-checked under the lock
                return
            # Receive time as bag timestamp (standard rosbag2 behavior); the
            # header.stamp inside the message is preserved for TF lookups.
            timestamp_ns = self.get_clock().now().nanoseconds
            # Serialize outside the lock: for image topics this is a ~MB copy,
            # and every topic's callback contends the one writer lock.
            serialized = serialize_message(msg)
            with self._writer_lock:
                if self._writer is None:
                    return
                try:
                    self._writer.write(_topic, serialized, timestamp_ns)
                    self._messages_written += 1
                    self._topic_msg_counts[_topic] = self._topic_msg_counts.get(_topic, 0) + 1
                except Exception as e:
                    self.get_logger().error(f"Write failed on {_topic}: {e}")
                    self._signal_stop()

        return self.create_subscription(msg_cls, topic, callback, qos, callback_group=self._cbg)

    # ---------- Action callbacks ----------

    def _arm_recording(self) -> str | None:
        """Claim the recording slot via the base work gate; reset the cancel flag.

        Returns the rejection reason (node inactive / recording in flight) or
        None on success. The stop event is armed by _try_claim_work at accept
        time, NOT in _record: a cancel or deactivate landing in the
        accept->execute window must have an event to set. The cancel flag
        resets here for the same reason — resetting it inside _record would
        eat a service cancel landing in that window.
        """
        reason = self._try_claim_work()
        if reason is None:
            self._cancel_requested = False
        return reason

    def _on_goal(self, _goal_request) -> GoalResponse:
        """Accept if active and not already recording/starting.

        Overrides the base to arm the cancel flag with the claim; the cancel
        callback stays the base's goal-id-routed default.
        """
        reason = self._arm_recording()
        if reason is not None:
            self.get_logger().warning(f"Goal rejected: {reason}")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_cancel_service(self, request, response):
        """
        Handle external Trigger service call to cancel recording.

        Signals the recording loop to stop and, for an action recording,
        records that the goal should finish early. The goal ends ABORTED with
        message 'Cancelled via service' (rcl only allows the CANCELED state
        from CANCELING, which requires an action-protocol cancel). The actual
        goal-handle transition is left to _record (the thread that owns the
        handle) to avoid two threads transitioning it concurrently.
        """
        if not self._busy.busy:
            response.success = False
            response.message = "No active recording"
            return response

        self.get_logger().info("cancel_recording service called: stopping recording")
        self._cancel_requested = True
        self._signal_stop()

        response.success = True
        response.message = "Cancel requested"
        return response

    def _on_start_service(self, request, response):
        """
        Handle StartRecording service call.

        Starts recording directly without the ROS2 action protocol.
        This is the primary interface for Foxglove extensions since the
        foxglove bridge cannot route to hidden _action/* services.
        """
        # Claim the recording slot before the writer exists so a concurrent
        # start (another service call or an action goal) is rejected.
        reason = self._arm_recording()
        if reason is not None:
            response.accepted = False
            response.message = f"Rejected: {reason}"
            return response

        prompt = request.prompt or ""
        self.get_logger().info(f"start_recording service called: prompt='{prompt}'")

        # Run the shared record loop in a background thread (the action path
        # runs it on the action executor thread instead).
        threading.Thread(target=self._record, args=(prompt,), daemon=True).start()

        response.accepted = True
        response.message = "Recording started"
        return response

    def _on_delete_last_bag_service(self, _request, response: Trigger.Response) -> Trigger.Response:
        """Delete the most recently completed bag directory."""
        if self._busy.busy:
            response.success = False
            # episode_keyboard_node's discard flow retries on this substring.
            response.message = "Cannot delete: recording in progress"
            return response
        if self._last_bag_dir is None:
            response.success = False
            response.message = "No bag to delete"
            return response

        bag_path = self._last_bag_dir
        try:
            if bag_path.exists():
                shutil.rmtree(bag_path)
                self._last_bag_dir = None
                self.get_logger().info(f"Deleted bag: {bag_path}")
                response.success = True
                response.message = f"Deleted: {bag_path.name}"
            else:
                response.success = False
                response.message = f"Bag path not found: {bag_path}"
        except Exception as e:
            self.get_logger().error(f"Failed to delete bag {bag_path}: {e}")
            response.success = False
            response.message = f"Delete failed: {e}"
        return response

    def _execute(self, goal_handle) -> RecordEpisode.Result:
        """Action-goal wrapper around the shared record loop."""
        return self._record(goal_handle.request.prompt or "", goal_handle)

    def _record(self, prompt: str, goal_handle=None) -> RecordEpisode.Result:
        """The one record loop, shared by both start paths.

        ``goal_handle=None`` means a service start: no feedback publishing, no
        cancel-request polling, and no terminal goal transition. Everything
        else — writer lifecycle, timeout, metadata, per-topic summary, state
        reconciliation — is identical by construction (the two paths used to
        be parallel copies and had drifted).

        The per-recording stop state was armed at accept time
        (_arm_recording) and is deliberately NOT reset here.
        """
        self._messages_written = 0
        self._topic_msg_counts = {}
        bag_dir = self._create_bag_dir()
        result = RecordEpisode.Result()
        result.bag_path = str(bag_dir)

        source = "action" if goal_handle is not None else "service"
        error: Exception | None = None
        writer_opened = False

        # _goal_work releases the claim on EVERY exit — including a
        # goal-handle transition raising — so no path can leave _busy held
        # and brick all subsequent recordings. It also binds the goal for
        # cancel routing (None for the service path: no routing).
        with self._goal_work(goal_handle) as stop_event:
            start = time.monotonic()
            try:
                max_duration = float(self.get_parameter("default_max_duration").value)
                # Range-validated at declare/set time (positive_rate_descriptor).
                feedback_rate = float(self.get_parameter("feedback_rate_hz").value)
                self.get_logger().info(f"Recording ({source}): {bag_dir}, max={max_duration}s")

                self._open_writer(bag_dir)
                writer_opened = True

                feedback = RecordEpisode.Feedback() if goal_handle is not None else None
                while not stop_event.wait(1.0 / feedback_rate):
                    elapsed = time.monotonic() - start

                    # Timeout (max_duration <= 0 means record until stopped)
                    remaining = max_duration - elapsed if max_duration > 0 else 0.0
                    if max_duration > 0 and remaining <= 0:
                        self.get_logger().info("Timeout reached")
                        break

                    if goal_handle is not None:
                        if goal_handle.is_cancel_requested:
                            break
                        # Read message count under lock for thread safety.
                        with self._writer_lock:
                            msg_count = self._messages_written
                        feedback.seconds_remaining = int(max(0.0, remaining))
                        feedback.messages_written = msg_count
                        feedback.status = "recording"
                        goal_handle.publish_feedback(feedback)
            except Exception as e:
                self.get_logger().error(f"Recording error: {e}")
                error = e

            # Epilogue — one path for success, cancel, timeout, and error.
            # Capture the discovered-topic list first: _close_writer clears it.
            discovered = list(self._discovered_topics)
            self._close_writer()
            if writer_opened:
                try:
                    self._write_metadata(bag_dir, prompt, self._contract_text)
                except RuntimeError as e:
                    # A partial bag with provenance is still a real error, but
                    # never mask an earlier one.
                    self.get_logger().error(f"Metadata error: {e}")
                    error = error or e
            if bag_dir.exists():
                # Failed partials stay deletable via ~/delete_last_bag.
                self._last_bag_dir = bag_dir

            result.messages_written = self._messages_written
            result.success = error is None
            result.message = "" if error is None else str(error)
            if error is None:
                self._log_topic_summary(bag_dir, discovered, time.monotonic() - start)

            # Terminal goal state. This thread (the goal-handle owner) is the
            # only place that transitions the handle; finish_goal uses only
            # rcl-legal transitions (a service cancel finishes as ABORTED —
            # CANCELED requires the CANCELING state, which only an
            # action-protocol cancel enters).
            if goal_handle is not None:
                finish_goal(
                    goal_handle,
                    result,
                    service_cancelled=self._cancel_requested,
                    success_message=f"Recorded {self._messages_written} messages",
                )
            return result

    def _log_topic_summary(self, bag_dir: Path, discovered_topics, elapsed: float) -> None:
        """Log per-topic counts so issues are visible without ros2 bag info."""
        lines = []
        for topic_name, _, _ in self._topics:
            count = self._topic_msg_counts.get(topic_name, 0)
            marker = " (!)" if count == 0 else ""
            lines.append(f"  {topic_name}: {count}{marker}")
        for topic_name, _, _ in discovered_topics:
            count = self._topic_msg_counts.get(topic_name, 0)
            if count > 0:
                lines.append(f"  {topic_name}: {count}")
        self.get_logger().info(
            f"Recorded {self._messages_written} messages to {bag_dir} ({elapsed:.1f}s)\n" + "\n".join(lines)
        )

    # ---------- rosbag2 helpers ----------

    def _create_bag_dir(self) -> Path:
        """Generate unique bag directory name."""
        t_ns = time.time_ns()
        sec, nsec = divmod(t_ns, 1_000_000_000)
        return self._bag_base / f"{sec:010d}_{nsec:09d}"

    def _open_writer(self, bag_dir: Path) -> None:
        """Discover topics, open the bag, register topics, then create per-episode subs."""
        if self.get_parameter("record_all").value:
            self._discovered_topics = self._discover_topics()

        storage_options = rosbag2_py.StorageOptions(
            uri=str(bag_dir),
            storage_id=self._storage_id,
        )
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )

        writer = rosbag2_py.SequentialWriter()
        writer.open(storage_options, converter_options)

        # Register contract + discovered topics with the ORIGINAL declared/
        # discovered QoS (never the widened subscription QoS below — the
        # offered profile is what playback re-offers).
        all_topics = self._topics + self._discovered_topics
        try:
            for idx, (topic, type_str, qos) in enumerate(all_topics):
                topic_info = rosbag2_py.TopicMetadata(
                    id=idx,
                    name=topic,
                    type=type_str,
                    serialization_format="cdr",
                    offered_qos_profiles=[qos_to_rosbag2(qos)],
                )
                writer.create_topic(topic_info)
        except Exception:
            writer.close()  # don't leak an unfinalized bag
            raise

        with self._writer_lock:
            self._writer = writer

        # Per-episode subscriptions, created only now that the writer is live
        # so everything they deliver takes the write path:
        # - all auto-discovered topics;
        # - latched (TRANSIENT_LOCAL) contract topics: the RMW redelivers each
        #   publisher's latched sample to the fresh subscription, landing
        #   pre-episode data (e.g. /tf_static) in every bag. The subscription
        #   depth is widened so one publisher's redelivered sample can't
        #   evict another's (see _latched_sub_depth).
        for topic, type_str, qos in self._discovered_topics:
            self._episode_subs[topic] = self._create_sub(topic, type_str, qos)
        for topic, type_str, qos in self._topics:
            if qos.durability == DurabilityPolicy.TRANSIENT_LOCAL:
                sub_qos = QoSProfile(
                    history=HistoryPolicy.KEEP_LAST,
                    depth=self._latched_sub_depth(qos.depth, self.count_publishers(topic)),
                    reliability=qos.reliability,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                )
                self._episode_subs[topic] = self._create_sub(topic, type_str, sub_qos)

    def _close_writer(self) -> None:
        """Close the writer and destroy per-episode subscriptions. Idempotent."""
        with self._writer_lock:
            if self._writer is not None:
                self._writer.close()  # Explicitly close to finalize MCAP indices
            self._writer = None
        # Atomic swap so two callers (record thread + _teardown) destroy disjoint sets.
        episode_subs, self._episode_subs = self._episode_subs, {}
        for sub in episode_subs.values():
            self.destroy_subscription(sub)
        self._discovered_topics = []

    def _write_metadata(self, bag_dir: Path, prompt: str, contract_text: str = "") -> None:
        """
        Write prompt and/or contract provenance to metadata.yaml as custom_data.

        Raises
        ------
            RuntimeError: If metadata.yaml cannot be written after retries.
                This is a fail-fast design - we don't silently lose the prompt
                or the contract.

        """
        entries: dict[str, str] = {}
        if prompt:
            entries[BAG_PROMPT_KEY] = prompt
        if contract_text:
            entries[BAG_CONTRACT_KEY] = contract_text
        if not entries:
            return

        # Retry loop: rosbag2 writes metadata.yaml asynchronously at bag
        # close, so the file may not exist yet on the first attempts. The
        # read-modify-write itself lives in bag_metadata (the porter reads
        # the same shape back).
        meta_path = bag_dir / "metadata.yaml"
        last_error: Exception | None = None

        for attempt in range(METADATA_RETRY_COUNT):
            try:
                update_custom_data(meta_path, entries)
                self.get_logger().debug(f"Wrote metadata on attempt {attempt + 1}")
                return
            except Exception as e:
                last_error = e
                self.get_logger().debug(f"Metadata write attempt {attempt + 1} failed: {e}")
                time.sleep(METADATA_RETRY_DELAY_SEC)

        # Fail fast - don't silently lose the prompt/contract
        raise RuntimeError(
            f"Failed to write metadata to {meta_path} after {METADATA_RETRY_COUNT} attempts. Last error: {last_error}"
        )


def main(args=None):
    """Run the episode recorder node."""
    return spin_lifecycle_node(EpisodeRecorderNode, args=args)


if __name__ == "__main__":
    sys.exit(main())
