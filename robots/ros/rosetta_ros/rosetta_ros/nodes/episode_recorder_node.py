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
    ros2 run rosetta_ros episode_recorder_node --ros-args \
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
import uuid
from pathlib import Path
from typing import Any

import rosbag2_py
from action_msgs.srv import CancelGoal
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rclpy.serialization import serialize_message
from rosidl_runtime_py.utilities import get_message
from std_srvs.srv import Trigger

from rosetta.contract.schema import parse_contract
from rosetta_interfaces.action import RecordEpisode
from rosetta_interfaces.msg import RecordEpisodeResult, RecordEpisodeStatus
from rosetta_interfaces.srv import StartRecording
from rosetta_ros.bag_metadata import write_provenance
from rosetta_ros.bag_topics import (
    Topic,
    adapt_publisher_qos,
    contract_topics,
    image_transport_auto_skips,
    latched_sub_depth,
    qos_to_rosbag2,
)
from rosetta_ros.node_utils import (
    RESULT_QOS,
    STATUS_QOS,
    TERMINATION_ERROR,
    TERMINATION_STOPPED,
    TERMINATION_TIMEOUT,
    copy_fields,
    finish_goal,
    positive_rate_descriptor,
    spin_lifecycle_node,
)
from rosetta_ros.rosetta_lifecycle_node import RosettaLifecycleNode

# Regex patterns for topics excluded from auto-recording by default.
# Uses the same regex convention as ``ros2 bag record --exclude``.
_DEFAULT_BLACKLIST = (
    r"^/rosout$",
    r"^/parameter_events$",
)


class EpisodeRecorderNode(RosettaLifecycleNode):
    """
    Stream-to-bag episode recorder with lifecycle and action interface.

    Design:
    - Non-latched contract topics get persistent subscriptions (created at
      configure) so an episode captures them from its very first instant.
    - Latched (TRANSIENT_LOCAL) contract topics and all auto-discovered
      topics get per-episode subscriptions, created after the bag writer is
      live. The RMW redelivers each publisher's latched sample to the fresh
      subscription. That is how pre-episode data (e.g. /tf_static) lands in
      every bag. It is the same mechanism ``ros2 bag record`` relies on.
    - The writer gates the write path: subscription callbacks write only while
      a writer is open. Concurrency is guarded separately by the base ``busy``
      property (a claim in flight), claimed at accept time before the writer
      opens, so two starts cannot both open a writer.
    """

    def __init__(self, **kwargs):
        """Initialize the episode recorder node and declare parameters.

        ``**kwargs`` pass through to the base (like the sibling nodes), so
        tests can construct with ``parameter_overrides=[...]``.
        """
        super().__init__("episode_recorder", **kwargs)

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
            "include_topics",
            [""],
            ParameterDescriptor(
                description="Regex patterns for topics to always auto-record, overriding "
                "exclude_topics and the one-stream-per-camera policy",
                read_only=True,
            ),
        )
        self.declare_parameter(
            "default_prompt",
            "",
            ParameterDescriptor(
                description="Task prompt used when a goal or service call leaves prompt empty. "
                "Read at each recording start. Empty means the bag stores no prompt, which is "
                "correct when the contract declares a `tasks` section instead."
            ),
        )
        self.declare_parameter(
            "default_max_duration_s",
            0.0,
            ParameterDescriptor(
                description="Maximum recording duration in seconds (0 or negative = record until stopped, no limit). "
                "Read at each recording start, and used only when the goal does not set max_duration_s."
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

        self._contract = None
        self._contract_text: str = ""
        self._bag_base: Path | None = None
        self._storage_id: str | None = None
        self._topics: list[Topic] = []  # (topic, type, qos)
        self._exclude_regex: re.Pattern | None = None
        self._include_regex: re.Pattern | None = None
        self._subs: dict[str, Any] = {}  # persistent subs (non-latched contract topics)
        self._episode_subs: dict[str, Any] = {}  # per-episode subs (latched contract + discovered)
        self._action_server: ActionServer | None = None
        self._cancel_client = None
        self._cancel_service = None
        self._start_service = None
        self._delete_last_bag_service = None
        self._status_pub = None
        self._result_pub = None

        self._writer: rosbag2_py.SequentialWriter | None = None
        self._writer_lock = threading.Lock()
        self._messages_written = 0
        self._topic_msg_counts: dict[str, int] = {}
        self._cbg = ReentrantCallbackGroup()
        # _bag_lock guards which bag ~/delete_last_bag targets: the recording
        # in progress (deleted when it ends) or the last finished one.
        self._bag_lock = threading.Lock()
        self._active_bag: Path | None = None
        self._discard_active = False
        self._last_bag_dir: Path | None = None

        self.get_logger().info("Node created (unconfigured)")

    # -------------------- Lifecycle hooks --------------------

    def setup(self) -> None:
        """Load the contract, create subscriptions, action server, and services.

        Every failure raises: the base logs the traceback and routes through
        on_error, which tears down whatever was partially built here.
        """
        contract_path = self.get_parameter("contract_path").value
        if not contract_path:
            raise ValueError("contract_path parameter required")

        # Single read: the text embedded into bag metadata is byte-identical
        # to the text the contract was parsed from, even if the file changes
        # on disk after configure.
        contract_text = Path(contract_path).read_text(encoding="utf-8")
        self._contract = parse_contract(contract_text, source=contract_path)
        self._contract_text = contract_text if self.get_parameter("embed_contract").value else ""

        self._bag_base = Path(self.get_parameter("bag_base_dir").value).expanduser()
        self._bag_base.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(f"Bag output directory: {self._bag_base.resolve()}")
        self._storage_id = self.get_parameter("storage_id").value

        # Merge default and user exclude patterns into one compiled regex.
        # Same convention as ``ros2 bag record --exclude``.
        user_patterns = tuple(p for p in self.get_parameter("exclude_topics").value if p)
        self._exclude_regex = re.compile("|".join(_DEFAULT_BLACKLIST + user_patterns))
        include_patterns = tuple(p for p in self.get_parameter("include_topics").value if p)
        self._include_regex = re.compile("|".join(include_patterns)) if include_patterns else None

        # use_sim_time is auto-declared on every rclpy node, so the read cannot fail.
        self._topics = contract_topics(self._contract, use_sim_time=self.get_parameter("use_sim_time").value)

        # Latched contract topics subscribe per episode instead (see the class docstring).
        for topic, type_str, qos in self._topics:
            if qos.durability != DurabilityPolicy.TRANSIENT_LOCAL:
                self._subs[topic] = self._create_sub(topic, type_str, qos)

        self._action_server = ActionServer(
            self,
            RecordEpisode,
            "record_episode",
            execute_callback=self._execute,
            goal_callback=self.on_goal,
            cancel_callback=self.on_cancel,
            callback_group=self._cbg,
        )

        # This server's own cancel service, so ~/cancel_recording performs a
        # real cancel. A relative name, so it expands under the node namespace.
        self._cancel_client = self.create_client(
            CancelGoal,
            "record_episode/_action/cancel_goal",
            callback_group=self._cbg,
        )

        # Service wrappers for clients that can call services but not actions (Foxglove).
        self._cancel_service = self.create_service(
            Trigger,
            "~/cancel_recording",
            self._on_cancel_service,
            callback_group=self._cbg,
        )

        self._start_service = self.create_service(
            StartRecording,
            "~/start_recording",
            self._on_start_service,
            callback_group=self._cbg,
        )

        self._delete_last_bag_service = self.create_service(
            Trigger,
            "~/delete_last_bag",
            self._on_delete_last_bag_service,
            callback_group=self._cbg,
        )

        # Feedback and result for every recording, including service starts,
        # whose callers get no action feedback.
        self._status_pub = self.create_publisher(RecordEpisodeStatus, "~/status", STATUS_QOS)
        self._result_pub = self.create_publisher(RecordEpisodeResult, "~/result", RESULT_QOS)

        self.get_logger().info(f"Configured: robot_type={self._contract.robot_type}, topics={len(self._topics)}")

    def teardown(self) -> None:
        """Destroy everything setup() created. Tolerates partial state.

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

        if self._cancel_client is not None:
            self.destroy_client(self._cancel_client)
            self._cancel_client = None

        for attr in ("_cancel_service", "_start_service", "_delete_last_bag_service"):
            service = getattr(self, attr)
            if service is not None:
                self.destroy_service(service)
                setattr(self, attr, None)

        for attr in ("_status_pub", "_result_pub"):
            publisher = getattr(self, attr)
            if publisher is not None:
                self.destroy_publisher(publisher)
                setattr(self, attr, None)

        self._contract = None
        self._contract_text = ""
        self._topics = []

    # -------------------- Topic and subscription management --------------------

    def _discover_topics(self) -> list[Topic]:
        """Discover non-contract topics on the ROS2 graph.

        Returns topics not already in self._topics and not matching the
        exclude regex. Topics whose message type cannot be imported are
        skipped with a warning. A broken third-party type must not abort the
        episode. Contract topics still fail hard at configure.

        Returns:
            List of (topic, type_str, qos) tuples for graph topics to record.

        """
        known = {t[0] for t in self._topics}
        discovered: list[Topic] = []

        def included(t: str) -> bool:
            return bool(self._include_regex and self._include_regex.search(t))

        def excluded(t: str) -> bool:
            return bool(self._exclude_regex and self._exclude_regex.search(t))

        topic_types = self.get_topic_names_and_types()
        image_skips = image_transport_auto_skips(
            {t: tl[0] for t, tl in topic_types}, known, include=included, exclude=excluded
        )
        if image_skips:
            self.get_logger().info(
                f"Skipping {len(image_skips)} redundant image_transport streams: {sorted(image_skips)}"
            )

        for topic, type_list in topic_types:
            if topic in known:
                continue
            # include_topics pins a topic past both the camera-stream policy
            # and exclude_topics; contract topics aren't discovery's to veto.
            if not included(topic) and (topic in image_skips or excluded(topic)):
                continue
            type_str = type_list[0]
            if len(type_list) > 1:
                self.get_logger().warning(f"Topic {topic} has multiple types {type_list}; recording {type_str}")
            try:
                get_message(type_str)
            except Exception as e:
                self.get_logger().warning(f"Skipping {topic}: cannot import type {type_str}: {e}")
                continue

            # Adapt QoS across all publishers, not just the first. A RELIABLE
            # subscription would silently never match a BEST_EFFORT publisher
            # on the same topic.
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
            write_error: str | None = None
            with self._writer_lock:
                if self._writer is None:
                    return
                try:
                    self._writer.write(_topic, serialized, timestamp_ns)
                    self._messages_written += 1
                    self._topic_msg_counts[_topic] = self._topic_msg_counts.get(_topic, 0) + 1
                except Exception as e:
                    write_error = f"Write failed on {_topic}: {e}"
            if write_error is not None:
                # Outside the writer lock: signal_stop takes the work gate,
                # and the two locks are never held together.
                self.get_logger().error(write_error)
                self.signal_stop(TERMINATION_ERROR, write_error)

        return self.create_subscription(msg_cls, topic, callback, qos, callback_group=self._cbg)

    # ---------- Action callbacks ----------

    def _on_cancel_service(self, request, response):
        """Cancel the active recording. See :meth:`RosettaLifecycleNode.cancel_current_work`."""
        return self.handle_cancel_service(response, self._cancel_client, what="recording")

    def _on_start_service(self, request, response):
        """Start a recording on a background thread, without the action protocol."""
        return self.handle_start_service(response, self._record, (request.prompt or "",), what="recording")

    def try_claim_work(self) -> str | None:
        """Claim the slot; the claimed recording becomes the last one, so no older bag stays deletable."""
        reason = RosettaLifecycleNode.try_claim_work(self)
        if reason is None:
            with self._bag_lock:
                self._last_bag_dir = None
        return reason

    def _on_delete_last_bag_service(self, _request, response: Trigger.Response) -> Trigger.Response:
        """Delete the recording in progress when it ends, or else the last finished bag.

        A live writer's directory is never removed here: the record loop
        deletes it itself once the writer is closed. Returns success False with
        a reason when there is nothing to delete or the removal fails.
        """
        with self._bag_lock:
            if self._active_bag is not None:
                self._discard_active = True
                response.success = True
                response.message = f"Will delete {self._active_bag.name} when the recording ends"
                return response
            bag_path, self._last_bag_dir = self._last_bag_dir, None
            starting = bag_path is None and self.busy

        if bag_path is None:
            response.success = False
            response.message = "A recording is starting; it has no bag yet" if starting else "No bag to delete"
            return response
        try:
            shutil.rmtree(bag_path)
        except OSError as e:
            self.get_logger().error(f"Failed to delete bag {bag_path}: {e}")
            response.success = False
            response.message = f"Delete failed: {e}"
            return response
        self.get_logger().info(f"Deleted bag: {bag_path}")
        response.success = True
        response.message = f"Deleted: {bag_path.name}"
        return response

    def _execute(self, goal_handle) -> RecordEpisode.Result:
        """Action-goal wrapper around the shared record loop."""
        return self._record(
            goal_handle.request.prompt or "",
            goal_handle,
            max_duration=goal_handle.request.max_duration_s,
        )

    def _record(self, prompt: str, goal_handle=None, max_duration: float = 0.0) -> RecordEpisode.Result:
        """The one record loop, shared by both start paths.

        ``goal_handle=None`` marks a service start. That path skips feedback
        publishing and the terminal goal transition; everything else is shared.

        The per-recording stop event and stop reason were armed at accept time
        by try_claim_work. They are deliberately not reset here: a cancel or
        deactivate landing in the accept->execute window must not be lost.

        Args:
            prompt: Task prompt stored in the bag metadata. Empty falls back to
                the ``default_prompt`` parameter.
            goal_handle: Action goal handle, or None for a service start.
            max_duration: Auto-stop after this many seconds. 0 or negative
                falls back to the ``default_max_duration_s`` parameter.

        Returns:
            The populated RecordEpisode.Result.

        """
        prompt = prompt or self.get_parameter("default_prompt").value

        self._messages_written = 0
        self._topic_msg_counts = {}
        bag_dir = self._create_bag_dir()
        with self._bag_lock:
            self._active_bag, self._discard_active = bag_dir, False
        result = RecordEpisode.Result()
        result.bag_path = str(bag_dir)

        source = "action" if goal_handle is not None else "service"
        error: Exception | None = None
        writer_opened = False
        discovered: list[Topic] = []
        # Set only when the loop ends on its own terms. A stop somebody else
        # asked for is recorded on the node instead (see the epilogue).
        loop_reason: str | None = None

        # goal_work binds the goal for cancel routing and releases the claim on every exit.
        with self.goal_work(goal_handle) as stop_event:
            start = time.monotonic()
            try:
                if max_duration <= 0.0:
                    max_duration = float(self.get_parameter("default_max_duration_s").value)
                # Range-validated at declare/set time (positive_rate_descriptor).
                feedback_rate = float(self.get_parameter("feedback_rate_hz").value)
                self.get_logger().info(f"Recording ({source}): {bag_dir}, max={max_duration}s")

                discovered = self._open_writer(bag_dir)
                writer_opened = True

                while not stop_event.wait(1.0 / feedback_rate):
                    elapsed = time.monotonic() - start

                    # max_duration <= 0 means record until stopped.
                    if 0 < max_duration <= elapsed:
                        self.get_logger().info("Timeout reached")
                        loop_reason = TERMINATION_TIMEOUT
                        break

                    with self._writer_lock:
                        msg_count = self._messages_written
                    status = RecordEpisodeStatus(elapsed_s=elapsed, messages_written=msg_count)
                    self.publish_output(self._status_pub, status)
                    if goal_handle is not None:
                        goal_handle.publish_feedback(copy_fields(status, RecordEpisode.Feedback()))
            except Exception as e:
                self.get_logger().error(f"Recording error: {e}")
                error = e

            # Epilogue. One path for success, cancel, timeout, and error.
            self._close_writer()
            if writer_opened:
                try:
                    write_provenance(
                        bag_dir,
                        prompt=prompt,
                        contract_text=self._contract_text,
                        goal_id=self._goal_id_str(goal_handle),
                    )
                except Exception as e:
                    # A partial bag with provenance is still a real error, but
                    # never mask an earlier one.
                    self.get_logger().error(f"Metadata error: {e}")
                    error = error or e
            with self._bag_lock:
                discard, self._discard_active, self._active_bag = self._discard_active, False, None
                # This take is now the last one. A discarded take, or one whose
                # writer never created its directory, leaves nothing to delete:
                # never fall back to an older bag. Failed partials stay deletable.
                self._last_bag_dir = bag_dir if not discard and bag_dir.exists() else None
            if discard:
                shutil.rmtree(bag_dir, ignore_errors=True)
                self.get_logger().info(f"Deleted bag: {bag_dir}")

            result.messages_written = self._messages_written

            # Why the recording ended, in precedence order:
            #   1. an exception raised on this thread
            #   2. a stop somebody asked for, which named itself (a cancel, a
            #      deactivate, a failed bag write)
            #   3. the loop's own decision (a timeout)
            #   4. nothing above: the loop was told to stop with no reason
            # Signalled beats loop-decided so a failed write landing in the same
            # tick as a timeout still reports as an error rather than a clean
            # finish.
            # ``message`` carries only what no other field does: the detail of
            # something going wrong. It stays empty on a clean end rather than
            # restating messages_written back at the caller.
            signalled, detail = self.stop_reason, self.stop_detail
            if error is not None:
                result.termination_reason = TERMINATION_ERROR
                result.message = str(error)
            elif signalled is not None:
                result.termination_reason = signalled
                result.message = detail
            else:
                result.termination_reason = loop_reason or TERMINATION_STOPPED

            if result.termination_reason != TERMINATION_ERROR:
                self._log_topic_summary(bag_dir, discovered, time.monotonic() - start)

            # Terminal goal state. This thread owns the goal handle and is the
            # only place that transitions it; finish_goal uses only rcl-legal
            # transitions and reads the reason set just above.
            self.publish_output(self._result_pub, copy_fields(result, RecordEpisodeResult()))
            if goal_handle is not None:
                finish_goal(goal_handle, result)
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

    def _open_writer(self, bag_dir: Path) -> list[Topic]:
        """Discover topics, open the bag, register topics, then create per-episode subs.

        Returns the discovered (non-contract) topics being recorded.
        """
        discovered = self._discover_topics() if self.get_parameter("record_all").value else []

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

        # Register contract and discovered topics with the original declared
        # or discovered QoS. Never use the widened subscription QoS built
        # below. The offered profile recorded here is what playback re-offers.
        all_topics = self._topics + discovered
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

        # Per-episode subscriptions (see the class docstring), created only now
        # that the writer is live. Latched depth is widened so one publisher's
        # redelivered sample can't evict another's.
        for topic, type_str, qos in discovered:
            self._episode_subs[topic] = self._create_sub(topic, type_str, qos)
        for topic, type_str, qos in self._topics:
            if qos.durability == DurabilityPolicy.TRANSIENT_LOCAL:
                sub_qos = QoSProfile(
                    history=HistoryPolicy.KEEP_LAST,
                    depth=latched_sub_depth(qos.depth, self.count_publishers(topic)),
                    reliability=qos.reliability,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                )
                self._episode_subs[topic] = self._create_sub(topic, type_str, sub_qos)
        return discovered

    def _close_writer(self) -> None:
        """Close the writer and destroy per-episode subscriptions. Idempotent."""
        with self._writer_lock:
            if self._writer is not None:
                self._writer.close()  # Explicitly close to finalize MCAP indices
            self._writer = None
        # Atomic swap so two callers (record thread + teardown) destroy disjoint sets.
        episode_subs, self._episode_subs = self._episode_subs, {}
        for sub in episode_subs.values():
            self.destroy_subscription(sub)

    @staticmethod
    def _goal_id_str(goal_handle) -> str:
        """The goal's UUID as canonical hex, or "" for a service start.

        ``goal_id.uuid`` is a 16-byte array on the wire; bag metadata is YAML
        read by humans and by the porter, so store the string form.
        """
        if goal_handle is None:
            return ""
        return str(uuid.UUID(bytes=bytes(goal_handle.goal_id.uuid)))


def main(args=None):
    """Run the episode recorder node."""
    return spin_lifecycle_node(EpisodeRecorderNode, args=args)


if __name__ == "__main__":
    sys.exit(main())
