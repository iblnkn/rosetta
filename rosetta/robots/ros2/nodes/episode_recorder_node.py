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

import hashlib
import re
import shutil
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import rclpy
import rosbag2_py
import yaml
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.qos import (
    QoSProfile,
)
from rclpy.serialization import serialize_message
from rosetta_interfaces.action import RecordEpisode
from rosetta_interfaces.srv import StartRecording
from rosidl_runtime_py.utilities import get_message
from std_srvs.srv import Trigger

from rosetta.contract.schema import load_contract
from rosetta.contract.specs import iter_specs
from rosetta.robots.ros2 import decoders as _decoders  # noqa: F401 - registers decoders
from rosetta.robots.ros2 import encoders as _encoders  # noqa: F401 - registers encoders
from rosetta.robots.ros2.nodes.node_utils import (
    BusyGuard,
    extract_qos_numeric_values,
    finish_goal,
    get_qos_depth,
    is_jazzy_or_newer,
    is_transient_local,
    wait_until,
)
from rosetta.robots.ros2.ros2_utils import qos_profile_from_dict

# Bag metadata keys
BAG_METADATA_KEY = "rosbag2_bagfile_information"
BAG_CUSTOM_DATA_KEY = "custom_data"
BAG_PROMPT_KEY = "lerobot.operator_prompt"
BAG_CONTRACT_KEY = "rosetta.contract_yaml"
BAG_CONTRACT_HASH_KEY = "rosetta.contract_hash"

# ---------- Constants ----------

# Metadata file retry settings (internal implementation detail)
METADATA_RETRY_COUNT = 10
METADATA_RETRY_DELAY_SEC = 0.1
# Maximum serialized bytes to buffer for a retained message (4 MiB)
MAX_BUFFER_BYTES = 4 * 1024 * 1024
# Safety ceiling on how many distinct latched messages we'll retain per
# topic, regardless of how many publishers are discovered (see _create_sub).
MAX_BUFFERED_MESSAGES_PER_TOPIC = 64

# ROS2 distribution compatibility flag
# Uses common utilities from ros2_compat module
_IS_JAZZY = is_jazzy_or_newer()

# Regex patterns for topics excluded from auto-recording by default.
# Uses the same regex convention as ``ros2 bag record --exclude``.
_DEFAULT_BLACKLIST = (
    r"^/rosout$",
    r"^/parameter_events$",
)


class EpisodeRecorderNode(LifecycleNode):
    """
    Stream-to-bag episode recorder with lifecycle and action interface.

    Follows rosbag2_py tutorial patterns:
    - SequentialWriter with StorageOptions/ConverterOptions
    - TopicMetadata for topic registration
    - serialize_message() for writing
    """

    def __init__(self):
        """Initialize the episode recorder node and declare parameters."""
        # Initialize with enable_logger_service on Jazzy (not supported in Humble)
        # The logger service allows runtime configuration of log levels via
        # ros2 service call /node_name/set_logger_level ...
        # In Humble, logger services are always enabled by default.
        if _IS_JAZZY:
            super().__init__("episode_recorder", enable_logger_service=True)
        else:
            super().__init__("episode_recorder")

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
                description="Maximum recording duration in seconds (0 or negative = record until stopped, no limit)"
            ),
        )
        self.declare_parameter(
            "feedback_rate_hz",
            2.0,
            ParameterDescriptor(description="Rate for publishing action feedback"),
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
                description="Embed the contract YAML (text + hash) into the bag's metadata.yaml custom_data",
                read_only=True,
            ),
        )

        # Initialize state variables (resources created in lifecycle callbacks)
        self._contract = None
        self._contract_text: str = ""
        self._contract_hash: str = ""
        self._bag_base: Path | None = None
        self._storage_id: str | None = None
        self._default_max_duration: float = 0.0
        self._feedback_rate_hz: float = 2.0
        self._topics: list[tuple[str, str, QoSProfile | int]] = []  # (topic, type, qos)
        self._discovered_topics: list[tuple[str, str, QoSProfile | int]] = []
        self._exclude_regex: re.Pattern | None = None
        self._subs: dict[str, Any] = {}  # topic -> subscription object
        self._action_server: ActionServer | None = None
        self._cancel_service = None
        self._start_service = None
        self._delete_last_bag_service = None
        self._accepting_goals = False

        # Recording state
        self._writer: rosbag2_py.SequentialWriter | None = None
        self._writer_lock = threading.Lock()
        # _is_recording stays strictly tied to "writer exists" -- it gates the
        # live-write path in the subscription callback, so it must only be true
        # while self._writer is not None.
        self._is_recording = False
        # Accept-time guard: claimed the moment a start is accepted (before
        # the writer opens) so concurrent starts (service+service or
        # service+action) can't both proceed during the writer-open window.
        self._busy = BusyGuard()
        self._messages_written = 0
        self._topic_msg_counts: dict[str, int] = {}
        self._stop_event = threading.Event()
        # _cancel_requested lets the cancel service ask _execute to finish the
        # goal as CANCELED without the service thread touching the goal handle.
        self._cancel_requested = False
        self._goal_handle = None
        self._cbg = ReentrantCallbackGroup()
        self._last_bag_dir: Optional[Path] = None
        # Buffers for TRANSIENT_LOCAL messages (like /tf_static)
        # Each buffer is a deque limited by history depth -- see _create_sub
        # for why that's max(configured QoS depth, live publisher count),
        # not just the configured depth.
        self._buffers: dict[str, deque] = {}
        self._buffer_lock = threading.Lock()

        self.get_logger().info("Node created (unconfigured)")

    # -------------------- Lifecycle callbacks --------------------

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Load contract, create subscriptions and action server."""
        try:
            contract_path = self.get_parameter("contract_path").value
            if not contract_path:
                self.get_logger().error("contract_path parameter required")
                return TransitionCallbackReturn.FAILURE

            try:
                self._contract = load_contract(contract_path)
            except Exception as e:
                self.get_logger().error(f"Failed to load contract: {e}")
                return TransitionCallbackReturn.FAILURE

            if self.get_parameter("embed_contract").value:
                self._contract_text = Path(contract_path).read_text()
                self._contract_hash = hashlib.sha256(self._contract_text.encode()).hexdigest()
            else:
                self._contract_text = ""
                self._contract_hash = ""

            self._bag_base = Path(self.get_parameter("bag_base_dir").value).expanduser()
            self._bag_base.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(f"Bag output directory: {self._bag_base.resolve()}")
            self._storage_id = self.get_parameter("storage_id").value
            self._default_max_duration = self.get_parameter("default_max_duration").value
            self._feedback_rate_hz = self.get_parameter("feedback_rate_hz").value

            # Build exclude regex: merge default patterns, user-specified
            # patterns, and this node's own topics into a single compiled
            # pattern — same convention as ``ros2 bag record --exclude``.
            user_patterns = tuple(p for p in self.get_parameter("exclude_topics").value if p)
            all_patterns = _DEFAULT_BLACKLIST + user_patterns
            self._exclude_regex = re.compile("|".join(all_patterns))

            # Build topic list from contract
            self._topics = self._build_topic_list()

            # Create subscriptions (callbacks no-op when not recording)
            for topic, type_str, qos in self._topics:
                sub = self._create_sub(topic, type_str, qos)
                self._subs[topic] = sub

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
            # action protocol directly. This sets the internal stop event and
            # attempts to transition the current goal to the canceled state.
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
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"Configuration failed: {e}", throttle_duration_sec=1.0)
            import traceback

            self.get_logger().error(f"Traceback: {traceback.format_exc()}", throttle_duration_sec=1.0)
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Enable goal acceptance."""
        self._accepting_goals = True
        self.get_logger().info("Activated and ready for recording")
        return super().on_activate(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Stop accepting goals and stop any in-progress recording."""
        self._accepting_goals = False

        # Stop any in-progress recording
        if self._is_recording:
            self.get_logger().info("Stopping in-progress recording...")
            self._stop_event.set()

            # Wait for recording to complete
            if not wait_until(lambda: not self._is_recording, timeout=5.0):
                self.get_logger().warning("Recording did not stop within timeout")

        self.get_logger().info("Deactivated")
        return super().on_deactivate(state)

    def _destroy_resources(self) -> None:
        """Destroy everything on_configure created (subs, action server, services).

        Leaked services would stack up across cleanup->configure cycles.
        Safe to call when nothing was created.
        """
        for sub in self._subs.values():
            self.destroy_subscription(sub)
        self._subs.clear()

        if self._action_server is not None:
            self.destroy_action_server(self._action_server)
            self._action_server = None

        for attr in ("_cancel_service", "_start_service", "_delete_last_bag_service"):
            service = getattr(self, attr)
            if service is not None:
                self.destroy_service(service)
                setattr(self, attr, None)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Release resources."""
        self._destroy_resources()

        # Clear state
        self._contract = None
        self._contract_text = ""
        self._contract_hash = ""
        self._topics = []

        self.get_logger().info("Cleaned up")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Clean up resources before destruction."""
        self._accepting_goals = False
        self._stop_event.set()
        self._close_writer()
        self._destroy_resources()

        self.get_logger().info("Shutdown complete")
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Handle errors by cleaning up resources."""
        self.get_logger().error(f"Error occurred in state: {state.label}")

        try:
            self._accepting_goals = False
            self._stop_event.set()
            self._close_writer()
            self._destroy_resources()
        except Exception as e:
            self.get_logger().error(f"Error during cleanup: {e}")

        return TransitionCallbackReturn.SUCCESS

    # -------------------- Topic and subscription management --------------------

    def _build_topic_list(self) -> list[tuple[str, str, QoSProfile | int, str]]:
        """
        Extract topics from contract.

        Includes:
        - Observation and action topics (from iter_specs)
        - Task topics
        - Extra topics (recording.extra_topics) - recorded but not mapped to keys

        A topic referenced by several contract sections (e.g. one topic read as
        both an observation and a reward) yields ONE entry — duplicates used
        to create two live subscriptions (double-writing every message) and
        register the topic twice with the bag writer. Conflicting type or
        qos declarations for one topic raise ValueError (surfaces as
        a failed on_configure).

        Returns:
            List of (topic, type_str, qos) tuples

        """
        topics: list[tuple[str, str, QoSProfile | int]] = []

        for spec in iter_specs(self._contract):
            qos = qos_profile_from_dict(spec.source.channel.qos) or 10
            topics.append((spec.source.channel.topic, spec.source.channel.type, qos))

        # Task channels
        for task in self._contract.tasks:
            qos = qos_profile_from_dict(task.channel.qos) or 10
            topics.append((task.channel.topic, task.channel.type, qos))

        # Adjunct channels (record-only, no key mapping)
        for adj in self._contract.adjunct:
            qos = qos_profile_from_dict(adj.qos) or 10
            topics.append((adj.topic, adj.type, qos))

        # If node is running with simulation time enabled, record the /clock
        # topic so playback can drive sim time. Use a safe get in case the
        # parameter wasn't declared by the launcher.
        try:
            use_sim = bool(self.get_parameter("use_sim_time").value)
        except Exception:
            use_sim = False

        if use_sim:
            # Use the standard ROS2 clock message type. QoS depth 10 is a
            # reasonable default for clock topic traffic.
            topics.append(("/clock", "rosgraph_msgs/msg/Clock", 10))

        return self._dedup_topics(topics)

    @staticmethod
    def _latched_buffer_depth(configured_depth: int, publisher_count: int) -> int:
        """How many distinct TRANSIENT_LOCAL messages to retain for a topic.

        The contract's configured QoS depth is a wire/DDS matching hint
        (conventionally 1) and is not how many distinct publishers may be
        latched on the topic -- multiple independent nodes can publish the
        same TRANSIENT_LOCAL topic (e.g. robot_state_publisher plus a
        sensor-calibration static_transform_publisher both on /tf_static),
        each delivering its own separate latched sample. Retaining fewer
        messages than there are live publishers would let one evict another.
        """
        return min(max(configured_depth, publisher_count), MAX_BUFFERED_MESSAGES_PER_TOPIC)

    @staticmethod
    def _dedup_topics(
        topics: list[tuple[str, str, QoSProfile | int]],
    ) -> list[tuple[str, str, QoSProfile | int]]:
        """Collapse duplicate topic entries; reject conflicting declarations."""
        deduped: dict[str, tuple[str, str, QoSProfile | int]] = {}
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

    def _discover_topics(self) -> list[tuple[str, str, QoSProfile | int]]:
        """Discover non-contract topics on the ROS2 graph.

        Returns topics not already in self._topics and not matching the
        exclude regex. Each tuple is (topic_name, type_str, qos).
        """
        known = {t[0] for t in self._topics}
        discovered: list[tuple[str, str, QoSProfile | int]] = []

        for topic, type_list in self.get_topic_names_and_types():
            if topic in known:
                continue
            if self._exclude_regex and self._exclude_regex.search(topic):
                continue
            if not type_list:
                continue
            type_str = type_list[0]

            # Match QoS from existing publishers
            pubs = self.get_publishers_info_by_topic(topic)
            qos = pubs[0].qos_profile if pubs else QoSProfile(depth=10)

            discovered.append((topic, type_str, qos))

        if discovered:
            names = [t[0] for t in discovered]
            self.get_logger().info(f"Auto-discovered {len(discovered)} topics: {names}")

        return discovered

    def _cleanup_discovered_subs(self) -> None:
        """Destroy subscriptions for auto-discovered topics."""
        for topic, _, _ in self._discovered_topics:
            if topic in self._subs:
                self.destroy_subscription(self._subs.pop(topic))
        self._discovered_topics = []

    def _create_sub(self, topic: str, type_str: str, qos: QoSProfile | int):
        """Create subscription that writes to bag when recording.

        TRANSIENT_LOCAL (latched) topics are buffered while idle so the
        latched message lands in the bag even though it was published before
        the episode started; everything else is written live only.
        """
        msg_cls = get_message(type_str)

        def callback(msg: Any, _topic: str = topic) -> None:
            timestamp_ns = self.get_clock().now().nanoseconds
            # Buffer TRANSIENT_LOCAL messages when not recording
            if not self._is_recording:
                is_tl = is_transient_local(qos)

                if is_tl:
                    # See _latched_buffer_depth: sized to the live publisher
                    # count, not just the configured QoS depth.
                    history_depth = self._latched_buffer_depth(
                        get_qos_depth(qos), self.count_publishers(_topic)
                    )
                    try:
                        serialized = serialize_message(msg)
                        if len(serialized) <= MAX_BUFFER_BYTES:
                            with self._buffer_lock:
                                if _topic not in self._buffers:
                                    self._buffers[_topic] = deque()
                                self._buffers[_topic].append((serialized, timestamp_ns))
                                # Enforce history depth limit
                                while len(self._buffers[_topic]) > history_depth:
                                    self._buffers[_topic].popleft()
                    except Exception:
                        pass  # Best-effort buffering
                return

            # Write live message to bag
            with self._writer_lock:
                if self._writer is None:
                    return
                try:
                    # Use receive time as bag timestamp (standard rosbag2 behavior)
                    # The header.stamp inside the message is preserved for TF lookups
                    self._writer.write(
                        _topic,
                        serialize_message(msg),
                        timestamp_ns,
                    )
                    self._messages_written += 1
                    self._topic_msg_counts[_topic] = self._topic_msg_counts.get(_topic, 0) + 1
                except Exception as e:
                    self.get_logger().error(f"Write failed on {_topic}: {e}")
                    self._stop_event.set()

        return self.create_subscription(msg_cls, topic, callback, qos, callback_group=self._cbg)

    # ---------- Action callbacks ----------

    def _on_goal(self, goal_request) -> GoalResponse:
        """Accept if active and not already recording/starting."""
        self.get_logger().info("Received goal request")
        if not self._accepting_goals:
            self.get_logger().warning("Rejected: node not active")
            return GoalResponse.REJECT
        if not self._busy.try_acquire():
            self.get_logger().warning("Rejected: already recording")
            return GoalResponse.REJECT
        self.get_logger().info("Goal accepted")
        return GoalResponse.ACCEPT

    def _on_cancel(self, _goal_handle) -> CancelResponse:
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info("Received cancel request")
        self._stop_event.set()
        return CancelResponse.ACCEPT

    def _on_cancel_service(self, request, response):
        """
        Handle external Trigger service call to cancel recording.

        Signals the recording loop to stop and, for an action recording,
        records that the goal should finish early. The goal ends ABORTED with
        message 'Cancelled via service' (rcl only allows the CANCELED state
        from CANCELING, which requires an action-protocol cancel). The actual
        goal-handle transition is left to _execute (the executor thread that
        owns the handle) to avoid two threads transitioning it concurrently.
        """
        if not self._is_recording:
            response.success = False
            response.message = "No active recording"
            return response

        self.get_logger().info("cancel_recording service called: stopping recording")
        # Ask _execute to finish the goal as CANCELED, then signal the loop to
        # stop. _execute (the goal-handle owner) performs the terminal
        # transition; the service thread never touches the handle.
        self._cancel_requested = True
        self._stop_event.set()

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
        if not self._accepting_goals:
            response.accepted = False
            response.message = "Node not active"
            return response

        # Claim the recording slot before the writer exists so a concurrent
        # start (another service call or an action goal) is rejected.
        if not self._busy.try_acquire():
            response.accepted = False
            response.message = "Already recording"
            return response

        prompt = request.prompt or ""
        self.get_logger().info(f"start_recording service called: prompt='{prompt}'")

        # Run the shared record loop in a background thread (the action path
        # runs it on the action executor thread instead).
        record_thread = threading.Thread(
            target=self._record,
            args=(prompt,),
            daemon=True,
        )
        record_thread.start()

        response.accepted = True
        response.message = "Recording started"
        return response

    def _on_delete_last_bag_service(self, _request, response: Trigger.Response) -> Trigger.Response:
        """Delete the most recently completed bag directory."""
        if self._is_recording:
            response.success = False
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
        self._goal_handle = goal_handle
        try:
            return self._record(goal_handle.request.prompt or "", goal_handle)
        finally:
            self._goal_handle = None

    def _record(self, prompt: str, goal_handle=None) -> RecordEpisode.Result:
        """The one record loop, shared by both start paths.

        ``goal_handle=None`` means a service start: no feedback publishing, no
        cancel-request polling, and no terminal goal transition. Everything
        else — writer lifecycle, timeout, metadata, per-topic summary, state
        reconciliation — is identical by construction (the two paths used to
        be parallel copies and had drifted).
        """
        self._stop_event.clear()
        self._cancel_requested = False
        self._messages_written = 0
        self._topic_msg_counts = {}

        max_duration = self._default_max_duration
        bag_dir = self._create_bag_dir()
        result = RecordEpisode.Result()
        result.bag_path = str(bag_dir)

        source = "action" if goal_handle is not None else "service"
        self.get_logger().info(f"Recording ({source}): {bag_dir}, max={max_duration}s")

        # The outer finally reconciles the recorder's state flags on EVERY
        # exit — including a goal-handle transition raising — so no path can
        # leave _busy held and brick all subsequent recordings.
        try:
            start_time = time.time()
            try:
                # Open writer and register topics BEFORE setting _is_recording:
                # _open_writer flushes buffered TRANSIENT_LOCAL messages, and
                # the live-write path must never see the flag true while the
                # writer is still None (it would silently drop messages).
                self._open_writer(bag_dir)
                self._is_recording = True

                feedback = RecordEpisode.Feedback() if goal_handle is not None else None

                while not self._stop_event.is_set():
                    elapsed = time.time() - start_time

                    # Timeout (max_duration <= 0 means record until stopped)
                    if max_duration > 0:
                        remaining = max(0, max_duration - elapsed)
                        if remaining <= 0:
                            self.get_logger().info("Timeout reached")
                            break
                    else:
                        remaining = 0

                    if goal_handle is not None:
                        if goal_handle.is_cancel_requested:
                            self._stop_event.set()
                            break
                        # Read message count under lock for thread safety.
                        with self._writer_lock:
                            msg_count = self._messages_written
                        feedback.seconds_remaining = int(remaining)
                        feedback.messages_written = msg_count
                        feedback.status = "recording"
                        goal_handle.publish_feedback(feedback)

                    time.sleep(1.0 / self._feedback_rate_hz)

            except Exception as e:
                self.get_logger().error(f"Recording error: {e}")
                result.success = False
                result.message = str(e)
                self._close_writer()
                if goal_handle is not None:
                    try:
                        goal_handle.abort()
                    except Exception as abort_error:
                        self.get_logger().warning(f"Failed to abort goal handle: {abort_error}")
                return result

            # Capture the discovered-topic list BEFORE closing the writer:
            # _close_writer clears it, which used to silently drop the
            # record_all topic counts from the summary.
            discovered_topics = list(self._discovered_topics)
            self._close_writer()
            try:
                self._write_metadata(bag_dir, prompt, self._contract_text, self._contract_hash)
            except RuntimeError as e:
                # Metadata write failed - this is a real error, fail the run
                self.get_logger().error(f"Metadata error: {e}")
                result.success = False
                result.message = f"Recording completed but metadata failed: {e}"
                result.messages_written = self._messages_written
                if goal_handle is not None:
                    goal_handle.abort()
                return result

            self._last_bag_dir = bag_dir
            result.messages_written = self._messages_written
            self._log_topic_summary(bag_dir, discovered_topics, time.time() - start_time)

            # Set terminal state. The record thread (the goal-handle owner) is
            # the only place that transitions the handle; only rcl-legal
            # transitions are used (a service cancel finishes as ABORTED —
            # CANCELED requires the CANCELING state, which only an
            # action-protocol cancel enters).
            result.success = True
            if goal_handle is not None:
                finish_goal(
                    goal_handle,
                    result,
                    service_cancelled=self._cancel_requested,
                    success_message=f"Recorded {self._messages_written} messages",
                )
            return result
        finally:
            self._is_recording = False
            self._busy.release()

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
        """Open writer and register all topics."""
        # Step 0: Discover non-contract topics on the graph
        if self.get_parameter("record_all").value:
            self._discovered_topics = self._discover_topics()
            for topic, type_str, qos in self._discovered_topics:
                if topic not in self._subs:
                    sub = self._create_sub(topic, type_str, qos)
                    self._subs[topic] = sub

        # Step 3: Open storage and create writer
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

        # Combine contract topics with any auto-discovered topics
        all_topics = self._topics + self._discovered_topics

        # QoS conversion: Jazzy uses rosbag2_py._storage.QoS objects, Humble uses YAML strings
        if _IS_JAZZY:
            # Jazzy/Rolling: Use rosbag2_py._storage.QoS API
            def _qos_to_rosbag2(q: QoSProfile | int) -> rosbag2_py._storage.QoS:
                """Convert an rclpy QoSProfile (or int depth) to a rosbag2_py QoS."""
                from rosbag2_py._storage import (
                    Duration as Rosbag2Duration,
                )
                from rosbag2_py._storage import (
                    QoS as Rosbag2QoS,
                )
                from rosbag2_py._storage import (
                    rmw_qos_durability_policy_t,
                    rmw_qos_history_policy_t,
                    rmw_qos_liveliness_policy_t,
                    rmw_qos_reliability_policy_t,
                )

                # Extract numeric RMW values using unified helper
                vals = extract_qos_numeric_values(q)

                if isinstance(q, int):
                    return Rosbag2QoS(q).reliable()

                # Build rosbag2 QoS using the rosbag2_py enum types (not raw ints).
                # The QoS setter methods on Jazzy require rmw_qos_*_policy_t enums.
                bag_qos = Rosbag2QoS(vals["depth"])
                bag_qos = bag_qos.history(rmw_qos_history_policy_t(vals["history"]))
                bag_qos = bag_qos.reliability(rmw_qos_reliability_policy_t(vals["reliability"]))
                bag_qos = bag_qos.durability(rmw_qos_durability_policy_t(vals["durability"]))
                bag_qos = bag_qos.liveliness(rmw_qos_liveliness_policy_t(vals["liveliness"]))

                # Convert rclpy Duration to rosbag2 Duration.
                # rosbag2_py._storage.Duration takes int32 seconds / uint32 nanoseconds.
                # "Infinite"/unset deadlines discovered from real publishers come back
                # as RMW_DURATION_INFINITE (int64 nanoseconds max), which overflows
                # those bounds, so clamp to the same MAX_SEC/MAX_NSEC sentinel the
                # Humble YAML path below uses for "infinite".
                MAX_SEC = 2147483647
                MAX_NSEC = 4294967295

                def _dur(rclpy_dur) -> Rosbag2Duration:
                    ns = int(getattr(rclpy_dur, "nanoseconds", 0) or 0)
                    seconds = min(ns // 1_000_000_000, MAX_SEC)
                    nanoseconds = min(ns % 1_000_000_000, MAX_NSEC)
                    return Rosbag2Duration(seconds, nanoseconds)

                bag_qos = bag_qos.deadline(_dur(q.deadline))
                bag_qos = bag_qos.lifespan(_dur(q.lifespan))
                return bag_qos.liveliness_lease_duration(_dur(q.liveliness_lease_duration))

            # Register topics with Jazzy API
            for idx, (topic, type_str, qos) in enumerate(all_topics):
                topic_info = rosbag2_py.TopicMetadata(
                    id=idx,
                    name=topic,
                    type=type_str,
                    serialization_format="cdr",
                    offered_qos_profiles=[_qos_to_rosbag2(qos)],
                )
                writer.create_topic(topic_info)

        else:
            # Humble: Use YAML string format for offered_qos_profiles
            def _serialize_offered_qos(q: QoSProfile | int) -> str:
                """
                Emit a Humble-compatible YAML mapping string for rosbag2 metadata.

                Uses rclpy.qos enum values consistently with Jazzy approach.
                Output format: YAML list with single QoS mapping (prefixed with '- ').
                """
                # Numeric defaults for deadline/lifespan/liveliness_lease_duration
                # These represent "infinite" duration in RMW
                MAX_SEC = 2147483647
                MAX_NSEC = 4294967295

                # Extract all QoS numeric values using unified helper
                vals = extract_qos_numeric_values(q)

                # Build the Humble-style YAML string
                # rosbag2_player requires all fields present
                lines = [
                    f"- history: {vals['history']}",
                    f"  depth: {vals['depth']}",
                    f"  reliability: {vals['reliability']}",
                    f"  durability: {vals['durability']}",
                    "  deadline:",
                    f"    sec: {MAX_SEC}",
                    f"    nsec: {MAX_NSEC}",
                    "  lifespan:",
                    f"    sec: {MAX_SEC}",
                    f"    nsec: {MAX_NSEC}",
                    f"  liveliness: {vals['liveliness']}",
                    "  liveliness_lease_duration:",
                    f"    sec: {MAX_SEC}",
                    f"    nsec: {MAX_NSEC}",
                    "  avoid_ros_namespace_conventions: false",
                ]
                return "\n".join(lines)

            # Register topics with Humble API (YAML string format)
            for _idx, (topic, type_str, qos) in enumerate(all_topics):
                offered = _serialize_offered_qos(qos)
                topic_info = rosbag2_py.TopicMetadata(topic, type_str, "cdr", offered)
                writer.create_topic(topic_info)

        # Publish the writer atomically and flush buffered TRANSIENT_LOCAL messages
        with self._writer_lock:
            self._writer = writer

            # Flush buffered messages at bag start.
            # TRANSIENT_LOCAL messages (like /tf_static) are written at t=0
            # so they're available immediately when the bag is played back.
            # All buffered messages get the same timestamp because they're latched -
            # the bag player will re-publish them as TRANSIENT_LOCAL regardless.
            bag_start_ns = self.get_clock().now().nanoseconds

            with self._buffer_lock:
                for topic, buffer in self._buffers.items():
                    for serialized, _receive_ns in buffer:
                        # Write at bag start. The header.stamp inside the serialized
                        # message is preserved (often 0 for static TFs).
                        writer.write(topic, serialized, bag_start_ns)
                        self._messages_written += 1

                    if buffer:
                        self.get_logger().info(f"Flushed {len(buffer)} buffered messages for {topic}")

    def _close_writer(self) -> None:
        """Close the writer and finalize the bag file."""
        with self._writer_lock:
            if self._writer is not None:
                self._writer.close()  # Explicitly close to finalize MCAP indices
            self._writer = None
        self._cleanup_discovered_subs()

    def _write_metadata(self, bag_dir: Path, prompt: str, contract_text: str = "", contract_hash: str = "") -> None:
        """
        Write prompt and/or contract provenance to metadata.yaml as custom_data.

        Raises
        ------
            RuntimeError: If metadata.yaml cannot be written after retries.
                This is a fail-fast design - we don't silently lose the prompt
                or the contract.

        """
        if not prompt and not contract_text:
            return

        meta_path = bag_dir / "metadata.yaml"
        last_error: Exception | None = None

        for attempt in range(METADATA_RETRY_COUNT):
            try:
                if not meta_path.exists():
                    time.sleep(METADATA_RETRY_DELAY_SEC)
                    continue

                with meta_path.open("r") as f:
                    meta = yaml.safe_load(f) or {}

                # Handle case where values exist but are None
                info = meta.get(BAG_METADATA_KEY) or {}
                meta[BAG_METADATA_KEY] = info
                custom = info.get(BAG_CUSTOM_DATA_KEY) or {}
                info[BAG_CUSTOM_DATA_KEY] = custom
                if prompt:
                    custom[BAG_PROMPT_KEY] = prompt
                if contract_text:
                    custom[BAG_CONTRACT_KEY] = contract_text
                    custom[BAG_CONTRACT_HASH_KEY] = contract_hash

                with meta_path.open("w") as f:
                    yaml.safe_dump(meta, f, sort_keys=False)

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
    rclpy.init(args=args)
    node = EpisodeRecorderNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Lifecycle callbacks handle cleanup; just destroy and shutdown
        node.destroy_node()
        rclpy.try_shutdown()

    return 0


if __name__ == "__main__":
    main()
