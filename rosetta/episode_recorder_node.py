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

    ros2 action send_goal /episode_recorder/record_episode \
        rosetta_interfaces/action/RecordEpisode "{prompt: 'pick up cube'}" --feedback
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any
from collections import deque
from typing import Optional

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.qos import (
    QoSProfile,
    HistoryPolicy,
    ReliabilityPolicy,
    DurabilityPolicy,
    LivelinessPolicy,
)
from rclpy.serialization import serialize_message

import rosbag2_py
import yaml
from rcl_interfaces.msg import ParameterDescriptor
from std_srvs.srv import Trigger
from rosidl_runtime_py.utilities import get_message
from rosetta_interfaces.action import RecordEpisode
from rosetta_interfaces.srv import StartRecording

# Import contract utilities
from .common.contract import load_contract
from .common import decoders as _decoders  # noqa: F401 - registers decoders
from .common import encoders as _encoders  # noqa: F401 - registers encoders
from .common.contract_utils import iter_specs
from .common.ros2_utils import (
    qos_profile_from_dict,
    detect_ros_distro,
    is_jazzy_or_newer,
    extract_qos_numeric_values,
    is_transient_local,
    get_qos_depth,
)

# Bag metadata keys
BAG_METADATA_KEY = "rosbag2_bagfile_information"
BAG_CUSTOM_DATA_KEY = "custom_data"
BAG_PROMPT_KEY = "lerobot.operator_prompt"

# ---------- Constants ----------

# Metadata file retry settings (internal implementation detail)
METADATA_RETRY_COUNT = 10
METADATA_RETRY_DELAY_SEC = 0.1
# Maximum serialized bytes to buffer for a retained message (4 MiB)
MAX_BUFFER_BYTES = 4 * 1024 * 1024

# ROS2 distribution compatibility flag
# Uses common utilities from ros2_compat module
_IS_JAZZY = is_jazzy_or_newer()


class EpisodeRecorderNode(LifecycleNode):
    """
    Stream-to-bag episode recorder with lifecycle and action interface.

    Follows rosbag2_py tutorial patterns:
    - SequentialWriter with StorageOptions/ConverterOptions
    - TopicMetadata for topic registration
    - serialize_message() for writing
    """

    def __init__(self):
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
            "contract_path", "",
            ParameterDescriptor(description="Path to contract YAML file", read_only=True)
        )
        self.declare_parameter(
            "bag_base_dir", "/workspaces/rosetta_ws/datasets/bags",
            ParameterDescriptor(description="Base directory for bag storage", read_only=True)
        )
        self.declare_parameter(
            "storage_id", "mcap",
            ParameterDescriptor(description="Bag storage format (mcap, sqlite3)", read_only=True)
        )
        self.declare_parameter(
            "default_max_duration", 300.0,
            ParameterDescriptor(description="Maximum recording duration in seconds")
        )
        self.declare_parameter(
            "feedback_rate_hz", 2.0,
            ParameterDescriptor(description="Rate for publishing action feedback")
        )

        # Initialize state variables (resources created in lifecycle callbacks)
        self._contract = None
        self._bag_base: Path | None = None
        self._storage_id: str | None = None
        self._default_max_duration: float = 300.0
        self._feedback_rate_hz: float = 2.0
        self._topics: list[tuple[str, str, QoSProfile | int, str]] = []  # (topic, type, qos, buffering_strategy)
        self._subs: dict[str, Any] = {}  # topic -> subscription object
        self._action_server: ActionServer | None = None
        self._accepting_goals = False

        # Recording state
        self._writer: rosbag2_py.SequentialWriter | None = None
        self._writer_lock = threading.Lock()
        self._is_recording = False
        self._messages_written = 0
        self._stop_event = threading.Event()
        self._goal_handle = None
        self._cbg = ReentrantCallbackGroup()
        # Buffers for TRANSIENT_LOCAL messages (like /tf_static)
        # Each buffer is a deque limited by QoS history depth
        self._buffers: dict[str, deque] = {}
        self._buffer_lock = threading.Lock()

        self.get_logger().info("Node created (unconfigured)")

    # -------------------- Lifecycle callbacks --------------------

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Load contract, create subscriptions and action server."""
        contract_path = self.get_parameter("contract_path").value
        if not contract_path:
            self.get_logger().error("contract_path parameter required")
            return TransitionCallbackReturn.FAILURE

        try:
            self._contract = load_contract(contract_path)
        except Exception as e:
            self.get_logger().error(f"Failed to load contract: {e}")
            return TransitionCallbackReturn.FAILURE

        self._bag_base = Path(self.get_parameter("bag_base_dir").value)
        self._bag_base.mkdir(parents=True, exist_ok=True)
        self._storage_id = self.get_parameter("storage_id").value
        self._default_max_duration = self.get_parameter("default_max_duration").value
        self._feedback_rate_hz = self.get_parameter("feedback_rate_hz").value

        # Build topic list from contract
        self._topics = self._build_topic_list()

        # Create subscriptions (callbacks no-op when not recording)
        for topic, type_str, qos, buffering_strategy in self._topics:
            sub = self._create_sub(topic, type_str, qos, buffering_strategy)
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

        self.get_logger().info(
            f"Configured: robot_type={self._contract.robot_type}, topics={len(self._topics)}"
        )
        return TransitionCallbackReturn.SUCCESS

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
            timeout = 5.0
            start = time.time()
            while self._is_recording and (time.time() - start) < timeout:
                time.sleep(0.1)

            if self._is_recording:
                self.get_logger().warning("Recording did not stop within timeout")

        self.get_logger().info("Deactivated")
        return super().on_deactivate(state)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Release resources."""
        # Destroy subscriptions
        for sub in self._subs.values():
            self.destroy_subscription(sub)
        self._subs.clear()

        # Destroy action server
        if self._action_server is not None:
            self.destroy_action_server(self._action_server)
            self._action_server = None

        # Clear state
        self._contract = None
        self._topics = []

        self.get_logger().info("Cleaned up")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Final cleanup before destruction."""
        self._accepting_goals = False
        self._stop_event.set()
        self._close_writer()

        # Destroy subscriptions
        for sub in self._subs.values():
            self.destroy_subscription(sub)
        self._subs.clear()

        # Destroy action server
        if self._action_server is not None:
            self.destroy_action_server(self._action_server)
            self._action_server = None

        self.get_logger().info("Shutdown complete")
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Handle errors by cleaning up resources."""
        self.get_logger().error(f"Error occurred in state: {state.label}")

        try:
            self._accepting_goals = False
            self._stop_event.set()
            self._close_writer()
        except Exception as e:
            self.get_logger().error(f"Error during cleanup: {e}")

        return TransitionCallbackReturn.SUCCESS

    # -------------------- Topic and subscription management --------------------

    def _build_topic_list(self) -> list[tuple[str, str, QoSProfile | int, str]]:
        """Extract topics from contract.

        Includes:
        - Observation and action topics (from iter_specs)
        - Task topics
        - Extra topics (recording.extra_topics) - recorded but not mapped to keys
        
        Returns:
            List of (topic, type_str, qos, buffering_strategy) tuples
        """
        topics: list[tuple[str, str, QoSProfile | int, str]] = []

        for spec in iter_specs(self._contract):
            qos = qos_profile_from_dict(spec.qos) or 10
            # Default buffering strategy for observation/action topics
            topics.append((spec.topic, spec.msg_type, qos, "no_buffer"))

        # Task topics
        for task in self._contract.tasks or []:
            qos = qos_profile_from_dict(task.qos) or 10
            topics.append((task.topic, task.type, qos, "no_buffer"))

        # Adjunct topics (record-only, no key mapping) - these can have buffering strategies
        for adj in self._contract.adjunct or []:
            qos = qos_profile_from_dict(adj.qos) or 10
            # Use buffering strategy from contract, default to "accumulate" for transient_local
            buffering_strategy = adj.buffering_strategy
            if buffering_strategy is None:
                # Auto-detect: use accumulate for transient_local, no_buffer otherwise
                if is_transient_local(qos):
                    buffering_strategy = "accumulate"
                else:
                    buffering_strategy = "no_buffer"
            
            topics.append((adj.topic, adj.type, qos, buffering_strategy))

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
            topics.append(("/clock", "rosgraph_msgs/msg/Clock", 10, "no_buffer"))

        return topics

    def _create_sub(self, topic: str, type_str: str, qos: QoSProfile | int, buffering_strategy: str):
        """Create subscription that writes to bag when recording."""
        msg_cls = get_message(type_str)
        
        # Helper to extract header timestamp (used for buffering and deduplication)
        def get_header_stamp_ns(msg: Any) -> Optional[int]:
            """Extract header.stamp as nanoseconds, or None if not present."""
            try:
                # Try msg.header first (most common)
                hdr = getattr(msg, "header", None)
                if hdr is not None and hasattr(hdr, "stamp"):
                    ts = hdr.stamp
                    return int(ts.sec) * 1_000_000_000 + int(getattr(ts, "nanosec", 0))
                # Try msg.transforms[0].header for TFMessage
                if hasattr(msg, "transforms") and len(getattr(msg, "transforms", [])) > 0:
                    fh = getattr(msg.transforms[0], "header", None)
                    if fh is not None and hasattr(fh, "stamp"):
                        ts = fh.stamp
                        return int(ts.sec) * 1_000_000_000 + int(getattr(ts, "nanosec", 0))
            except Exception:
                pass
            return None

        def callback(msg: Any, _topic: str = topic) -> None:
            timestamp_ns = self.get_clock().now().nanoseconds
            # Buffer TRANSIENT_LOCAL messages when not recording (based on strategy)
            if not self._is_recording:
                # Check if this topic should be buffered
                is_tl = is_transient_local(qos)
                history_depth = get_qos_depth(qos)

                # Only buffer if TRANSIENT_LOCAL and strategy is not no_buffer
                if is_tl and buffering_strategy != "no_buffer":
                    try:
                        serialized = serialize_message(msg)
                        if len(serialized) <= MAX_BUFFER_BYTES:
                            header_stamp = get_header_stamp_ns(msg)
                            
                            with self._buffer_lock:
                                if _topic not in self._buffers:
                                    self._buffers[_topic] = deque()
                                self._buffers[_topic].append((serialized, timestamp_ns, header_stamp))
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
                except Exception as e:
                    self.get_logger().error(f"Write failed on {_topic}: {e}")
                    self._stop_event.set()

        return self.create_subscription(msg_cls, topic, callback, qos, callback_group=self._cbg)

    # ---------- Action callbacks ----------

    def _on_goal(self, goal_request) -> GoalResponse:
        """Accept if active and not already recording."""
        self.get_logger().info("Received goal request")
        if not self._accepting_goals:
            self.get_logger().warning("Rejected: node not active")
            return GoalResponse.REJECT
        if self._is_recording:
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
        """Handle external Trigger service call to cancel recording.

        Sets the internal stop event and attempts to transition the active
        goal to the canceled state. Returns a Trigger response indicating
        whether a recording was active when the call arrived.
        """
        if not self._is_recording:
            response.success = False
            response.message = "No active recording"
            return response

        self.get_logger().info("cancel_recording service called: stopping recording")
        # Signal the recording loop to stop
        self._stop_event.set()

        # Try to move the action goal to canceled if present
        if self._goal_handle is not None:
            try:
                # If the executor/loop is inside _execute, calling canceled()
                # here will transition the goal state. The execute loop also
                # checks is_cancel_requested and will perform its own cleanup.
                self._goal_handle.canceled()
            except Exception as e:
                self.get_logger().debug(f"Failed to cancel goal handle: {e}")

        response.success = True
        response.message = "Cancel requested"
        return response

    def _on_start_service(self, request, response):
        """Handle StartRecording service call.

        Starts recording directly without the ROS2 action protocol.
        This is the primary interface for Foxglove extensions since the
        foxglove bridge cannot route to hidden _action/* services.
        """
        if not self._accepting_goals:
            response.accepted = False
            response.message = "Node not active"
            return response

        if self._is_recording:
            response.accepted = False
            response.message = "Already recording"
            return response

        prompt = request.prompt or ""
        self.get_logger().info(f"start_recording service called: prompt='{prompt}'")

        # Start recording in a background thread (mirrors _execute logic)
        self._stop_event.clear()
        self._messages_written = 0
        self._goal_handle = None  # No action goal for service-based recording
        self._is_recording = True  # Set recording flag immediately to start writing live messages
        
        record_thread = threading.Thread(
            target=self._service_record,
            args=(prompt,),
            daemon=True,
        )
        record_thread.start()

        response.accepted = True
        response.message = "Recording started"
        return response

    def _service_record(self, prompt: str) -> None:
        """Recording loop for service-based starts (no action goal handle)."""
        bag_dir = self._create_bag_dir()
        max_duration = self._default_max_duration

        self.get_logger().info(f"Recording (service): {bag_dir}, max={max_duration}s")

        try:
            self._open_writer(bag_dir)
            self._is_recording = True

            start_time = time.time()
            while not self._stop_event.is_set():
                elapsed = time.time() - start_time
                if elapsed >= max_duration:
                    self.get_logger().info("Timeout reached")
                    break
                time.sleep(1.0 / self._feedback_rate_hz)

        except Exception as e:
            self.get_logger().error(f"Recording error: {e}")

        # Finalize
        self._close_writer()
        try:
            self._write_metadata(bag_dir, prompt)
        except RuntimeError as e:
            self.get_logger().error(f"Metadata error: {e}")

        self.get_logger().info(f"Recorded {self._messages_written} messages to {bag_dir}")
        self._is_recording = False

    def _execute(self, goal_handle) -> RecordEpisode.Result:
        """Execute recording episode."""
        self._goal_handle = goal_handle
        self._stop_event.clear()  # Reset for new recording
        self._messages_written = 0

        prompt = goal_handle.request.prompt or ""
        max_duration = self._default_max_duration

        # Create unique bag directory
        bag_dir = self._create_bag_dir()
        result = RecordEpisode.Result()
        result.bag_path = str(bag_dir)

        self.get_logger().info(f"Recording: {bag_dir}, max={max_duration}s")

        try:
            # Open writer and register topics BEFORE setting _is_recording
            # This allows _open_writer to flush buffered TRANSIENT_LOCAL messages
            self._open_writer(bag_dir)
            
            # NOW set recording flag so live messages start being written
            self._is_recording = True

            # Recording loop with feedback
            start_time = time.time()
            feedback = RecordEpisode.Feedback()

            while not self._stop_event.is_set():
                elapsed = time.time() - start_time
                remaining = max(0, max_duration - elapsed)

                # Check timeout
                if remaining <= 0:
                    self.get_logger().info("Timeout reached")
                    break

                # Check cancel
                if goal_handle.is_cancel_requested:
                    self._stop_event.set()
                    break

                # Publish feedback (read message count under lock for thread safety)
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
            self._cleanup(goal_handle, aborted=True)
            return result

        # Finalize - close writer and write metadata
        self._close_writer()
        try:
            self._write_metadata(bag_dir, prompt)
        except RuntimeError as e:
            # Metadata write failed - this is a real error, fail the action
            self.get_logger().error(f"Metadata error: {e}")
            result.success = False
            result.message = f"Recording completed but metadata failed: {e}"
            result.messages_written = self._messages_written
            goal_handle.abort()
            self._is_recording = False
            self._goal_handle = None
            return result

        result.messages_written = self._messages_written
        self.get_logger().info(f"Recorded {self._messages_written} messages to {bag_dir}")

        # Set terminal state
        if goal_handle.is_cancel_requested:
            result.success = False
            result.message = "Cancelled"
            goal_handle.canceled()
        else:
            result.success = True
            result.message = f"Recorded {self._messages_written} messages"
            goal_handle.succeed()

        self._is_recording = False
        self._goal_handle = None
        return result

    def _cleanup(self, goal_handle, aborted: bool = False):
        """Clean up after error or abort."""
        self._close_writer()
        self._is_recording = False
        self._goal_handle = None
        if aborted:
            try:
                goal_handle.abort()
            except Exception as e:
                self.get_logger().warning(f"Failed to abort goal handle: {e}")

    # ---------- rosbag2 helpers ----------

    def _create_bag_dir(self) -> Path:
        """Generate unique bag directory name."""
        t_ns = time.time_ns()
        sec, nsec = divmod(t_ns, 1_000_000_000)
        bag_dir = self._bag_base / f"{sec:010d}_{nsec:09d}"
        return bag_dir

    def _open_writer(self, bag_dir: Path) -> None:
        """Open writer and register all topics."""
        
        # Step 1: Re-subscribe to topics with resubscribe_on_start strategy
        # This must happen BEFORE opening the writer so messages buffer properly
        resubscribe_topics = [
            (topic, type_str, qos, strategy)
            for topic, type_str, qos, strategy in self._topics
            if strategy == "resubscribe_on_start"
        ]
        
        for topic, type_str, qos, strategy in resubscribe_topics:
            self.get_logger().info(f"Re-subscribing to {topic} for fresh TRANSIENT_LOCAL data...")
            
            # Find and destroy old subscriber
            if topic in self._subs:
                old_sub = self._subs[topic]
                self.destroy_subscription(old_sub)
                del self._subs[topic]
            
            # Clear buffer for this topic
            with self._buffer_lock:
                if topic in self._buffers:
                    old_count = len(self._buffers[topic])
                    self._buffers[topic].clear()
                    self.get_logger().info(f"Cleared {old_count} stale messages from {topic} buffer")
            
            # Create new subscriber - will immediately receive latched TRANSIENT_LOCAL messages
            new_sub = self._create_sub(topic, type_str, qos, strategy)
            self._subs[topic] = new_sub
        
        # Step 2: Brief sleep to allow TRANSIENT_LOCAL delivery to new subscribers
        if resubscribe_topics:
            time.sleep(0.2)  # 200ms should be plenty for latched message delivery
            with self._buffer_lock:
                for topic, _, _, _ in resubscribe_topics:
                    buf_size = len(self._buffers.get(topic, []))
                    if buf_size == 0:
                        self.get_logger().warning(
                            f"After re-subscribe, {topic} buffer is EMPTY! "
                            f"This means the publisher is not publishing with TRANSIENT_LOCAL QoS, "
                            f"or the publisher is not running. Recording will continue without this data."
                        )
                    else:
                        self.get_logger().info(
                            f"After re-subscribe, {topic} buffer has {buf_size} message(s)"
                        )
        
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

        # QoS conversion: Jazzy uses rosbag2_py._storage.QoS objects, Humble uses YAML strings
        if _IS_JAZZY:
            # Jazzy/Rolling: Use rosbag2_py._storage.QoS API
            def _qos_to_rosbag2(q: QoSProfile | int) -> rosbag2_py._storage.QoS:
                """Convert an rclpy QoSProfile (or int depth) to a rosbag2_py QoS."""
                from rosbag2_py._storage import (
                    QoS as Rosbag2QoS,
                    Duration as Rosbag2Duration,
                )
                
                # Extract numeric RMW values using unified helper
                vals = extract_qos_numeric_values(q)
                
                if isinstance(q, int):
                    return Rosbag2QoS(q).reliable()
                
                # Build rosbag2 QoS using numeric enum values
                bag_qos = Rosbag2QoS(vals["depth"])
                bag_qos = bag_qos.history(vals["history"])
                bag_qos = bag_qos.reliability(vals["reliability"])
                bag_qos = bag_qos.durability(vals["durability"])
                bag_qos = bag_qos.liveliness(vals["liveliness"])

                # Convert rclpy Duration to rosbag2 Duration
                def _dur(rclpy_dur) -> Rosbag2Duration:
                    ns = int(getattr(rclpy_dur, "nanoseconds", 0) or 0)
                    return Rosbag2Duration(ns // 1_000_000_000, ns % 1_000_000_000)

                bag_qos = bag_qos.deadline(_dur(q.deadline))
                bag_qos = bag_qos.lifespan(_dur(q.lifespan))
                bag_qos = bag_qos.liveliness_lease_duration(_dur(q.liveliness_lease_duration))

                return bag_qos
            
            # Register topics with Jazzy API
            for idx, (topic, type_str, qos, strategy) in enumerate(self._topics):
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
                    f"  deadline:",
                    f"    sec: {MAX_SEC}",
                    f"    nsec: {MAX_NSEC}",
                    f"  lifespan:",
                    f"    sec: {MAX_SEC}",
                    f"    nsec: {MAX_NSEC}",
                    f"  liveliness: {vals['liveliness']}",
                    f"  liveliness_lease_duration:",
                    f"    sec: {MAX_SEC}",
                    f"    nsec: {MAX_NSEC}",
                    f"  avoid_ros_namespace_conventions: false",
                ]
                return "\n".join(lines)
            
            # Register topics with Humble API (YAML string format)
            for idx, (topic, type_str, qos, strategy) in enumerate(self._topics):
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
                    for serialized, _, _ in buffer:
                        # Write at bag start. The header.stamp inside the serialized
                        # message is preserved (often 0 for static TFs).
                        writer.write(topic, serialized, bag_start_ns)
                        self._messages_written += 1
                    
                    if buffer:
                        self.get_logger().info(
                            f"Flushed {len(buffer)} buffered messages for {topic}"
                        )

    def _close_writer(self) -> None:
        """Close the writer and finalize the bag file."""
        with self._writer_lock:
            if self._writer is not None:
                self._writer.close()  # Explicitly close to finalize MCAP indices
            self._writer = None

    def _write_metadata(self, bag_dir: Path, prompt: str) -> None:
        """
        Write prompt to metadata.yaml as custom_data.

        Raises:
            RuntimeError: If metadata.yaml cannot be written after retries.
                This is a fail-fast design - we don't silently lose the prompt.
        """
        if not prompt:
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
                custom[BAG_PROMPT_KEY] = prompt

                with meta_path.open("w") as f:
                    yaml.safe_dump(meta, f, sort_keys=False)

                self.get_logger().debug(f"Wrote prompt to metadata on attempt {attempt + 1}")
                return
            except Exception as e:
                last_error = e
                self.get_logger().debug(f"Metadata write attempt {attempt + 1} failed: {e}")
                time.sleep(METADATA_RETRY_DELAY_SEC)

        # Fail fast - don't silently lose the prompt
        raise RuntimeError(
            f"Failed to write prompt to {meta_path} after {METADATA_RETRY_COUNT} attempts. "
            f"Last error: {last_error}"
        )


def main(args=None):
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