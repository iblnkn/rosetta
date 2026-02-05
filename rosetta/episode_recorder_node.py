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

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.qos import QoSProfile
from rclpy.serialization import serialize_message

import rosbag2_py
import yaml
from rcl_interfaces.msg import ParameterDescriptor
from rosidl_runtime_py.utilities import get_message
from rosetta_interfaces.action import RecordEpisode

# Import contract utilities
from .common.contract import load_contract
from .common import decoders as _decoders  # noqa: F401 - registers decoders
from .common import encoders as _encoders  # noqa: F401 - registers encoders
from .common.contract_utils import iter_specs
from .common.ros2_utils import qos_profile_from_dict

# Bag metadata keys
BAG_METADATA_KEY = "rosbag2_bagfile_information"
BAG_CUSTOM_DATA_KEY = "custom_data"
BAG_PROMPT_KEY = "lerobot.operator_prompt"

# ---------- Constants ----------

# Metadata file retry settings (internal implementation detail)
METADATA_RETRY_COUNT = 10
METADATA_RETRY_DELAY_SEC = 0.1


class EpisodeRecorderNode(LifecycleNode):
    """
    Stream-to-bag episode recorder with lifecycle and action interface.

    Follows rosbag2_py tutorial patterns:
    - SequentialWriter with StorageOptions/ConverterOptions
    - TopicMetadata for topic registration
    - serialize_message() for writing
    """

    def __init__(self):
        super().__init__("episode_recorder", enable_logger_service=True)

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
        self._topics: list[tuple[str, str, QoSProfile | int]] = []
        self._subs: list = []
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
        for topic, type_str, qos in self._topics:
            self._subs.append(self._create_sub(topic, type_str, qos))

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
        for sub in self._subs:
            self.destroy_subscription(sub)
        self._subs = []

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
        for sub in self._subs:
            self.destroy_subscription(sub)
        self._subs = []

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

    def _build_topic_list(self) -> list[tuple[str, str, QoSProfile | int]]:
        """Extract topics from contract.

        Includes:
        - Observation and action topics (from iter_specs)
        - Task topics
        - Extra topics (recording.extra_topics) - recorded but not mapped to keys
        """
        topics: list[tuple[str, str, QoSProfile | int]] = []

        for spec in iter_specs(self._contract):
            qos = qos_profile_from_dict(spec.qos) or 10
            topics.append((spec.topic, spec.msg_type, qos))

        # Task topics
        for task in self._contract.tasks or []:
            qos = qos_profile_from_dict(task.qos) or 10
            topics.append((task.topic, task.type, qos))

        # Adjunct topics (record-only, no key mapping)
        for adj in self._contract.adjunct:
            qos = qos_profile_from_dict(adj.qos) or 10
            topics.append((adj.topic, adj.type, qos))

        return topics

    def _create_sub(self, topic: str, type_str: str, qos: QoSProfile | int):
        """Create subscription that writes to bag when recording."""
        msg_cls = get_message(type_str)

        def callback(msg: Any, _topic: str = topic) -> None:
            if not self._is_recording:
                return

            with self._writer_lock:
                if self._writer is None:
                    return
                try:
                    timestamp_ns = self.get_clock().now().nanoseconds
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

    def _execute(self, goal_handle) -> RecordEpisode.Result:
        """Execute recording episode."""
        self._goal_handle = goal_handle
        self._is_recording = True
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
            # Open writer and register topics
            self._open_writer(bag_dir)

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

        # Register all topics
        for idx, (topic, type_str, _) in enumerate(self._topics):
            topic_info = rosbag2_py.TopicMetadata(
                id=idx,
                name=topic,
                type=type_str,
                serialization_format="cdr",
            )
            writer.create_topic(topic_info)

        with self._writer_lock:
            self._writer = writer

    def _close_writer(self) -> None:
        """Close the writer."""
        with self._writer_lock:
            self._writer = None  # SequentialWriter closes on del

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