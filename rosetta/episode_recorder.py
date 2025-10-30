#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Episode Recorder (ROS 2): stream-to-bag writer with action control.

Overview
--------
`EpisodeRecorderServer` writes incoming ROS 2 messages directly to a rosbag2 
as they arrive (no alignment or caching). The set of topics,
their types, QoS, and runtime parameters come from a "contract" file.

The node exposes a `record_episode` Action (from
`rosetta_interfaces/action/RecordEpisode.action`) so clients can start and stop
recordings programmatically. A lightweight `record_episode/cancel` service is
provided as a cancel path.

Key behavior
------------
- Subscriptions are created once, at node startup, based on the contract.
- When a recording starts, a writer is opened in a unique directory
  (`bag_base_dir/<sec>_<nsec>`), all topics are registered, and every received
  message is written to the bag with a timestamp (receive time or header time).
- Two timers are created per episode:
  * a periodic feedback timer (2 Hz) for action feedback
  * a one-shot timeout timer (from `contract.max_duration_s`)
- When the episode stops (cancel/timeout/error), the node closes the writer,
  attempts to amend the bag's `metadata.yaml` with the user's prompt, and
  tears down the per-episode timers.

Parameters
----------
contract_path : str (required)
    Path to the YAML contract that specifies topics/types/QoS/rate/etc.
bag_base_dir : str, default "/tmp/episodes"
    Directory under which unique per-episode bag directories are created.
storage_preset_profile : str, default ""
    Optional rosbag2 storage preset (e.g., "zstd_fast"). Applied when supported
    by the storage backend (MCAP ignores if not applicable).
storage_config_uri : str, default ""
    Optional file URI or path to a rosbag2 storage config. Applied when
    supported by the backend.

Action
------
Action Name: `record_episode`
Goal fields:
    - prompt (str): free-form operator prompt stored into bag metadata
Feedback:
    - seconds_remaining (int)
    - feedback_message (str): human-readable progress
Result:
    - success (bool), message (str)

Notes
-----
- Write failures are treated as fatal for the current episode,
  the error is logged with a traceback, and the episode is ended cleanly.
- Subscriptions persist across episodes to avoid churn and DDS re-negotiation.

"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import rclpy
import yaml
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.serialization import serialize_message
from rclpy.time import Time
from rclpy.timer import Timer
from rosidl_runtime_py.utilities import get_message
from std_srvs.srv import Trigger

import rosbag2_py

from rosetta_interfaces.action import RecordEpisode
from rosetta.common.contract_utils import load_contract
from rosetta.common.contract_utils import qos_profile_from_dict


# ------------------------------ Constants ------------------------------

FEEDBACK_PERIOD_S: float = 0.5
METADATA_RETRIES: int = 20
METADATA_RETRY_PERIOD_S: float = 0.1
DEFAULT_QOS_DEPTH: int = 10


# ------------------------------ Dataclasses ----------------------------


@dataclass(slots=True)
class _TopicCounter:
    """Per-topic counters.

    Attributes
    ----------
    seen : int
        Number of messages received on the topic.
    written : int
        Number of messages successfully written to the bag.
    """

    seen: int = 0
    written: int = 0


@dataclass(slots=True)
class Flags:
    """Recorder runtime flags.

    Attributes
    ----------
    is_recording : bool
        Whether an episode is currently being recorded.
    fatal_error : bool
        Whether a fatal writer error occurred during this episode.
    stop_requested : bool
        Whether a stop has been requested (cancel or timeout).
    shutting_down : bool
        Whether the node is shutting down.
    """

    is_recording: bool = False
    fatal_error: bool = False
    stop_requested: bool = False
    shutting_down: bool = False


@dataclass(slots=True)
class WriterState:
    """Shared state for rosbag2 writer access.

    Attributes
    ----------
    writer : Optional[rosbag2_py.SequentialWriter]
        The active writer; `None` when not recording.
    writer_lock : threading.Lock
        Mutex guarding access to the writer.
    counts : dict[str, _TopicCounter]
        Per-topic counters for observability and debugging.
    """

    writer: Optional[rosbag2_py.SequentialWriter] = None
    writer_lock: threading.Lock = field(default_factory=threading.Lock)
    counts: Dict[str, _TopicCounter] = field(default_factory=dict)


# ------------------------------ Node -----------------------------------


class EpisodeRecorderServer(Node):
    """Stream-to-bag episode recorder (no alignment/caching).

    Subscriptions are created once at startup. During an episode, each incoming
    message is serialized and written immediately. The action interface governs
    lifecycle (start/stop) and surfaces progress via periodic feedback.
    """

    def __init__(self) -> None:
        """Construct the recorder and create long-lived subscriptions/action.

        Raises
        ------
        RuntimeError
            If the required `contract_path` parameter is missing.
        """
        super().__init__("recorder_server")

        # Parameters
        self.declare_parameter("contract_path", "")
        self.declare_parameter("bag_base_dir", "/tmp/episodes")
        # Storage tuning (kept optional & conservative by default)
        self.declare_parameter("storage_preset_profile", "")  # e.g., "zstd_fast"
        self.declare_parameter("storage_config_uri", "")  # file:// or path

        contract_path = (
            self.get_parameter("contract_path").get_parameter_value().string_value
        )
        if not contract_path:
            raise RuntimeError(
                "Parameter 'contract_path' is required (path to YAML contract)."
            )
        self._contract = load_contract(contract_path)

        bag_base = self.get_parameter("bag_base_dir").get_parameter_value().string_value
        self._bag_base = Path(bag_base)
        self._bag_base.mkdir(parents=True, exist_ok=True)

        self._storage_preset_profile = (
            self.get_parameter("storage_preset_profile")
            .get_parameter_value()
            .string_value
            or ""
        )
        self._storage_config_uri = (
            self.get_parameter("storage_config_uri").get_parameter_value().string_value
            or ""
        )

        self._flags = Flags()
        self._ws = WriterState()

        # Executor/callback group (reentrant so timers/subs/actions can co-exist)
        self._cbg = ReentrantCallbackGroup()

        # Derive unified topic list (topic, type, qos_dict) from contract sections.
        obs = self._contract.observations or []
        tks = self._contract.tasks or []
        acts = self._contract.actions or []
        self._topics: list[Tuple[str, str, Dict]] = []
        self._topics += [(o.topic, o.type, o.qos or {}) for o in obs]
        self._topics += [(t.topic, t.type, t.qos or {}) for t in tks]
        self._topics += [(a.publish_topic, a.type, a.publish_qos or {}) for a in acts]

        # Subscriptions (created once; callbacks no-op unless recording)
        self._subs: list[Any] = []
        for topic, type_str, qos_dict in self._topics:
            self._ws.counts[topic] = _TopicCounter()
            self._subs.append(self._make_sub(topic, type_str, qos_dict))

        # Action server
        self._current_goal_handle = None
        self._server = ActionServer(
            self,
            RecordEpisode,
            "record_episode",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self._cbg,
        )

        # Cancel service
        self._cancel_service = self.create_service(
            Trigger,
            "record_episode/cancel",
            self._cancel_service_cb,
            callback_group=self._cbg,
        )

        # ROS timers for episode lifecycle/feedback (created per-episode)
        self._timeout_timer: Optional[Timer] = None
        self._feedback_timer: Optional[Timer] = None
        # Used only as a latch (set from timer/cancel/error)
        self._episode_done_evt = threading.Event()

        # Shutdown hook
        self.context.on_shutdown(self._shutdown_cb)

        self.get_logger().info(f"Recorder ready with contract '{self._contract.name}'.")

    # ---------- Action callbacks ----------

    def goal_callback(self, _req: Any) -> GoalResponse:
        """Decide whether to accept a new goal.

        Returns
        -------
        GoalResponse
            ACCEPT if not currently recording; otherwise REJECT.
        """
        if self._flags.is_recording:
            self.get_logger().warning("Rejecting goal: already recording")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle: Any) -> CancelResponse:
        """Handle a cancel request.

        Parameters
        ----------
        goal_handle : Any
            Handle of the goal to cancel.

        Returns
        -------
        CancelResponse
            ACCEPT only for the active goal; otherwise REJECT.
        """
        if not self._flags.is_recording or goal_handle is not self._current_goal_handle:
            self.get_logger().warning("Rejecting cancel: not recording/active")
            return CancelResponse.REJECT
        self.get_logger().info(
            "Action cancel requested - transitioning to CANCELING state"
        )
        # The actual transition to CANCELING happens in execute_callback when is_cancel_requested is checked
        return CancelResponse.ACCEPT

    def _cancel_service_cb(
        self, _req: Trigger.Request, resp: Trigger.Response
    ) -> Trigger.Response:
        """Cancel via service: mirror action cancel semantics (best-effort)."""
        self.get_logger().info("Cancel service called")
        self._flags.stop_requested = True
        self._episode_done_evt.set()
        resp.success = True
        resp.message = "Recording cancelled"
        return resp

    def _shutdown_cb(self) -> None:
        """Abort current goal cleanly on shutdown; do **not** destroy subscriptions."""

        with self._ws.writer_lock:
            self._ws.writer = None

        if self._flags.is_recording and self._current_goal_handle is not None:
            # Set shutting down flag to prevent late writes in sub callbacks
            self._flags.shutting_down = True

            # Now abort the goal
            try:
                self._current_goal_handle.abort()  # TODO: should this be canceled?
            except Exception as exc:  # pragma: no cover (best-effort)
                self.get_logger().warning(
                    f"Failed to abort goal during shutdown: {exc!r}"
                )
        self._flags.is_recording = False
        self._episode_done_evt.set()

    # ---------- rosbag2 helpers ----------

    def _open_writer(
        self, bag_uri: str, storage_id: str
    ) -> rosbag2_py.SequentialWriter:
        """Open a rosbag2 writer with conservative defaults and optional presets.

        Parameters
        ----------
        bag_uri : str
            Destination directory for the bag (will be created).
        storage_id : str
            rosbag2 storage plugin (e.g., "mcap", "sqlite3").

        Returns
        -------
        rosbag2_py.SequentialWriter
            An opened writer ready to register topics and write messages.
        """
        # Base options
        storage_options = rosbag2_py.StorageOptions(uri=bag_uri, storage_id=storage_id)

        # Optional tuning (MCAP supports preset/config; harmless if empty)
        if self._storage_preset_profile:
            if hasattr(storage_options, "storage_preset_profile"):
                setattr(storage_options, "storage_preset_profile", self._storage_preset_profile)
            else:
                self.get_logger().info(
                    "storage_preset_profile ignored on this rosbag2 distro (not supported)"
                )
        if self._storage_config_uri:
            if hasattr(storage_options, "storage_config_uri"):
                setattr(storage_options, "storage_config_uri", self._storage_config_uri)
            else:
                self.get_logger().info(
                    "storage_config_uri ignored on this rosbag2 distro (not supported)"
                )

        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        )
        writer = rosbag2_py.SequentialWriter()
        writer.open(storage_options, converter_options)
        return writer

    def _register_topic(self, topic: str, type_str: str) -> None:
        """Register a topic with the active writer (idempotent per writer)."""
        meta = rosbag2_py.TopicMetadata(
            name=topic, type=type_str, serialization_format="cdr"
        )
        assert self._ws.writer is not None
        self._ws.writer.create_topic(meta)

    def _make_sub(self, topic: str, type_str: str, qos_dict: Dict) -> Any:
        """Create a subscription that writes each message when the writer is open.

        The callback:
        - updates per-topic counters,
        - takes a read-only snapshot of the current writer under a lock,
        - uses arrival time as the write timestamp,
        - serializes and writes the message,
        - signals a fatal error and ends the episode on write exceptions.
        """
        msg_cls = get_message(type_str)
        qos = qos_profile_from_dict(qos_dict) or QoSProfile(depth=DEFAULT_QOS_DEPTH)

        def cb(msg: Any, _topic: str = topic) -> None:
            # Counters
            cnt = self._ws.counts.get(_topic)
            if cnt:
                cnt.seen += 1

            # Writer snapshot (cheap read)
            with self._ws.writer_lock:
                writer = self._ws.writer

            if (
                not self._flags.is_recording
                or writer is None
                or self._flags.shutting_down
            ):
                return  # not recording or shutting down

            # Timestamp: always use arrival time
            ts_ns = self.get_clock().now().nanoseconds

            data = serialize_message(msg)
            try:
                with self._ws.writer_lock:
                    if self._ws.writer is not None:
                        self._ws.writer.write(_topic, data, ts_ns)
                        if cnt:
                            cnt.written += 1
            except (RuntimeError, OSError, ValueError) as exc:
                # Signal fatal; execute loop will finalize
                self._flags.fatal_error = True
                self.get_logger().error(
                    f"Write failed on {_topic}: {exc!r}\n{traceback.format_exc()}"
                )
                self._flags.stop_requested = True
                self._episode_done_evt.set()

        return self.create_subscription(
            msg_cls, topic, cb, qos, callback_group=self._cbg
        )

    # ---------- per-episode helpers ----------

    def _get_total_messages_written(self) -> int:
        """Calculate total messages written by summing all topic counters."""
        return sum(cnt.written for cnt in self._ws.counts.values())

    def _start_feedback_timer(self, end_time: Time) -> None:
        """(Re)create the 2 Hz feedback timer.

        Parameters
        ----------
        end_time : rclpy.time.Time
            Episode wall-clock deadline used to populate `seconds_remaining`.
        """
        if self._feedback_timer is not None:
            self.destroy_timer(self._feedback_timer)
            self._feedback_timer = None

        fb = RecordEpisode.Feedback()

        def _tick() -> None:
            # If not recording, allow executor to clean this up after finalize
            if not self._flags.is_recording or self._current_goal_handle is None:
                return
            # Early return if goal is canceled to prevent spurious publish
            if self._current_goal_handle.is_cancel_requested:
                return
            now = self.get_clock().now()
            remaining_ns = max(0, end_time.nanoseconds - now.nanoseconds)
            fb.seconds_remaining = remaining_ns // 1_000_000_000
            fb.feedback_message = f"writing… total={self._get_total_messages_written()}"
            try:
                self._current_goal_handle.publish_feedback(fb)
            except Exception as exc:  # client may vanish mid-episode
                self.get_logger().warning(f"Feedback publish failed: {exc!r}")

        self._feedback_timer = self.create_timer(
            FEEDBACK_PERIOD_S, _tick, callback_group=self._cbg
        )

    def _start_timeout_timer(self, max_duration_s: float) -> None:
        """Create a one-shot timer that stops the episode when it fires."""
        if self._timeout_timer is not None:
            self.destroy_timer(self._timeout_timer)
            self._timeout_timer = None

        def _on_timeout() -> None:
            if not self._flags.is_recording:
                return
            self.get_logger().info("Episode timeout reached.")
            self._flags.stop_requested = True
            self._episode_done_evt.set()
            # Destroy oneself (one-shot)
            if self._timeout_timer is not None:
                self.destroy_timer(self._timeout_timer)

        self._timeout_timer = self.create_timer(
            float(max_duration_s), _on_timeout, callback_group=self._cbg
        )

    def _unique_bag_dir(self) -> Path:
        """Generate a unique bag directory name based on system time.

        Returns
        -------
        pathlib.Path
            A non-existent directory path under `bag_base_dir`.
        """
        sec = int(time.time())
        nsec = int((time.time() - sec) * 1e9)
        base = f"{sec:010d}_{nsec:09d}"
        bag_dir = self._bag_base / base
        suffix = 1
        while bag_dir.exists():
            bag_dir = self._bag_base / f"{base}_{suffix}"
            suffix += 1
        return bag_dir

    def _finalize_episode(self, bag_dir: Path, prompt: str) -> None:
        """Close writer, patch metadata, and tear down per-episode timers.

        Parameters
        ----------
        bag_dir : pathlib.Path
            Directory of the recorded bag.
        prompt : str
            Operator prompt to embed into `metadata.yaml` (if non-empty).
        """
        # Close writer, keep subs alive
        with self._ws.writer_lock:
            self._ws.writer = None

        # Write metadata
        self._write_episode_metadata(bag_dir, prompt)

        # Kill timers
        if self._feedback_timer is not None:
            self.destroy_timer(self._feedback_timer)
            self._feedback_timer = None
        if self._timeout_timer is not None:
            self.destroy_timer(self._timeout_timer)
            self._timeout_timer = None

        # Clear latch and reset flags
        self._episode_done_evt.clear()
        self._flags.stop_requested = False
        self._flags.is_recording = False

    # ---------- main action loop ----------

    def execute_callback(self, goal_handle: Any) -> RecordEpisode.Result:
        """Execute a single recording episode to completion.

        Orchestrates writer creation/topic registration, feedback/timeout
        timers, and the blocking wait until the episode ends due to cancel,
        timeout, or error. On completion, closes the writer and amends metadata.

        Returns
        -------
        RecordEpisode.Result
            Success flag and summary message.
        """
        # Store goal handle and transition to EXECUTING state
        self._current_goal_handle = goal_handle
        self._flags.is_recording = True
        self._flags.fatal_error = False
        self._flags.stop_requested = False
        for k in list(self._ws.counts.keys()):
            self._ws.counts[k] = _TopicCounter()  # reset per-episode counters

        prompt = getattr(goal_handle.request, "prompt", "") or ""
        storage = (
            (self._contract.recording.get("storage") or "mcap")
            if self._contract.recording
            else "mcap"
        )
        max_s = float(getattr(self._contract, "max_duration_s", 300.0))

        # Unique episode dir (system time)
        bag_dir = self._unique_bag_dir()

        # Open writer + register topics
        try:
            with self._ws.writer_lock:
                self._ws.writer = self._open_writer(str(bag_dir), storage)
                for t, typ, _ in self._topics:
                    self._register_topic(t, typ)
        except (RuntimeError, OSError, ValueError) as exc:
            self._flags.is_recording = False
            self._current_goal_handle = None
            goal_handle.abort()
            msg = f"Failed to open writer: {exc!r}"
            self.get_logger().error(msg)
            return RecordEpisode.Result(success=False, message=msg)

        # Start timeout and feedback timers
        start_time = self.get_clock().now()
        end_time = Time(
            nanoseconds=start_time.nanoseconds + int(max_s * 1e9),
            clock_type=start_time.clock_type,
        )
        self._start_feedback_timer(end_time)
        self._start_timeout_timer(max_s)

        # Main execution loop - check for cancel requests and wait for completion
        while (
            self._flags.is_recording
            and not self._flags.fatal_error
            and not self._flags.stop_requested
        ):
            if goal_handle.is_cancel_requested:
                # Transition to CANCELING state
                self._flags.stop_requested = True
                self._episode_done_evt.set()
                break
            self._episode_done_evt.wait(timeout=0.1)

        # Finalize episode (cleanup first)
        total_written = self._get_total_messages_written()
        self._finalize_episode(bag_dir, prompt)
        self._current_goal_handle = None

        # Emit terminal transition exactly once, after cleanup
        if self._flags.fatal_error:
            goal_handle.abort()
            return RecordEpisode.Result(success=False, message="Writer error")
        elif goal_handle.is_cancel_requested or self._flags.stop_requested:
            goal_handle.canceled()
            return RecordEpisode.Result(success=False, message="Cancelled")
        else:
            goal_handle.succeed()
            self.get_logger().info(
                f"Episode complete: wrote {total_written} messages to {bag_dir}"
            )
            return RecordEpisode.Result(
                success=True, message=f"Wrote {total_written} messages to {bag_dir}"
            )

    # ---------- metadata ----------

    def _write_episode_metadata(self, bag_dir: Path, prompt: str) -> None:
        """Patch the bag's `metadata.yaml` with the operator prompt (best-effort).

        Tries multiple times with a short ROS timer delay in case the storage
        backend is still flushing. If the file can't be read/parsed, the
        function silently retries up to `METADATA_RETRIES`.

        Parameters
        ----------
        bag_dir : pathlib.Path
            Directory of the recorded bag.
        prompt : str
            Operator prompt to store under `rosbag2_bagfile_information.custom_data`
            as `lerobot.operator_prompt`.
        """
        if not prompt:
            return
        meta_path = bag_dir / "metadata.yaml"
        # Try a few times in case the writer/storage is still flushing the file
        for _ in range(METADATA_RETRIES):
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = yaml.safe_load(f) or {}
                info = meta.get("rosbag2_bagfile_information") or {}
                custom = info.get("custom_data") or {}
                custom["lerobot.operator_prompt"] = str(prompt)
                info["custom_data"] = custom
                meta["rosbag2_bagfile_information"] = info
                with meta_path.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(meta, f, sort_keys=False)
                return
            except (OSError, yaml.YAMLError):
                # Back off by spinning ROS executor with timeout (non-blocking)
                # This allows other callbacks to execute during the retry delay
                rclpy.spin_once(self, timeout_sec=METADATA_RETRY_PERIOD_S)


def main() -> None:
    """Entry point: start the recorder node and spin a multi-threaded executor."""
    try:
        rclpy.init()
        node = EpisodeRecorderServer()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        # Quiet exit on Ctrl-C or orchestrated shutdown.
        pass
    finally:
        rclpy.shutdown()
