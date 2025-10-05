#!/usr/bin/env python3
"""Minimal ROS 2 bag recorder.

Subscribes per contract and writes each message immediately using the node clock
timestamp. The only custom metadata written is `lerobot.operator_prompt`.
"""

from __future__ import annotations

import os
import time
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import yaml

import rclpy
from rclpy.action import ActionServer
from rclpy.action import CancelResponse
from rclpy.action import GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.serialization import serialize_message
from rosidl_runtime_py.utilities import get_message

import rosbag2_py

from rosetta_interfaces.action import RecordEpisode
from .contract_utils import load_contract
from .contract_utils import qos_profile_from_dict


@dataclass(slots=True)
class _TopicCounter:
    seen: int = 0
    written: int = 0
    header_missing: int = 0


@dataclass(slots=True)
class Flags:
    """Recording state flags."""
    is_recording: bool = False
    fatal_error: bool = False
    shutting_down: bool = False


@dataclass(slots=True)
class WriterState:
    """Writer state and subscriptions."""
    writer: Optional[rosbag2_py.SequentialWriter] = None
    writer_lock: threading.Lock = threading.Lock()
    subs: List[Any] = None
    counts: Dict[str, _TopicCounter] = None
    messages_written: int = 0

    def __post_init__(self) -> None:
        if self.subs is None:
            self.subs = []
        if self.counts is None:
            self.counts = {}


class EpisodeRecorderServer(Node):
    """Episode recorder that writes messages as they arrive (no alignment/caching)."""

    def __init__(self) -> None:
        """Initialize node, parameters, writer state, and action server."""
        super().__init__('episode_recorder_server')

        # Parameters
        self.declare_parameter('contract_path', '')
        self.declare_parameter('bag_base_dir', '/tmp/episodes')

        cp = self.get_parameter('contract_path').get_parameter_value().string_value
        if not cp:
            raise RuntimeError(
                "Parameter 'contract_path' is required (path to YAML contract)."
            )
        self._contract = load_contract(cp)
        self._bag_base = self.get_parameter('bag_base_dir').get_parameter_value().string_value

        # Consolidated flags & writer state (keep attribute count low)
        self._flags = Flags()
        self._ws = WriterState()

        # Allow concurrent callbacks with a multithreaded executor
        self._cbg = ReentrantCallbackGroup()

        # Action server
        self._server = ActionServer(
            self,
            RecordEpisode,
            'record_episode',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self._cbg,
        )
        self.get_logger().info(
            f"Recorder ready with contract '{self._contract.name}'."
        )

    # -------- Action callbacks --------

    def goal_callback(self, _req) -> GoalResponse:
        """Accept only if not already recording."""
        if self._flags.is_recording:
            self.get_logger().warning('Rejecting goal: already recording')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, _goal_handle) -> CancelResponse:
        """Always allow cancel."""
        self.get_logger().info('Cancel requested.')
        return CancelResponse.ACCEPT

    # -------- rosbag2 helpers --------

    def _open_writer(self, bag_uri: str, storage_id: str) -> rosbag2_py.SequentialWriter:
        """Create and open a SequentialWriter for the given bag URI and storage."""
        storage_options = rosbag2_py.StorageOptions(uri=bag_uri, storage_id=storage_id)
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr',
        )
        writer = rosbag2_py.SequentialWriter()
        writer.open(storage_options, converter_options)
        return writer

    def _register_topic(self, topic: str, type_str: str) -> None:
        """Declare a topic in the bag before writing to it."""
        meta = rosbag2_py.TopicMetadata(name=topic, type=type_str, serialization_format='cdr')
        assert self._ws.writer is not None
        self._ws.writer.create_topic(meta)

    def _make_sub(self, topic: str, type_str: str, qos_dict: Dict) -> Any:
        """Create a subscription that writes every message to the bag immediately."""
        msg_cls = get_message(type_str)

        qos = qos_profile_from_dict(qos_dict)
        if qos is None:
            qos = QoSProfile(depth=10)

        self._ws.counts[topic] = _TopicCounter()

        def cb(msg, _topic=topic) -> None:
            if self._flags.shutting_down:
                return

            # Timestamp: node clock; mirrors the tutorial pattern
            now_ns = self.get_clock().now().nanoseconds
            data = serialize_message(msg)

            cnt = self._ws.counts.get(_topic)
            if cnt:
                cnt.seen += 1
                if not getattr(getattr(msg, 'header', None), 'stamp', None):
                    cnt.header_missing += 1

            try:
                with self._ws.writer_lock:
                    if self._ws.writer is not None:
                        self._ws.writer.write(_topic, data, now_ns)
                        self._ws.messages_written += 1
                        if cnt:
                            cnt.written += 1
            except (RuntimeError, OSError, ValueError) as exc:
                # Fail fast; execute loop will shut down and write metadata
                self._flags.fatal_error = True
                self.get_logger().error(
                    f'Write failed on {_topic}: {exc!r}\n{traceback.format_exc()}'
                )

        return self.create_subscription(
            msg_cls,
            topic,
            cb,
            qos,
            callback_group=self._cbg,
        )

    # -------- main action loop (no timer; just spin) --------

    def execute_callback(self, goal_handle) -> RecordEpisode.Result:
        """Execute the recording goal until duration elapses or canceled/error."""
        if self._flags.is_recording:
            self.get_logger().warning('Already recording, rejecting new goal')
            goal_handle.abort()
            return RecordEpisode.Result(success=False, message='Already recording')

        self._flags.is_recording = True
        self._flags.fatal_error = False
        self._ws.messages_written = 0
        prompt = getattr(goal_handle.request, 'prompt', '')

        storage = (self._contract.recording.get('storage') or 'mcap')
        seconds = float(self._contract.max_duration_s)

        # Unique episode directory (rosbag2 requires it not exist)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        bag_dir = os.path.join(self._bag_base, f'{stamp}')

        # Build topic list from contract
        obs = self._contract.observations or []
        tks = self._contract.tasks or []
        acts = self._contract.actions or []

        topics: List[Tuple[str, str, Dict]] = []
        topics += [(o.topic, o.type, o.qos or {}) for o in obs]
        topics += [(t.topic, t.type, t.qos or {}) for t in tks]
        topics += [(a.publish.topic, a.publish.type, a.publish.qos or {}) for a in acts]

        # Open writer
        try:
            self._ws.writer = self._open_writer(bag_dir, storage)
        except (RuntimeError, OSError, ValueError) as exc:
            self._flags.is_recording = False
            goal_handle.abort()
            msg = f'Failed to open writer: {exc!r}'
            self.get_logger().error(msg)
            return RecordEpisode.Result(success=False, message=msg)

        # Register topics and subscribe
        try:
            with self._ws.writer_lock:
                for t, typ, _q in topics:
                    self._register_topic(t, typ)
            self._ws.subs = [self._make_sub(t, typ, qos) for (t, typ, qos) in topics]
        except (RuntimeError, OSError, ValueError) as exc:
            self._flags.is_recording = False
            goal_handle.abort()
            self._teardown()
            msg = f'Failed to set up topics/subs: {exc!r}'
            self.get_logger().error(msg)
            return RecordEpisode.Result(success=False, message=msg)

        end_ns = self.get_clock().now().nanoseconds + int(seconds * 1_000_000_000)
        fb = RecordEpisode.Feedback()
        next_feedback = time.monotonic()
        feedback_period = 0.5  # seconds

        try:
            while rclpy.ok():
                if self._flags.fatal_error:
                    self._flags.is_recording = False
                    goal_handle.abort()
                    self._teardown()
                    self._write_episode_metadata(bag_dir, prompt)
                    return RecordEpisode.Result(
                        success=False, message='Recording stopped due to writer error'
                    )

                if goal_handle.is_cancel_requested:
                    self._flags.is_recording = False
                    goal_handle.canceled()
                    self._teardown()
                    self._write_episode_metadata(bag_dir, prompt)
                    return RecordEpisode.Result(success=False, message='Cancelled')

                now_ns = self.get_clock().now().nanoseconds
                remaining = max(0, int((end_ns - now_ns) / 1_000_000_000))
                now = time.monotonic()
                if now >= next_feedback:
                    fb.seconds_remaining = remaining
                    fb.feedback_message = (
                        f'writing… total={self._ws.messages_written}'
                    )
                    goal_handle.publish_feedback(fb)
                    next_feedback = now + feedback_period

                if now_ns >= end_ns:
                    break

                # Sleep briefly to avoid spinning hot; callbacks do the real work
                time.sleep(0.02)
        finally:
            total_written = self._ws.messages_written  # capture before teardown resets
            self._teardown()
            self._write_episode_metadata(bag_dir, prompt)

        self._flags.is_recording = False
        goal_handle.succeed()
        msg = f'Wrote {total_written} messages to {bag_dir}'
        return RecordEpisode.Result(success=True, message=msg)
    # -------- teardown & metadata --------

    def _teardown(self) -> None:
        """Stop subscriptions, drop writer, and reset state flags."""
        self._flags.shutting_down = True

        with self._ws.writer_lock:
            for s in self._ws.subs:
                try:
                    self.destroy_subscription(s)
                except (RuntimeError, ValueError, TypeError):
                    # Best-effort cleanup; safe to ignore during shutdown
                    pass
            self._ws.subs.clear()
            self._ws.writer = None

        self._ws.counts.clear()
        self._ws.messages_written = 0
        self._flags.is_recording = False
        self._flags.shutting_down = False

    def _write_episode_metadata(self, bag_dir: str, prompt: str) -> None:
        """Write only `lerobot.operator_prompt` into bag metadata.yaml (if provided)."""
        if not prompt:
            return

        meta_path = os.path.join(bag_dir, 'metadata.yaml')
        for _ in range(20):  # ~2s
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = yaml.safe_load(f) or {}
                info = meta.get('rosbag2_bagfile_information') or {}
                custom = info.get('custom_data') or {}
                custom['lerobot.operator_prompt'] = str(prompt)
                info['custom_data'] = custom
                meta['rosbag2_bagfile_information'] = info
                with open(meta_path, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(meta, f, sort_keys=False)
                return
            except (OSError, yaml.YAMLError):
                time.sleep(0.1)


def main() -> None:
    """Entry point: spin the EpisodeRecorderServer with a multithreaded executor."""
    try:
        rclpy.init()
        node = EpisodeRecorderServer()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        rclpy.shutdown()
