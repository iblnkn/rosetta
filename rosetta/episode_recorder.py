#!/usr/bin/env python3
"""Minimal ROS 2 bag recorder (robust version).

- Subscriptions are created once at node startup and live for the node lifetime.
- Each message callback writes immediately with the node-clock RECEIVE timestamp
  *only if* a writer is open; otherwise it no-ops.
- Action cancel + /record_episode/cancel service are both supported.
- Episode ends on cancel, writer error, or contract max_duration_s.
- Only custom metadata written is `lerobot.operator_prompt`.
"""

from __future__ import annotations

import os
import time
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import yaml
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.serialization import serialize_message
from rosidl_runtime_py.utilities import get_message
from std_srvs.srv import Trigger

import rosbag2_py

from rosetta_interfaces.action import RecordEpisode
from rosetta.common.contract_utils import load_contract, qos_profile_from_dict


@dataclass(slots=True)
class _TopicCounter:
    seen: int = 0
    written: int = 0
    header_missing: int = 0


@dataclass(slots=True)
class Flags:
    is_recording: bool = False
    fatal_error: bool = False
    shutting_down: bool = False


@dataclass(slots=True)
class WriterState:
    writer: Optional[rosbag2_py.SequentialWriter] = None
    writer_lock: threading.Lock = field(default_factory=threading.Lock)
    counts: Dict[str, _TopicCounter] = field(default_factory=dict)
    messages_written: int = 0


class EpisodeRecorderServer(Node):
    """Episode recorder that writes messages as they arrive (no alignment/caching)."""

    def __init__(self) -> None:
        super().__init__('recorder_server')

        # Parameters
        self.declare_parameter('contract_path', '')
        self.declare_parameter('bag_base_dir', '/tmp/episodes')

        cp = self.get_parameter('contract_path').get_parameter_value().string_value
        if not cp:
            raise RuntimeError("Parameter 'contract_path' is required (path to YAML contract).")
        self._contract = load_contract(cp)

        self._bag_base = self.get_parameter('bag_base_dir').get_parameter_value().string_value
        os.makedirs(self._bag_base, exist_ok=True)

        self._flags = Flags()
        self._ws = WriterState()

        # Executor/callback group
        self._cbg = ReentrantCallbackGroup()

        # Build topic list from contract once
        obs = self._contract.observations or []
        tks = self._contract.tasks or []
        acts = self._contract.actions or []
        self._topics: List[Tuple[str, str, Dict]] = []
        self._topics += [(o.topic, o.type, o.qos or {}) for o in obs]
        self._topics += [(t.topic, t.type, t.qos or {}) for t in tks]
        self._topics += [(a.publish_topic, a.type, a.publish_qos or {}) for a in acts]

        # Create subscriptions ONCE; callbacks will no-op when writer is None
        self._subs: List[Any] = []
        for topic, type_str, qos_dict in self._topics:
            self._ws.counts[topic] = _TopicCounter()
            self._subs.append(self._make_sub(topic, type_str, qos_dict))

        # Action server
        self._current_goal_handle = None
        self._server = ActionServer(
            self,
            RecordEpisode,
            'record_episode',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self._cbg,
        )

        # Cancel service
        self._cancel_service = self.create_service(
            Trigger, 'record_episode/cancel', self._cancel_service_cb, callback_group=self._cbg
        )

        # Shutdown hook
        self.context.on_shutdown(self._shutdown_cb)

        self.get_logger().info(f"Recorder ready with contract '{self._contract.name}'.")

    # ---------- Action callbacks ----------

    def goal_callback(self, _req) -> GoalResponse:
        if self._flags.is_recording:
            self.get_logger().warning('Rejecting goal: already recording')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, _goal_handle) -> CancelResponse:
        self.get_logger().info('Action cancel requested')
        # Just drop the flag; execute loop will observe and finish
        self._flags.is_recording = False
        return CancelResponse.ACCEPT

    def _cancel_service_cb(self, _req, resp):
        self.get_logger().info('Cancel service called')
        self._flags.is_recording = False
        resp.success = True
        resp.message = 'Recording cancelled'
        return resp

    def _shutdown_cb(self):
        """Abort current goal cleanly on shutdown; do NOT destroy subscriptions."""
        if self._flags.is_recording and self._current_goal_handle is not None:
            self.get_logger().info('Node shutting down, aborting current goal')
            self._flags.is_recording = False
            try:
                self._current_goal_handle.abort()
            except Exception as e:
                self.get_logger().warning(f'Failed to abort goal during shutdown: {e}')
        # Close writer if any
        with self._ws.writer_lock:
            self._ws.writer = None

    # ---------- rosbag2 helpers ----------

    def _open_writer(self, bag_uri: str, storage_id: str) -> rosbag2_py.SequentialWriter:
        storage_options = rosbag2_py.StorageOptions(uri=bag_uri, storage_id=storage_id)
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr',
        )
        writer = rosbag2_py.SequentialWriter()
        writer.open(storage_options, converter_options)
        return writer

    def _register_topic(self, topic: str, type_str: str) -> None:
        meta = rosbag2_py.TopicMetadata(name=topic, type=type_str, serialization_format='cdr')
        assert self._ws.writer is not None
        self._ws.writer.create_topic(meta)

    def _make_sub(self, topic: str, type_str: str, qos_dict: Dict) -> Any:
        """Create a subscription that writes each message with *receive time* when writer is open."""
        msg_cls = get_message(type_str)
        qos = qos_profile_from_dict(qos_dict) or QoSProfile(depth=10)

        def cb(msg, _topic=topic) -> None:
            # Always gather counters; write only if writer is live
            cnt = self._ws.counts.get(_topic)
            if cnt:
                cnt.seen += 1
                if not getattr(getattr(msg, 'header', None), 'stamp', None):
                    cnt.header_missing += 1

            with self._ws.writer_lock:
                writer = self._ws.writer

            if not self._flags.is_recording or writer is None:
                return  # not recording

            now_ns = self.get_clock().now().nanoseconds  # RECEIVE time (design choice)
            data = serialize_message(msg)
            try:
                with self._ws.writer_lock:
                    if self._ws.writer is not None:
                        self._ws.writer.write(_topic, data, now_ns)
                        self._ws.messages_written += 1
                        if cnt:
                            cnt.written += 1
            except (RuntimeError, OSError, ValueError) as exc:
                # Signal fatal and let execute loop stop gracefully
                self._flags.fatal_error = True
                self.get_logger().error(f'Write failed on {_topic}: {exc!r}\n{traceback.format_exc()}')

        return self.create_subscription(msg_cls, topic, cb, qos, callback_group=self._cbg)

    # ---------- main action loop ----------

    def execute_callback(self, goal_handle) -> RecordEpisode.Result:
        if self._flags.is_recording:
            self.get_logger().warning('Already recording, rejecting new goal')
            goal_handle.abort()
            return RecordEpisode.Result(success=False, message='Already recording')

        self._current_goal_handle = goal_handle
        self._flags.is_recording = True
        self._flags.fatal_error = False
        self._ws.messages_written = 0
        for k in list(self._ws.counts.keys()):
            self._ws.counts[k] = _TopicCounter()  # reset per-episode counters

        prompt = getattr(goal_handle.request, 'prompt', '')
        storage = (self._contract.recording.get('storage') or 'mcap') if self._contract.recording else 'mcap'
        max_s = float(getattr(self._contract, 'max_duration_s', 300.0))

        # Unique episode dir
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        bag_dir = os.path.join(self._bag_base, f'{stamp}')
        suffix = 1
        while os.path.exists(bag_dir):
            bag_dir = os.path.join(self._bag_base, f'{stamp}_{suffix}')
            suffix += 1

        # Open writer
        try:
            with self._ws.writer_lock:
                self._ws.writer = self._open_writer(bag_dir, storage)
                # Register topics for this bag
                for t, typ, _ in self._topics:
                    self._register_topic(t, typ)
        except (RuntimeError, OSError, ValueError) as exc:
            self._flags.is_recording = False
            self._current_goal_handle = None
            goal_handle.abort()
            msg = f'Failed to open writer: {exc!r}'
            self.get_logger().error(msg)
            return RecordEpisode.Result(success=False, message=msg)

        # Timing & feedback
        end_ns = self.get_clock().now().nanoseconds + int(max_s * 1e9)
        fb = RecordEpisode.Feedback()
        next_feedback = time.monotonic()
        feedback_period = 0.5

        try:
            while rclpy.ok():
                if self._flags.fatal_error:
                    self._flags.is_recording = False
                    goal_handle.abort()
                    self._write_episode_metadata(bag_dir, prompt)
                    return RecordEpisode.Result(success=False, message='Writer error')

                if goal_handle.is_cancel_requested:
                    self._flags.is_recording = False
                    goal_handle.canceled()
                    self._write_episode_metadata(bag_dir, prompt)
                    return RecordEpisode.Result(success=False, message='Cancelled')

                now_ns = self.get_clock().now().nanoseconds
                if now_ns >= end_ns:
                    break

                now = time.monotonic()
                if now >= next_feedback:
                    fb.seconds_remaining = max(0, int((end_ns - now_ns) / 1_000_000_000))
                    fb.feedback_message = f'writing… total={self._ws.messages_written}'
                    goal_handle.publish_feedback(fb)
                    next_feedback = now + feedback_period

                time.sleep(0.02)  # callbacks do the work
        finally:
            total_written = self._ws.messages_written
            # Close writer, keep subs alive
            with self._ws.writer_lock:
                self._ws.writer = None
            self._write_episode_metadata(bag_dir, prompt)
            self._current_goal_handle = None
            self._flags.is_recording = False

        goal_handle.succeed()
        msg = f'Wrote {total_written} messages to {bag_dir}'
        return RecordEpisode.Result(success=True, message=msg)

    # ---------- metadata ----------

    def _write_episode_metadata(self, bag_dir: str, prompt: str) -> None:
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
