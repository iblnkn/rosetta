import os
import re
import time
import threading
from datetime import datetime
from typing import Dict, List

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse
from rclpy.executors import ExternalShutdownException
from rclpy.serialization import serialize_message
from rosidl_runtime_py.utilities import get_message

import rosbag2_py
from rosetta_interfaces.action import RecordEpisode
from .contract_utils import load_contract

def _slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r'\s+', '-', text)
    text = re.sub(r'[^a-z0-9\-_]+', '', text)
    return text or 'episode'

class EpisodeRecorderServer(Node):
    def __init__(self):
        super().__init__('episode_recorder_server')
        self.declare_parameter('contract_path', '')
        cp = self.get_parameter('contract_path').get_parameter_value().string_value
        if not cp:
            raise RuntimeError("Parameter 'contract_path' is required (path to YAML).")
        self._contract = load_contract(cp)

        self._writer = None
        self._writer_lock = threading.Lock()
        self._subs = []
        self._messages_written = 0

        self._server = ActionServer(
            self,
            RecordEpisode,
            'record_episode',
            execute_callback=self.execute_callback,
            goal_callback=lambda _req: rclpy.action.GoalResponse.ACCEPT,
            cancel_callback=self.cancel_callback,
        )
        self.get_logger().info(f"Recorder ready with contract '{self._contract.name}'.")

    def cancel_callback(self, _goal_handle):
        self.get_logger().info('Cancel requested.')
        return CancelResponse.ACCEPT

    def _resolve_topic_types(self, topics: List[str]) -> Dict[str, str]:
        names_and_types = dict(self.get_topic_names_and_types())
        resolved = {}
        for t in topics:
            if t in names_and_types and names_and_types[t]:
                resolved[t] = names_and_types[t][0]
            else:
                self.get_logger().warn(f"Topic '{t}' not found (or has no type); skipping.")
        return resolved

    def _open_writer(self, bag_uri: str, storage_id: str):
        os.makedirs(bag_uri, exist_ok=False)
        storage_options = rosbag2_py.StorageOptions(uri=bag_uri, storage_id=storage_id)
        converter_options = rosbag2_py.ConverterOptions('', '')
        w = rosbag2_py.SequentialWriter()
        w.open(storage_options, converter_options)
        return w

    def _register_topic(self, topic: str, type_str: str):
        meta = rosbag2_py.TopicMetadata(
            name=topic, type=type_str, serialization_format='cdr')
        self._writer.create_topic(meta)

    def _make_sub(self, topic: str, type_str: str):
        msg_cls = get_message(type_str)
        def cb(msg, _topic=topic):
            data = serialize_message(msg)
            ts = self.get_clock().now().nanoseconds
            with self._writer_lock:
                self._writer.write(_topic, data, ts)
                self._messages_written += 1
        return self.create_subscription(msg_cls, topic, cb, 10)

    async def execute_callback(self, goal_handle):
        self._messages_written = 0

        topics = list(self._contract.record_topics or [])
        seconds = int(self._contract.seconds)
        storage = self._contract.storage or 'mcap'
        base = self._contract.bag_base_dir or '/tmp/episodes'

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        bag_dir = os.path.join(base, f"{stamp}_{_slug(goal_handle.request.prompt)}")

        resolved = self._resolve_topic_types(topics)
        try:
            self._writer = self._open_writer(bag_dir, storage)
        except FileExistsError:
            return RecordEpisode.Result(success=False, message=f'Bag path exists: {bag_dir}')
        except Exception as exc:
            return RecordEpisode.Result(success=False, message=f'Failed to open writer: {exc}')

        with self._writer_lock:
            for t, typ in resolved.items():
                self._register_topic(t, typ)

        self._subs = [self._make_sub(t, typ) for t, typ in resolved.items()]

        end_ns = self.get_clock().now().nanoseconds + seconds * 1_000_000_000
        fb = RecordEpisode.Feedback()
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self._teardown()
                return RecordEpisode.Result(success=False, message='Cancelled')
            now_ns = self.get_clock().now().nanoseconds
            remaining = max(0, int((end_ns - now_ns) / 1_000_000_000))
            fb.seconds_remaining = remaining
            fb.feedback_message = f'writing… total={self._messages_written}'
            goal_handle.publish_feedback(fb)
            if now_ns >= end_ns:
                break
            time.sleep(1.0)

        goal_handle.succeed()
        msg = f'Wrote {self._messages_written} messages to {bag_dir}'
        self._teardown()
        return RecordEpisode.Result(success=True, message=msg)

    def _teardown(self):
        for s in self._subs:
            try:
                self.destroy_subscription(s)
            except Exception:
                pass
        self._subs.clear()
        with self._writer_lock:
            self._writer = None

def main():
    try:
        rclpy.init()
        node = EpisodeRecorderServer()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        rclpy.shutdown()
