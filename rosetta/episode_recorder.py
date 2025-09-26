from __future__ import annotations
import os
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import yaml  # only for writing the prompt into metadata.yaml

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import ExternalShutdownException
from rclpy.serialization import serialize_message
from rosidl_runtime_py.utilities import get_message

import rosbag2_py

from rosetta_interfaces.action import RecordEpisode
from .contract_utils import load_contract, qos_profile_from_dict


class EpisodeRecorderServer(Node):
    """
    Minimal ROS 2 episode recorder (Option A: tasks are topics)

    - Reads contract (observations / tasks / actions / recording).
    - Subscribes to observations + tasks + action publish topics.
    - Two write modes:
        * pass-through (default): write on every receive, stamp = msg.header.stamp if present else node time
        * sample-to-rate (recording.sample_to_rate = true):
            - Cache latest per topic.
            - At rate_hz ticks, write with tick timestamp.
            - Per-topic optional align:
                - strategy: 'hold' (default) | 'asof' | 'drop'
                - tol_ms: int (for 'asof')
                - stamp: 'header' (default if present) | 'receive'
              * 'hold': always write cached sample if any
              * 'asof': write only if cached sample staleness <= tol_ms
              * 'drop': write only if a new sample arrived since last tick
    - Stores action goal prompt into metadata.yaml under custom_data.lerobot.operator_prompt
    """

    def __init__(self):
        super().__init__("episode_recorder_server")

        # Parameters
        self.declare_parameter("contract_path", "")
        self.declare_parameter("bag_base_dir", "/tmp/episodes")

        cp = self.get_parameter("contract_path").get_parameter_value().string_value
        if not cp:
            raise RuntimeError("Parameter 'contract_path' is required (path to YAML contract).")
        self._contract = load_contract(cp)

        self._bag_base = self.get_parameter("bag_base_dir").get_parameter_value().string_value

        # Writer state
        self._writer: Optional[rosbag2_py.SequentialWriter] = None
        self._writer_lock = threading.Lock()
        self._subs = []
        self._messages_written = 0

        # Sampling-to-rate state (used only if enabled)
        self._sample_to_rate: bool = bool(self._contract.recording.get("sample_to_rate", False))
        self._cache: Dict[str, Dict] = {}          # topic -> {data, header_ns, recv_ns, fresh, align}
        self._tick_timer = None
        self._last_tick_ns = 0

        # Action server
        self._server = ActionServer(
            self,
            RecordEpisode,
            "record_episode",
            execute_callback=self.execute_callback,
            goal_callback=lambda _req: GoalResponse.ACCEPT,
            cancel_callback=self.cancel_callback,
        )
        self.get_logger().info(
            f"Recorder ready with contract '{self._contract.name}', "
            f"sample_to_rate={self._sample_to_rate}, rate_hz={self._contract.rate_hz:.3f}."
        )

    def cancel_callback(self, _goal_handle):
        self.get_logger().info("Cancel requested.")
        return CancelResponse.ACCEPT

    # -------- rosbag2 helpers --------

    def _open_writer(self, bag_uri: str, storage_id: str):
        storage_options = rosbag2_py.StorageOptions(uri=bag_uri, storage_id=storage_id)
        converter_options = rosbag2_py.ConverterOptions("", "")
        w = rosbag2_py.SequentialWriter()
        w.open(storage_options, converter_options)
        return w

    def _register_topic(self, topic: str, type_str: str):
        meta = rosbag2_py.TopicMetadata(name=topic, type=type_str, serialization_format="cdr")
        self._writer.create_topic(meta)

    def _make_sub(self, topic: str, type_str: str, qos_dict: Dict, align: Optional[Dict]):
        msg_cls = get_message(type_str)
        qos = qos_profile_from_dict(qos_dict) or 10  # default queue depth

        # Cache align settings for tick-based writer
        if self._sample_to_rate:
            self._cache[topic] = {
                "data": None,
                "header_ns": None,
                "recv_ns": None,
                "fresh": False,
                "align": align or {},  # {strategy, tol_ms, stamp}
            }

        def cb(msg, _topic=topic):
            # Time sources
            recv_ns = self.get_clock().now().nanoseconds
            header_ns = None
            try:
                hdr = getattr(msg, "header", None)
                if hdr is not None and hasattr(hdr, "stamp"):
                    header_ns = int(hdr.stamp.sec) * 1_000_000_000 + int(hdr.stamp.nanosec)
            except Exception:
                header_ns = None

            data = serialize_message(msg)

            if self._sample_to_rate:
                # Only cache; timer will decide if/when to write
                with self._writer_lock:
                    ent = self._cache.get(_topic)
                    if ent is not None:
                        ent["data"] = data
                        ent["header_ns"] = header_ns
                        ent["recv_ns"] = recv_ns
                        ent["fresh"] = True
                return

            # Pass-through mode: write immediately (use header if present)
            ts_ns = header_ns if header_ns is not None else recv_ns
            with self._writer_lock:
                if self._writer is not None:
                    self._writer.write(_topic, data, ts_ns)
                    self._messages_written += 1

        return self.create_subscription(msg_cls, topic, cb, qos)

    # -------- tick writer (sample_to_rate) --------

    def _on_tick(self):
        tick_ns = self.get_clock().now().nanoseconds
        with self._writer_lock:
            if self._writer is None:
                return
            for topic, ent in self._cache.items():
                data = ent["data"]
                if data is None:
                    continue  # never seen

                header_ns = ent["header_ns"]
                recv_ns = ent["recv_ns"]
                fresh = ent["fresh"]
                align = ent["align"] or {}

                strategy = str(align.get("strategy", "hold")).lower()
                tol_ns = int(align.get("tol_ms", 0)) * 1_000_000
                stamp_sel = str(align.get("stamp", "header")).lower()
                use_header = (stamp_sel == "header") and (header_ns is not None)
                src_ns = header_ns if use_header else (recv_ns if recv_ns is not None else tick_ns)
                staleness = tick_ns - src_ns

                should_write = False
                if strategy == "drop":
                    should_write = bool(fresh)
                elif strategy == "asof":
                    should_write = (tol_ns <= 0) or (staleness <= tol_ns)
                else:  # hold (default)
                    should_write = True

                if should_write:
                    # Stamp with the tick to align on a uniform grid
                    self._writer.write(topic, data, tick_ns)
                    self._messages_written += 1

                # fresh consumed for this tick
                ent["fresh"] = False

        self._last_tick_ns = tick_ns

    # -------- main action loop --------

    async def execute_callback(self, goal_handle):
        self._messages_written = 0
        prompt = getattr(goal_handle.request, "prompt", "")

        storage = (self._contract.recording.get("storage") or "mcap")
        seconds = float(self._contract.max_duration_s)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bag_dir = os.path.join(self._bag_base, f"{stamp}")

        # observations + tasks + action publish topics
        obs = self._contract.observations or []
        tks = self._contract.tasks or []
        acts = self._contract.actions or []

        topics: List[Tuple[str, str, Dict, Optional[Dict]]] = []
        topics += [(o.topic, o.type, o.qos or {}, getattr(o, "align", None)) for o in obs]
        topics += [(t.topic, t.type, t.qos or {}, getattr(t, "align", None)) for t in tks]
        topics += [(a.publish.topic, a.publish.type, a.publish.qos or {}, None) for a in acts]

        # Deduplicate by topic (first one wins)
        dedup: Dict[str, Tuple[str, Dict, Optional[Dict]]] = {}
        for t, typ, qos, align in topics:
            if t not in dedup:
                dedup[t] = (typ, qos, align)
        topics = [(t, typ, qos, align) for t, (typ, qos, align) in dedup.items()]

        # Open writer
        try:
            self._writer = self._open_writer(bag_dir, storage)
        except Exception as exc:
            goal_handle.abort()
            return RecordEpisode.Result(success=False, message=f"Failed to open writer: {exc}")

        # Register topics and subscribe
        try:
            with self._writer_lock:
                for t, typ, _q, _a in topics:
                    self._register_topic(t, typ)
            self._subs = [self._make_sub(t, typ, qos, align) for (t, typ, qos, align) in topics]
        except Exception as exc:
            goal_handle.abort()
            self._teardown()
            return RecordEpisode.Result(success=False, message=f"Failed to set up topics/subs: {exc}")

        # Start rate sampler if enabled
        if self._sample_to_rate:
            period = 1.0 / float(self._contract.rate_hz or 20.0)
            self._tick_timer = self.create_timer(period, self._on_tick)

        self.get_logger().info(
            f"Recording -> {bag_dir} (storage={storage}, sample_to_rate={self._sample_to_rate})"
        )

        end_ns = self.get_clock().now().nanoseconds + int(seconds * 1_000_000_000)
        fb = RecordEpisode.Feedback()

        try:
            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    self._teardown()
                    return RecordEpisode.Result(success=False, message="Cancelled")

                now_ns = self.get_clock().now().nanoseconds
                remaining = max(0, int((end_ns - now_ns) / 1_000_000_000))
                fb.seconds_remaining = remaining
                fb.feedback_message = f"writing… total={self._messages_written}"
                goal_handle.publish_feedback(fb)

                if now_ns >= end_ns:
                    break

                await asyncio.sleep(0.5)
        finally:
            self._teardown()

        # Persist the operator prompt alongside rosbag metadata (tasks are topics)
        self._write_episode_prompt(bag_dir, prompt)

        goal_handle.succeed()
        msg = f"Wrote {self._messages_written} messages to {bag_dir}"
        return RecordEpisode.Result(success=True, message=msg)

    # -------- teardown & metadata --------

    def _teardown(self):
        # Stop tick timer first
        if self._tick_timer is not None:
            try:
                self.destroy_timer(self._tick_timer)
            except Exception:
                pass
            self._tick_timer = None

        # Drop subscriptions so no more writes come in
        for s in self._subs:
            try:
                self.destroy_subscription(s)
            except Exception:
                pass
        self._subs.clear()

        # Clear caches
        self._cache.clear()
        self._last_tick_ns = 0

        # Close writer
        with self._writer_lock:
            self._writer = None  # writer closes via destructor

    def _write_episode_prompt(self, bag_dir: str, prompt: str):
        """
        Append the operator prompt to metadata.yaml under:
          custom_data:
            lerobot.operator_prompt: "<prompt>"
        Ignore errors silently to keep the recording valid.
        """
        if not prompt:
            return
        meta_path = os.path.join(bag_dir, "metadata.yaml")
        try:
            with open(meta_path, "r") as f:
                meta = yaml.safe_load(f) or {}
            custom = (meta.get("custom_data") or {})
            custom["lerobot.operator_prompt"] = str(prompt)
            meta["custom_data"] = custom
            with open(meta_path, "w") as f:
                yaml.safe_dump(meta, f, sort_keys=False)
        except Exception as exc:
            self.get_logger().warn(f"Could not write operator_prompt into metadata.yaml: {exc}")


def main():
    try:
        rclpy.init()
        node = EpisodeRecorderServer()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        rclpy.shutdown()
