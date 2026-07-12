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
Bag frame source: backend-neutral frame production from rosbag2 recordings.

The offline side of the backend-neutral interface. It reads rosbag2 files,
decodes and resamples them through the contract (the same StreamBuffer used by
live inference, for train/inference parity), and yields frame dicts
({contract_key: np.ndarray | str, "is_first"/"is_last"/"is_terminal": (1,) bool,
"task": str}) for a DatasetWriter to consume. No dependency on any policy
framework.

Frame assembly is shared with the live TopicBridge via
rosetta.frames.layout.FrameLayout.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import rosbag2_py
import yaml
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from rosetta.contract.specs import StreamSpec
from rosetta.frames.layout import FrameLayout
from rosetta.frames.resample import StreamBuffer

# Register the ROS codecs (import side effect populates the core registry).
from rosetta.robots.ros2 import decoders as _decoders
from rosetta.robots.ros2 import encoders as _encoders
from rosetta.robots.ros2.ingest import StreamIngest

del _decoders, _encoders

# Bag metadata keys
BAG_METADATA_KEY = "rosbag2_bagfile_information"
BAG_CUSTOM_DATA_KEY = "custom_data"
BAG_PROMPT_KEY = "lerobot.operator_prompt"
BAG_CONTRACT_HASH_KEY = "rosetta.contract_hash"


# ---------- Bag discovery ----------


def find_bag_dirs(raw_dir: Path) -> list[Path]:
    """Find all bag directories (identified by metadata.yaml)."""
    bag_dirs = sorted(p.parent for p in raw_dir.rglob("metadata.yaml"))
    if not bag_dirs:
        raise RuntimeError(f"No bag directories found in {raw_dir}")
    return bag_dirs


def read_bag_contract_hash(bag_dir: Path) -> str:
    """Read the recorded contract's sha256 hex digest for one bag (empty if absent)."""
    return _read_contract_hash(_read_bag_metadata(bag_dir))


# ---------- Internal helpers ----------


def _read_bag_metadata(bag_dir: Path) -> dict[str, Any]:
    """Read bag metadata.yaml."""
    meta_path = bag_dir / "metadata.yaml"
    if not meta_path.exists():
        return {}
    with meta_path.open() as f:
        return yaml.safe_load(f) or {}


def _read_prompt(meta: dict[str, Any]) -> str:
    """Read prompt from metadata custom_data."""
    info = meta.get(BAG_METADATA_KEY, {})
    custom_data = info.get(BAG_CUSTOM_DATA_KEY, {})
    if isinstance(custom_data, dict):
        return custom_data.get(BAG_PROMPT_KEY, "")
    return ""


def _read_contract_hash(meta: dict[str, Any]) -> str:
    """Read the recorded contract's sha256 hex digest from metadata custom_data."""
    info = meta.get(BAG_METADATA_KEY, {})
    custom_data = info.get(BAG_CUSTOM_DATA_KEY, {})
    if isinstance(custom_data, dict):
        return custom_data.get(BAG_CONTRACT_HASH_KEY, "")
    return ""


def _get_topic_types(reader: rosbag2_py.SequentialReader) -> dict[str, str]:
    """Get topic -> type mapping from bag."""
    return {t.name: t.type for t in reader.get_all_topics_and_types()}


def _build_buffers(
    specs: list[StreamSpec],
    topic_types: dict[str, str],
) -> tuple[list[tuple[StreamSpec, StreamBuffer]], dict[str, list[int]]]:
    """
    Create per-spec StreamBuffers plus a topic routing table.

    Every spec gets a buffer, in spec order — including specs whose topic is
    missing from the bag (warned; their buffers never fill, so they zero-fill
    at sample time, matching the live bridge and the declared feature shapes).
    Absent *observation* topics are additionally rejected by iter_bag_frames'
    warmup gate — only non-warmup streams (actions, rewards, signals) may
    legitimately be absent. ``routing`` maps each bag topic to the indices of
    all entries reading it (several specs may read one topic with different
    selectors). Raises ValueError if specs routed to one topic declare
    different msg_types (the message is deserialized once per topic).

    Returns
    -------
        (entries, routing): ordered list of (spec, buffer), and
        topic -> [entry indices].

    """
    entries: list[tuple[StreamSpec, StreamBuffer]] = []
    routing: dict[str, list[int]] = {}
    found_any = False

    for i, spec in enumerate(specs):
        entries.append((spec, StreamBuffer.from_spec(spec)))

        if spec.source.channel.topic not in topic_types:
            logging.warning("Topic %s not in bag, %s will be zero-filled", spec.source.channel.topic, spec.key)
            continue
        found_any = True
        routing.setdefault(spec.source.channel.topic, []).append(i)

    if not found_any:
        raise RuntimeError("No contract topics found in bag")

    # One deserialize per topic requires all routed specs to agree on the type.
    for topic, idxs in routing.items():
        types = {entries[i][0].source.channel.type for i in idxs}
        if len(types) > 1:
            raise ValueError(
                f"Topic '{topic}' is declared with conflicting msg_types "
                f"{sorted(types)} across contract specs; unify the entries."
            )

    return entries, routing


def _get_bag_time_bounds_ns(reader: rosbag2_py.SequentialReader) -> tuple[int, int]:
    """Get time bounds from bag metadata."""
    metadata = reader.get_metadata()
    start_time = metadata.starting_time
    duration = metadata.duration
    # rosbag2_py returns Time/Duration objects with .nanoseconds property
    start_ns = start_time.nanoseconds
    duration_ns = duration.nanoseconds
    return start_ns, start_ns + duration_ns


def _sample_frame(
    tick_ns: int,
    entries: list[tuple[StreamSpec, StreamBuffer]],
    layout: FrameLayout,
) -> dict[str, Any]:
    """
    Sample a single frame from buffers at the given tick time.

    Samples every buffer in spec order and passes the values to
    layout.assemble, the by-key aggregation shared with the live bridge.
    """
    return layout.assemble([buffer.sample(tick_ns) for _, buffer in entries])


# Warmup ticks are skipped silently up to these bounds; beyond them the skip is
# surfaced as a warning (late sensor, wrong topic, clock skew are all worth a look).
WARMUP_WARN_SECONDS = 1.0
WARMUP_WARN_FRACTION = 0.1


def iter_bag_frames(
    bag_dir: Path,
    specs: list[StreamSpec],
    *,
    warmup_keys: set[str],
    task_topics: dict[str, str] | None = None,
) -> Iterator[dict[str, Any]]:
    """
    Yield resampled frame dicts from a single bag.

    Uses StreamBuffer for resampling, identical to live inference:

    - A message received exactly at a tick is included in that tick's frame
      (offline tick t covers receive times <= t, matching the live
      ``StreamBuffer.sample`` semantics).
    - Ticks are skipped ("warmup") until every stream whose key is in
      ``warmup_keys`` has at least one sample, so the first emitted frame never
      carries zero-filled observations the way a cold live bridge never serves
      frames before its subscriptions deliver. The tick grid stays anchored at
      the bag start; skipped ticks reduce the emitted count.

    Each frame carries the contract data keys plus is_first/is_last/is_terminal
    boundary markers (is_first marks the first *emitted* frame) and a per-frame
    ``task`` string. When ``task_topics`` (topic -> msg type, from the
    contract's ``tasks`` section) is given, each frame's task is the latest
    string received on a task topic at or before that tick (hold semantics, so
    the task may change mid-episode); frames before the first task message —
    or whole bags with none — fall back to the operator prompt recorded in the
    bag metadata. Raises RuntimeError if a warmup stream's topic is absent from
    the bag entirely (fail-fast: a wrong topic name in the contract would
    otherwise train on 100% fabricated zeros) or if warmup never completes
    (a present topic produced no samples).
    """
    fps = specs[0].fps
    step_ns = int(1e9 / fps)

    meta = _read_bag_metadata(bag_dir)
    info = meta.get(BAG_METADATA_KEY, {})
    storage_id = info.get("storage_identifier", "mcap")
    prompt = _read_prompt(meta)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=storage_id),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    topic_types = _get_topic_types(reader)
    layout = FrameLayout(specs)
    entries, routing = _build_buffers(specs, topic_types)

    # Fail fast on warmup (observation) topics absent from the bag: their
    # buffers could never fill, so every frame would silently carry fabricated
    # zeros for them. Non-warmup streams keep the warn + zero-fill behavior.
    absent = sorted(
        f"{spec.key} ({spec.source.channel.topic})"
        for spec, _ in entries
        if spec.key in warmup_keys and spec.source.channel.topic not in topic_types
    )
    if absent:
        raise RuntimeError(
            f"{bag_dir.name}: observation topics missing from bag: {absent}. "
            f"Check the contract topic names against the recording."
        )

    warmup_idxs = sorted({i for idxs in routing.values() for i in idxs if entries[i][0].key in warmup_keys})

    start_ns, end_ns = _get_bag_time_bounds_ns(reader)
    n_ticks = max(1, int((end_ns - start_ns) // step_ns) + 1)

    tick_idx = 0
    tick_ns = start_ns
    warmed_up = not warmup_idxs  # no routed warmup streams -> emit from tick 0
    skipped = 0
    emitted = 0
    # Same timestamp/decode/push policy as the live bridge (parity by
    # construction): decode failures drop the message with a warn-once.
    ingest = StreamIngest(warn=logging.warning, info=logging.info)

    # Per-frame task: topic value preferred, prompt fallback (hold semantics).
    task_topics = task_topics or {}
    current_task: str | None = None
    task_warned = False

    def _log_warmup() -> None:
        seconds = skipped * step_ns / 1e9
        delays = {entries[i][0].key: (entries[i][1].last_ts or start_ns) - start_ns for i in warmup_idxs}
        detail = ", ".join(f"{k}: +{ns / 1e9:.2f}s" for k, ns in sorted(delays.items()))
        if skipped > max(int(fps * WARMUP_WARN_SECONDS), int(n_ticks * WARMUP_WARN_FRACTION)):
            logging.warning(
                "%s: skipped %d warmup ticks (%.2fs) waiting for observation "
                "streams; stream delays (latest sample at warmup): %s",
                bag_dir.name,
                skipped,
                seconds,
                detail,
            )
        else:
            logging.info("%s: skipped %d warmup ticks (%.2fs)", bag_dir.name, skipped, seconds)

    def _try_emit() -> dict[str, Any] | None:
        """Emit the frame for the current tick, or None while warming up."""
        nonlocal warmed_up, skipped, emitted
        if not warmed_up:
            warmed_up = all(entries[i][1].last_ts is not None for i in warmup_idxs)
            if warmed_up and skipped:
                _log_warmup()
        if not warmed_up:
            skipped += 1
            return None
        frame = _sample_frame(tick_ns, entries, layout)
        frame["is_first"] = np.array([emitted == 0], dtype=bool)
        is_last = tick_idx == n_ticks - 1
        frame["is_last"] = np.array([is_last], dtype=bool)
        frame["is_terminal"] = np.array([is_last], dtype=bool)
        frame["task"] = prompt if current_task is None else current_task
        emitted += 1
        return frame

    while reader.has_next():
        topic, data, bag_ns = reader.read_next()

        # Emit ticks strictly before this message's receive time. The strict
        # comparison keeps a message received exactly at tick t in frame t
        # (it is pushed below, before the tick is emitted).
        while tick_idx < n_ticks and tick_ns < bag_ns:
            frame = _try_emit()
            if frame is not None:
                yield frame
            tick_idx += 1
            tick_ns = start_ns + tick_idx * step_ns

        if topic in task_topics:
            msg = deserialize_message(data, get_message(task_topics[topic]))
            text = getattr(msg, "data", None)
            if isinstance(text, str):
                current_task = text
            elif not task_warned:
                logging.warning(
                    "%s: task topic %s (%s) has no string 'data' field; ignoring",
                    bag_dir.name,
                    topic,
                    task_topics[topic],
                )
                task_warned = True

        # Route the message to every spec reading this topic (one deserialize,
        # one ingest per routed spec — selectors may differ).
        if topic in routing:
            msg = None
            for i in routing[topic]:
                spec, buffer = entries[i]
                if msg is None:
                    msg = deserialize_message(data, get_message(spec.source.channel.type))
                ingest.ingest(msg, spec, buffer, i, fallback_ns=bag_ns)

    # Emit remaining frames
    while tick_idx < n_ticks:
        frame = _try_emit()
        if frame is not None:
            yield frame
        tick_idx += 1
        tick_ns = start_ns + tick_idx * step_ns

    if emitted == 0:
        stale = [entries[i][0].key for i in warmup_idxs if entries[i][1].last_ts is None]
        raise RuntimeError(f"{bag_dir.name}: no frames emitted; observation streams never produced a sample: {stale}")


class BagFrameSource:
    """Iterate episodes (bags) under a directory, yielding per-episode frame streams.

    Optional sharding (num_shards/shard_index) selects a deterministic stride
    subset of bags for parallel or SLURM porting. Each episode is a
    (bag_dir, frame_iterator) pair. The caller drives a DatasetWriter per
    episode.
    """

    def __init__(
        self,
        raw_dir: Path,
        specs: list[StreamSpec],
        *,
        num_shards: int | None = None,
        shard_index: int | None = None,
        warmup_keys: set[str],
        task_topics: dict[str, str] | None = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.specs = list(specs)
        self.num_shards = num_shards
        self.shard_index = shard_index
        self.warmup_keys = warmup_keys
        self.task_topics = dict(task_topics or {})

    def bag_dirs(self) -> list[Path]:
        """Resolve the (optionally sharded) list of bag directories."""
        all_bag_dirs = find_bag_dirs(self.raw_dir)
        logging.info("Found %d bags in %s", len(all_bag_dirs), self.raw_dir)

        if self.num_shards is None:
            return all_bag_dirs

        if self.shard_index is None:
            raise ValueError("shard_index required when num_shards is specified")
        if self.shard_index >= self.num_shards:
            raise ValueError(f"shard_index ({self.shard_index}) >= num_shards ({self.num_shards})")

        shard = all_bag_dirs[self.shard_index :: self.num_shards]
        logging.info("Shard %d/%d: processing %d bags", self.shard_index, self.num_shards, len(shard))
        return shard

    def episodes(self) -> Iterator[tuple[Path, Iterator[dict[str, Any]]]]:
        """Yield (bag_dir, frame_iterator) for each selected bag."""
        for bag_dir in self.bag_dirs():
            yield (
                bag_dir,
                iter_bag_frames(bag_dir, self.specs, warmup_keys=self.warmup_keys, task_topics=self.task_topics),
            )
