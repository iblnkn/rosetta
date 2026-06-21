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
rosetta.core.contract_utils.aggregate_frame.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
import yaml

from rosetta.core.contract import ObservationStreamSpec, StreamSpec
from rosetta.core.contract_utils import aggregate_frame, StreamBuffer
from rosetta.core.converters import decode_value
from rosetta.ros2.ros2_utils import get_message_timestamp_ns

# Register the ROS codecs (import side effect populates the core registry).
from rosetta.ros2 import decoders as _decoders  # noqa: F401
from rosetta.ros2 import encoders as _encoders  # noqa: F401

del _decoders, _encoders

# Bag metadata keys
BAG_METADATA_KEY = 'rosbag2_bagfile_information'
BAG_CUSTOM_DATA_KEY = 'custom_data'
BAG_PROMPT_KEY = 'lerobot.operator_prompt'


# ---------- Bag discovery ----------


def find_bag_dirs(raw_dir: Path) -> list[Path]:
    """Find all bag directories (identified by metadata.yaml)."""
    bag_dirs = sorted(
        p.parent for p in raw_dir.rglob('metadata.yaml') if (p.parent / 'metadata.yaml').exists()
    )
    if not bag_dirs:
        raise RuntimeError(f'No bag directories found in {raw_dir}')
    return bag_dirs


# ---------- Internal helpers ----------


def _read_bag_metadata(bag_dir: Path) -> dict[str, Any]:
    """Read bag metadata.yaml."""
    meta_path = bag_dir / 'metadata.yaml'
    if not meta_path.exists():
        return {}
    with meta_path.open() as f:
        return yaml.safe_load(f) or {}


def _read_prompt(meta: dict[str, Any]) -> str:
    """Read prompt from metadata custom_data."""
    info = meta.get(BAG_METADATA_KEY, {})
    custom_data = info.get(BAG_CUSTOM_DATA_KEY, {})
    if isinstance(custom_data, dict):
        return custom_data.get(BAG_PROMPT_KEY, '')
    return ''


def _get_topic_types(reader: rosbag2_py.SequentialReader) -> dict[str, str]:
    """Get topic -> type mapping from bag."""
    return {t.name: t.type for t in reader.get_all_topics_and_types()}


def _build_buffers(
    specs: list[StreamSpec],
    topic_types: dict[str, str],
) -> dict[str, tuple[StreamSpec, StreamBuffer]]:
    """
    Create StreamBuffers keyed by topic.

    Returns
    -------
        Dict of topic -> (spec, buffer), preserving insertion order.

    """
    buffers: dict[str, tuple[StreamSpec, StreamBuffer]] = {}

    for spec in specs:
        if spec.topic not in topic_types:
            logging.warning('Topic %s not in bag, skipping %s', spec.topic, spec.key)
            continue

        if isinstance(spec, ObservationStreamSpec):
            buffer = StreamBuffer.from_spec(spec)
        else:
            step_ns = int(1e9 / spec.fps) if spec.fps > 0 else int(1e9 / 30)
            buffer = StreamBuffer(policy='hold', step_ns=step_ns, tol_ns=0)

        buffers[spec.topic] = (spec, buffer)

    if not buffers:
        raise RuntimeError('No contract topics found in bag')

    return buffers


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
    buffers: dict[str, tuple[StreamSpec, StreamBuffer]],
) -> dict[str, Any]:
    """
    Sample a single frame from buffers at the given tick time.

    Samples every buffer and passes the values to aggregate_frame, which does
    the by-key aggregation shared with the live bridge.
    """
    samples = [(spec, buffer.sample(tick_ns)) for spec, buffer in buffers.values()]
    return aggregate_frame(samples)


def iter_bag_frames(bag_dir: Path, specs: list[StreamSpec]) -> Iterator[dict[str, Any]]:
    """
    Yield resampled frame dicts from a single bag.

    Uses StreamBuffer for resampling, identical to live inference. Each frame
    carries the contract data keys plus is_first/is_last/is_terminal boundary
    markers and the episode task string.
    """
    fps = specs[0].fps
    step_ns = int(1e9 / fps)

    meta = _read_bag_metadata(bag_dir)
    info = meta.get(BAG_METADATA_KEY, {})
    storage_id = info.get('storage_identifier', 'mcap')
    prompt = _read_prompt(meta)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=storage_id),
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr',
        ),
    )

    topic_types = _get_topic_types(reader)
    buffers = _build_buffers(specs, topic_types)

    start_ns, end_ns = _get_bag_time_bounds_ns(reader)
    n_frames = max(1, int((end_ns - start_ns) // step_ns) + 1)

    current_tick_idx = 0
    current_tick_ns = start_ns
    header_warned: set[str] = set()

    def _emit(tick_idx: int, tick_ns: int) -> dict[str, Any]:
        frame = _sample_frame(tick_ns, buffers)
        frame['is_first'] = np.array([tick_idx == 0], dtype=bool)
        frame['is_last'] = np.array([tick_idx == n_frames - 1], dtype=bool)
        frame['is_terminal'] = np.array([tick_idx == n_frames - 1], dtype=bool)
        frame['task'] = prompt
        return frame

    while reader.has_next():
        topic, data, bag_ns = reader.read_next()

        # Emit frames whose tick time has passed
        while current_tick_idx < n_frames and bag_ns >= current_tick_ns:
            yield _emit(current_tick_idx, current_tick_ns)
            current_tick_idx += 1
            current_tick_ns = start_ns + current_tick_idx * step_ns

        # Push message to buffer
        if topic in buffers:
            spec, buffer = buffers[topic]
            msg = deserialize_message(data, get_message(spec.msg_type))

            ts, used_fallback = get_message_timestamp_ns(msg, spec, bag_ns)
            if spec.stamp_src == 'header' and used_fallback and spec.key not in header_warned:
                logging.warning(
                    "Header stamp unavailable for '%s' in %s, using bag receive time",
                    spec.key,
                    bag_dir.name,
                )
                header_warned.add(spec.key)
            val = decode_value(msg, spec)
            if val is not None:
                buffer.push(ts, val)

    # Emit remaining frames
    while current_tick_idx < n_frames:
        yield _emit(current_tick_idx, current_tick_ns)
        current_tick_idx += 1
        current_tick_ns = start_ns + current_tick_idx * step_ns


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
    ):
        self.raw_dir = Path(raw_dir)
        self.specs = list(specs)
        self.num_shards = num_shards
        self.shard_index = shard_index

    def bag_dirs(self) -> list[Path]:
        """Resolve the (optionally sharded) list of bag directories."""
        all_bag_dirs = find_bag_dirs(self.raw_dir)
        logging.info('Found %d bags in %s', len(all_bag_dirs), self.raw_dir)

        if self.num_shards is None:
            return all_bag_dirs

        if self.shard_index is None:
            raise ValueError('shard_index required when num_shards is specified')
        if self.shard_index >= self.num_shards:
            raise ValueError(
                f'shard_index ({self.shard_index}) >= num_shards ({self.num_shards})'
            )

        shard = all_bag_dirs[self.shard_index :: self.num_shards]
        logging.info(
            'Shard %d/%d: processing %d bags', self.shard_index, self.num_shards, len(shard)
        )
        return shard

    def episodes(self) -> Iterator[tuple[Path, Iterator[dict[str, Any]]]]:
        """Yield (bag_dir, frame_iterator) for each selected bag."""
        for bag_dir in self.bag_dirs():
            yield bag_dir, iter_bag_frames(bag_dir, self.specs)
