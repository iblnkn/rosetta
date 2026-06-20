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
ROS2 bag reader: stream contract-decoded, resampled frames from a rosbag.

This is the ROS2 source adapter for the porting pipeline. It reads a bag,
decodes messages via the contract codecs, resamples with the same StreamBuffer
used at live inference, and yields plain frame dicts (numpy arrays + episode
boundary markers). It has NO lerobot dependency -- a dataset writer (e.g.
``rosetta.lerobot.dataset_writer``) consumes the frame stream.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
import yaml

# Register the ROS codecs (import side effect populates the registry).
import rosetta.ros2.decoders  # noqa: F401
import rosetta.ros2.encoders  # noqa: F401  # registers encoders (parity with recorder)
from rosetta.core.contract import ObservationStreamSpec, StreamSpec
from rosetta.core.contract_utils import StreamBuffer, zeros_for_spec
from rosetta.core.converters import decode_value, get_decoder_dtype
from rosetta.ros2.ros2_utils import get_message_timestamp_ns

# Bag metadata keys
BAG_METADATA_KEY = 'rosbag2_bagfile_information'
BAG_CUSTOM_DATA_KEY = 'custom_data'
BAG_PROMPT_KEY = 'lerobot.operator_prompt'

# Map LeRobot dtype strings to numpy dtypes
DTYPE_MAP = {
    'float32': np.float32,
    'float64': np.float64,
    'int32': np.int32,
    'int64': np.int64,
    'bool': bool,
}

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
        Topic-keyed dict: topic -> (spec, buffer), preserving insertion order.

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

    Specs sharing the same key are aggregated (concatenated in insertion order).
    """
    # Group by output key, preserving insertion order
    by_key: dict[str, list[tuple[StreamSpec, StreamBuffer]]] = {}
    for spec, buffer in buffers.values():
        by_key.setdefault(spec.key, []).append((spec, buffer))

    frame: dict[str, Any] = {}

    for key, items in by_key.items():
        first_spec = items[0][0]

        if isinstance(first_spec, ObservationStreamSpec) and first_spec.is_image:
            # Image: single value (no aggregation)
            spec, buffer = items[0]
            val = buffer.sample(tick_ns)
            if val is None:
                frame[key] = zeros_for_spec(spec)
            else:
                frame[key] = np.asarray(val, dtype=np.uint8)
        elif isinstance(first_spec, ObservationStreamSpec) and first_spec.dtype == 'string':
            # String: pass through
            spec, buffer = items[0]
            val = buffer.sample(tick_ns)
            frame[key] = str(val) if val is not None else ''
        elif isinstance(first_spec, ObservationStreamSpec) and first_spec.dtype in (
            'bool',
            'int32',
            'int64',
        ):
            # Scalar types: single value
            spec, buffer = items[0]
            val = buffer.sample(tick_ns)
            np_dtype = DTYPE_MAP[first_spec.dtype]  # already validated above
            if val is None:
                frame[key] = np.zeros(1, dtype=np_dtype)
            else:
                frame[key] = np.asarray(val, dtype=np_dtype).flatten()
        else:
            # Vector: concatenate all specs with this key
            # Determine dtype from spec or decoder registry
            if isinstance(first_spec, ObservationStreamSpec):
                dtype_str = first_spec.dtype
            else:
                # ActionStreamSpec: get dtype from decoder registry
                dtype_str = get_decoder_dtype(first_spec.msg_type)

            if dtype_str not in DTYPE_MAP:
                raise ValueError(
                    f"Unsupported dtype '{dtype_str}' for key '{key}'. Add to DTYPE_MAP."
                )
            np_dtype = DTYPE_MAP[dtype_str]

            values = []
            for spec, buffer in items:
                val = buffer.sample(tick_ns)
                if val is None:
                    val = np.zeros(max(len(spec.names), 1), dtype=np_dtype)
                else:
                    val = np.asarray(val, dtype=np_dtype).flatten()
                values.append(val)

            frame[key] = np.concatenate(values) if len(values) > 1 else values[0]

    return frame


def stream_frames_from_bag(bag_dir: Path, specs: list[StreamSpec]):
    """
    Stream frames from a bag file.

    Uses StreamBuffer for resampling (identical to live inference).
    Specs sharing the same key are aggregated into single tensors. Episode
    boundary markers (``is_first``/``is_last``/``is_terminal``) and ``task``
    are attached to each frame.
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

    while reader.has_next():
        topic, data, bag_ns = reader.read_next()

        # Emit frames whose tick time has passed
        while current_tick_idx < n_frames and bag_ns >= current_tick_ns:
            frame = _sample_frame(current_tick_ns, buffers)
            frame['is_first'] = np.array([current_tick_idx == 0], dtype=bool)
            frame['is_last'] = np.array([current_tick_idx == n_frames - 1], dtype=bool)
            frame['is_terminal'] = np.array([current_tick_idx == n_frames - 1], dtype=bool)
            frame['task'] = prompt

            yield frame

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
        frame = _sample_frame(current_tick_ns, buffers)
        frame['is_first'] = np.array([current_tick_idx == 0], dtype=bool)
        frame['is_last'] = np.array([current_tick_idx == n_frames - 1], dtype=bool)
        frame['is_terminal'] = np.array([current_tick_idx == n_frames - 1], dtype=bool)
        frame['task'] = prompt

        yield frame

        current_tick_idx += 1
        current_tick_ns = start_ns + current_tick_idx * step_ns
