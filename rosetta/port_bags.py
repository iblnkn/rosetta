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
ROS2 bag → LeRobot dataset porting script.

Converts rosbag recordings to LeRobot datasets using contract-driven decoding.
Uses the same decoders and resampling as live inference for consistency.

Usage:
    # Port all bags
    python -m rosetta.port_bags \\
        --raw-dir /path/to/bags \\
        --repo-id my_dataset \\
        --contract /path/to/contract.yaml

    # Port a single shard (for SLURM parallel processing)
    python -m rosetta.port_bags \\
        --raw-dir /path/to/bags \\
        --repo-id my_dataset \\
        --contract /path/to/contract.yaml \\
        --num-shards 100 \\
        --shard-index 0

    # Push to HuggingFace Hub
    python -m rosetta.port_bags \\
        --raw-dir /path/to/bags \\
        --repo-id my_org/my_dataset \\
        --contract /path/to/contract.yaml \\
        --push-to-hub
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import rosbag2_py
import yaml
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import get_elapsed_time_in_days_hours_minutes_seconds

from .common.converters import DTYPES, decode_value, get_decoder_dtype
from .common.contract import ObservationStreamSpec, StreamSpec, load_contract
from .common.contract_utils import (
    build_feature,
    get_namespaced_names,
    iter_specs,
    StreamBuffer,
    zeros_for_spec,
)
from .common.ros2_utils import get_message_timestamp_ns, stamp_from_header_ns

# Bag metadata keys
BAG_METADATA_KEY = "rosbag2_bagfile_information"
BAG_CUSTOM_DATA_KEY = "custom_data"
BAG_PROMPT_KEY = "lerobot.operator_prompt"
# Import decoders/encoders to register them
from .common import decoders as _decoders  # noqa: F401
from .common import encoders as _encoders  # noqa: F401


# ---------- Bag discovery ----------


def find_bag_dirs(raw_dir: Path) -> list[Path]:
    """Find all bag directories (identified by metadata.yaml)."""
    bag_dirs = sorted(
        p.parent for p in raw_dir.rglob("metadata.yaml")
        if (p.parent / "metadata.yaml").exists()
    )
    if not bag_dirs:
        raise RuntimeError(f"No bag directories found in {raw_dir}")
    return bag_dirs


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


def _get_topic_types(reader: rosbag2_py.SequentialReader) -> dict[str, str]:
    """Get topic -> type mapping from bag."""
    return {t.name: t.type for t in reader.get_all_topics_and_types()}


def _build_buffers(
    specs: list[StreamSpec],
    topic_types: dict[str, str],
) -> dict[str, tuple[StreamSpec, StreamBuffer]]:
    """Create StreamBuffers keyed by topic.

    Returns:
        Topic-keyed dict: topic -> (spec, buffer), preserving insertion order.
    """
    buffers: dict[str, tuple[StreamSpec, StreamBuffer]] = {}

    for spec in specs:
        if spec.topic not in topic_types:
            logging.warning("Topic %s not in bag, skipping %s", spec.topic, spec.key)
            continue

        if isinstance(spec, ObservationStreamSpec):
            buffer = StreamBuffer.from_spec(spec)
        else:
            step_ns = int(1e9 / spec.fps) if spec.fps > 0 else int(1e9 / 30)
            buffer = StreamBuffer(policy="hold", step_ns=step_ns, tol_ns=0)

        buffers[spec.topic] = (spec, buffer)

    if not buffers:
        raise RuntimeError("No contract topics found in bag")

    return buffers


def _build_features(specs: list[StreamSpec]) -> dict[str, dict[str, Any]]:
    """Build LeRobot feature definitions from contract specs.

    Specs sharing the same key are aggregated (names concatenated for vectors).
    """
    # Group specs by output key
    by_key: dict[str, list[StreamSpec]] = {}
    for spec in specs:
        by_key.setdefault(spec.key, []).append(spec)

    features = {}
    for key, key_specs in by_key.items():
        first = key_specs[0]
        dtype = DTYPES[first.msg_type]

        if dtype in ("video", "image"):
            # Images: no aggregation
            features[key] = build_feature(first)
        elif dtype == "string":
            # Strings: no aggregation
            features[key] = build_feature(first)
        else:
            # Numeric: aggregate names from all specs
            all_names = []
            for spec in key_specs:
                all_names.extend(get_namespaced_names(spec))
            n = len(all_names) or 1
            features[key] = {
                "dtype": dtype,
                "shape": (n,),
                "names": {"axes": all_names} if all_names else None,
            }

    # Frame boundary markers
    features["is_first"] = {"dtype": "bool", "shape": (1,), "names": None}
    features["is_last"] = {"dtype": "bool", "shape": (1,), "names": None}
    features["is_terminal"] = {"dtype": "bool", "shape": (1,), "names": None}

    return features


def _get_bag_time_bounds_ns(reader: rosbag2_py.SequentialReader) -> tuple[int, int]:
    """Get time bounds from bag metadata."""
    metadata = reader.get_metadata()
    start_time = metadata.starting_time
    duration = metadata.duration
    # rosbag2_py returns Time/Duration objects with .nanoseconds property
    start_ns = start_time.nanoseconds
    duration_ns = duration.nanoseconds
    return start_ns, start_ns + duration_ns


# Map LeRobot dtype strings to numpy dtypes
DTYPE_MAP = {
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "int64": np.int64,
    "bool": bool,
}


def _sample_frame(
    tick_ns: int,
    buffers: dict[str, tuple[StreamSpec, StreamBuffer]],
) -> dict[str, Any]:
    """Sample a single frame from buffers at the given tick time.

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
        elif isinstance(first_spec, ObservationStreamSpec) and first_spec.dtype == "string":
            # String: pass through
            spec, buffer = items[0]
            val = buffer.sample(tick_ns)
            frame[key] = str(val) if val is not None else ""
        elif isinstance(first_spec, ObservationStreamSpec) and first_spec.dtype in ("bool", "int32", "int64"):
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
                raise ValueError(f"Unsupported dtype '{dtype_str}' for key '{key}'. Add to DTYPE_MAP.")
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


def _stream_frames_from_bag(bag_dir: Path, specs: list[StreamSpec]):
    """Stream LeRobot frames from a bag file.

    Uses StreamBuffer for resampling (identical to live inference).
    Specs sharing the same key are aggregated into single tensors.
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
            frame["is_first"] = np.array([current_tick_idx == 0], dtype=bool)
            frame["is_last"] = np.array([current_tick_idx == n_frames - 1], dtype=bool)
            frame["is_terminal"] = np.array([current_tick_idx == n_frames - 1], dtype=bool)
            frame["task"] = prompt

            yield frame

            current_tick_idx += 1
            current_tick_ns = start_ns + current_tick_idx * step_ns

        # Push message to buffer
        if topic in buffers:
            spec, buffer = buffers[topic]
            msg = deserialize_message(data, get_message(spec.msg_type))

            if (
                spec.stamp_src == "header"
                and spec.key not in header_warned
                and stamp_from_header_ns(msg) is None
            ):
                logging.warning(
                    "Header stamp unavailable for '%s' in %s, using bag receive time",
                    spec.key, bag_dir.name,
                )
                header_warned.add(spec.key)

            ts = get_message_timestamp_ns(msg, spec, bag_ns)
            val = decode_value(msg, spec)
            if val is not None:
                buffer.push(ts, val)

    # Emit remaining frames
    while current_tick_idx < n_frames:
        frame = _sample_frame(current_tick_ns, buffers)
        frame["is_first"] = np.array([current_tick_idx == 0], dtype=bool)
        frame["is_last"] = np.array([current_tick_idx == n_frames - 1], dtype=bool)
        frame["is_terminal"] = np.array([current_tick_idx == n_frames - 1], dtype=bool)
        frame["task"] = prompt

        yield frame

        current_tick_idx += 1
        current_tick_ns = start_ns + current_tick_idx * step_ns


# ---------- Main porting function ----------


def port_bags(
    raw_dir: Path,
    repo_id: str,
    contract_path: Path,
    root: Path | None = None,
    push_to_hub: bool = False,
    num_shards: int | None = None,
    shard_index: int | None = None,
    vcodec: str = "libsvtav1",
):
    """
    Port ROS2 bags to LeRobot dataset format.

    Args:
        raw_dir: Directory containing bag subdirectories.
        repo_id: HuggingFace repository ID (e.g., "my_org/my_dataset").
        contract_path: Path to Rosetta contract YAML.
        root: Output directory for dataset. Defaults to ~/.cache/huggingface/lerobot.
        push_to_hub: Whether to upload to HuggingFace Hub after porting.
        num_shards: Total number of shards for parallel processing.
        shard_index: Index of this shard (0 to num_shards-1).
        vcodec: Video codec for encoding. Options: 'libsvtav1' (default, good compression),
            'libx264'/'h264' (fast), 'hevc', 'h264_nvenc' (GPU).
    """
    contract = load_contract(contract_path)
    specs = list(iter_specs(contract))
    features = _build_features(specs)

    all_bag_dirs = find_bag_dirs(raw_dir)
    total_bags = len(all_bag_dirs)
    logging.info("Found %d bags in %s", total_bags, raw_dir)

    # Select shard subset if sharding
    if num_shards is not None:
        if shard_index is None:
            raise ValueError("shard_index required when num_shards is specified")
        if shard_index >= num_shards:
            raise ValueError(f"shard_index ({shard_index}) >= num_shards ({num_shards})")

        bag_dirs = all_bag_dirs[shard_index::num_shards]
        logging.info("Shard %d/%d: processing %d bags", shard_index, num_shards, len(bag_dirs))
    else:
        bag_dirs = all_bag_dirs

    if not bag_dirs:
        logging.warning("No bags to process in this shard")
        return

    # LeRobot uses root directly as dataset path, so append repo_id
    dataset_root = root / repo_id if root else None
    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_root,
        robot_type=contract.robot_type,
        fps=contract.fps,
        features=features,
        vcodec=vcodec,
    )

    start_time = time.time()
    num_episodes = len(bag_dirs)
    successful = 0
    failed: list[tuple[Path, str]] = []

    for episode_index, bag_dir in enumerate(bag_dirs):
        elapsed_time = time.time() - start_time
        d, h, m, s = get_elapsed_time_in_days_hours_minutes_seconds(elapsed_time)

        logging.info(
            f"{episode_index} / {num_episodes} episodes processed "
            f"(after {d} days, {h} hours, {m} minutes, {s:.3f} seconds)"
        )

        try:
            frame_count = 0
            for frame in _stream_frames_from_bag(bag_dir, specs):
                lerobot_dataset.add_frame(frame)
                frame_count += 1

            lerobot_dataset.save_episode()
            successful += 1
            logging.info("  -> %d frames from %s", frame_count, bag_dir.name)

        except Exception as e:
            failed.append((bag_dir, str(e)))
            logging.error("  -> FAILED %s: %s", bag_dir.name, e)
            continue

    elapsed_time = time.time() - start_time
    d, h, m, s = get_elapsed_time_in_days_hours_minutes_seconds(elapsed_time)
    logging.info(
        f"\nCompleted: {successful}/{num_episodes} episodes "
        f"({len(failed)} failed) in {d}d {h}h {m}m {s:.1f}s"
    )

    if failed:
        logging.warning("Failed bags:")
        for bag_dir, error in failed:
            logging.warning("  - %s: %s", bag_dir.name, error)

    if successful == 0:
        raise RuntimeError(f"All {num_episodes} bags failed to convert")

    lerobot_dataset.finalize()

    if push_to_hub:
        lerobot_dataset.push_to_hub(
            tags=["rosetta", "rosbag"],
            private=False,
        )


# ---------- CLI ----------


def main():
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Port ROS2 bags to LeRobot dataset"
    )

    parser.add_argument(
        "--raw-dir", type=Path, required=True,
        help="Directory containing bag subdirectories"
    )
    parser.add_argument(
        "--repo-id", type=str, default=None,
        help="HuggingFace repository ID (e.g., my_org/my_dataset). Defaults to raw-dir name."
    )
    parser.add_argument(
        "--contract", type=Path, required=True,
        help="Rosetta contract YAML path"
    )
    parser.add_argument(
        "--root", type=Path, default=None,
        help="Parent directory for datasets. Dataset saved to root/repo-id. (default: ~/.cache/huggingface/lerobot)"
    )
    parser.add_argument(
        "--push-to-hub", action="store_true",
        help="Upload to HuggingFace Hub after porting"
    )
    parser.add_argument(
        "--num-shards", type=int, default=None,
        help="Total number of shards for parallel processing"
    )
    parser.add_argument(
        "--shard-index", type=int, default=None,
        help="Index of this shard (0 to num-shards-1)"
    )
    parser.add_argument(
        "--vcodec", type=str, default="libsvtav1",
        choices=["libsvtav1", "libx264", "h264", "hevc", "h264_nvenc"],
        help="Video codec for encoding (default: libsvtav1). Use libx264/h264 for faster encoding."
    )

    args = parser.parse_args()

    repo_id = args.repo_id or args.raw_dir.name

    try:
        port_bags(
            raw_dir=args.raw_dir,
            repo_id=repo_id,
            contract_path=args.contract,
            root=args.root,
            push_to_hub=args.push_to_hub,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            vcodec=args.vcodec,
        )
    except KeyboardInterrupt:
        logging.info("\nInterrupted by user")
    except Exception as e:
        logging.error("Error: %s", e)
        raise


if __name__ == "__main__":
    main()
