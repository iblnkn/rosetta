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

    # Unified NAS convention (no --root/--repo-id): saves to
    #   /mnt/nas/dataset/robot_learning/lerobot/<date>/<time>_dataset_<robot>_<task>_<user>
    python -m rosetta.port_bags \\
        --raw-dir /path/to/bags \\
        --contract /path/to/contract.yaml \\
        --task-config configs/task/tossing.yaml \\
        --user-name alice
"""

from __future__ import annotations

import argparse
import getpass
import logging
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rosbag2_py
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import get_elapsed_time_in_days_hours_minutes_seconds
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from .common.contract import (ROLE_RECORD, ActionStreamSpec,
                              ObservationStreamSpec, StreamSpec,
                              is_unified_contract, load_contract,
                              load_processor_spec, load_unified_contract)
from .common.contract_utils import (StreamBuffer, build_features,
                                    get_namespaced_names, iter_specs,
                                    zeros_for_spec)
from .common.contract_validation import check_processor_vs_contract
from .common.converters import decode_value, get_decoder_dtype
from .common.decoders import _nearest_resize
from .common.ros2_utils import get_message_timestamp_ns

# Type alias for processors (optional import)
try:
    from lerobot.processor import RobotProcessorPipeline
    from lerobot.processor.converters import (observation_to_transition,
                                              robot_action_to_transition,
                                              transition_to_observation,
                                              transition_to_robot_action)

    PROCESSORS_AVAILABLE = True
except ImportError:
    PROCESSORS_AVAILABLE = False
    RobotProcessorPipeline = None

# Bag metadata keys
BAG_METADATA_KEY = "rosbag2_bagfile_information"
BAG_CUSTOM_DATA_KEY = "custom_data"
BAG_PROMPT_KEY = "lerobot.operator_prompt"

# Shared NAS location used when neither --root nor --repo-id is given. The dataset
# is saved under DATASET_NAS_ROOT/<date>/<time>_dataset_<robot_type>_<task>_<user>.
DATASET_NAS_ROOT = Path("/mnt/nas/dataset/robot_learning/lerobot")
# Timezone used for the date/time stamp in dataset folder names (UTC+8), so the
# naming is stable regardless of the host's local timezone.
DATASET_TZ = timezone(timedelta(hours=8))
# Import decoders/encoders/processors to register them
from .common import decoders as _decoders  # noqa: F401
from .common import encoders as _encoders  # noqa: F401
from .common import processors as _processors  # noqa: F401

# ---------- Bag discovery ----------


def find_bag_dirs(raw_dir: Path) -> list[Path]:
    """Find all bag directories (identified by metadata.yaml)."""
    bag_dirs = sorted(
        p.parent
        for p in raw_dir.rglob("metadata.yaml")
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


def _read_task_name(task_config_path: Path) -> str:
    """Read the `task_name` field from a task config YAML (e.g. configs/task/*.yaml)."""
    with task_config_path.open() as f:
        cfg = yaml.safe_load(f) or {}
    task_name = cfg.get("task_name")
    if not task_name:
        raise ValueError(
            f"No 'task_name' field found in task config {task_config_path}"
        )
    return str(task_name)


def _sanitize_name_part(value: str) -> str:
    """Make a string safe for use inside a dataset directory name.

    Underscores are replaced with dashes so they don't collide with the
    underscore separators in the dataset name (e.g. "so_101" -> "so-101").
    """
    return str(value).strip().replace("/", "-").replace(" ", "-").replace("_", "-")


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

    Delegates the contract -> feature mapping to
    ``contract_utils.build_features`` (the single source shared with the
    validators), then appends the dataset-writer frame-boundary markers.
    """
    features = build_features(specs)

    # Frame boundary markers (dataset-writer concern, not contract-derived)
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


def _sample_robot_dicts(
    tick_ns: int,
    buffers: dict[str, tuple[StreamSpec, StreamBuffer]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Sample buffers into robot-style observation and action dicts.

    Returns:
        (robot_observation, robot_action) dicts with individual keys.
        Keys are namespaced selector names (e.g., 'left_arm.position.joint_1').
        Images use short keys (e.g., 'front' not 'observation.images.front').
    """
    robot_obs: dict[str, Any] = {}
    robot_action: dict[str, Any] = {}

    for topic, (spec, buffer) in buffers.items():
        val = buffer.sample(tick_ns)

        if isinstance(spec, ObservationStreamSpec):
            if spec.is_image:
                # Image: use short key
                key = spec.key.removeprefix("observation.images.")
                robot_obs[key] = val if val is not None else zeros_for_spec(spec)
            elif spec.dtype == "string":
                key = spec.key.removeprefix("observation.")
                robot_obs[key] = str(val) if val is not None else ""
            else:
                # Numeric observation - individual keys
                names = get_namespaced_names(spec)
                if val is None:
                    val = np.zeros(len(names) if names else 1)
                val = np.asarray(val).flatten()
                for i, name in enumerate(names):
                    robot_obs[name] = float(val[i]) if i < len(val) else 0.0
        elif isinstance(spec, ActionStreamSpec):
            # Action spec - individual keys
            names = get_namespaced_names(spec)
            if val is None:
                val = np.zeros(len(names) if names else 1)
            val = np.asarray(val).flatten()
            for i, name in enumerate(names):
                robot_action[name] = float(val[i]) if i < len(val) else 0.0

    return robot_obs, robot_action


def _robot_dicts_to_frame(
    robot_obs: dict[str, Any],
    robot_action: dict[str, Any],
    specs: list[StreamSpec],
) -> dict[str, Any]:
    """Convert robot-style dicts to LeRobot frame format.

    Aggregates individual values into concatenated arrays matching feature definitions.

    Note: The function will only work correctly if the processor preserves the key names
    and data types.
    """
    frame: dict[str, Any] = {}

    # Separate specs by type
    obs_specs = [s for s in specs if isinstance(s, ObservationStreamSpec)]
    action_specs = [s for s in specs if isinstance(s, ActionStreamSpec)]

    # Group observation specs by key for aggregation
    obs_by_key: dict[str, list[ObservationStreamSpec]] = {}
    for spec in obs_specs:
        obs_by_key.setdefault(spec.key, []).append(spec)

    # Build observation features
    for key, key_specs in obs_by_key.items():
        first_spec = key_specs[0]

        if first_spec.is_image:
            # Image: single value
            short_key = key.removeprefix("observation.images.")
            img = robot_obs.get(short_key)
            if img is not None:
                frame[key] = np.asarray(img, dtype=np.uint8)
            else:
                frame[key] = zeros_for_spec(first_spec)
        elif first_spec.dtype == "string":
            short_key = key.removeprefix("observation.")
            frame[key] = robot_obs.get(short_key, "")
        elif first_spec.dtype in ("bool", "int32", "int64"):
            # Scalar types
            names = get_namespaced_names(first_spec)
            np_dtype = DTYPE_MAP[first_spec.dtype]
            if names:
                val = robot_obs.get(names[0], 0)
                frame[key] = np.array([val], dtype=np_dtype)
            else:
                frame[key] = np.zeros(1, dtype=np_dtype)
        else:
            # Numeric vector: concatenate all specs with this key
            dtype_str = first_spec.dtype
            np_dtype = DTYPE_MAP.get(dtype_str, np.float32)
            values = []
            for spec in key_specs:
                for name in get_namespaced_names(spec):
                    values.append(robot_obs.get(name, 0.0))
            if values:
                frame[key] = np.array(values, dtype=np_dtype)
            else:
                frame[key] = np.zeros(1, dtype=np_dtype)

    # Group action specs by key for aggregation
    action_by_key: dict[str, list[ActionStreamSpec]] = {}
    for spec in action_specs:
        action_by_key.setdefault(spec.key, []).append(spec)

    # Build action features
    for key, key_specs in action_by_key.items():
        first_spec = key_specs[0]
        dtype_str = get_decoder_dtype(first_spec.msg_type)
        np_dtype = DTYPE_MAP.get(dtype_str, np.float32)
        values = []
        for spec in key_specs:
            for name in get_namespaced_names(spec):
                values.append(robot_action.get(name, 0.0))
        frame[key] = (
            np.array(values, dtype=np_dtype) if values else np.zeros(1, dtype=np_dtype)
        )

    return frame


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
        elif (
            isinstance(first_spec, ObservationStreamSpec)
            and first_spec.dtype == "string"
        ):
            # String: pass through
            spec, buffer = items[0]
            val = buffer.sample(tick_ns)
            frame[key] = str(val) if val is not None else ""
        elif isinstance(first_spec, ObservationStreamSpec) and first_spec.dtype in (
            "bool",
            "int32",
            "int64",
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


def _conform_images_to_specs(
    frame: dict[str, Any],
    image_specs: list[ObservationStreamSpec],
) -> None:
    """Resize a frame's image entries to the contract's declared shape, in place.

    Decoders return native camera resolution; the dataset image feature is
    declared from ``spec.image_shape`` (see ``build_feature``), and
    ``zeros_for_spec`` placeholders are emitted at that shape. On the
    no-observation-processor path nothing else resizes the frame, so without
    this step native-resolution frames would mismatch the declared feature and
    ``add_frame`` would reject them. Nearest-neighbor matches the historical
    decode-time resize, so datasets ported before the decoder change stay
    consistent. Already-correct frames (incl. ``zeros_for_spec``) short-circuit
    in ``_nearest_resize``.
    """
    for spec in image_specs:
        img = frame.get(spec.key)
        if img is None or spec.image_shape is None:
            continue
        h, w = spec.image_shape
        frame[spec.key] = _nearest_resize(
            np.asarray(img, dtype=np.uint8), int(h), int(w)
        )


def _stream_frames_from_bag(
    bag_dir: Path,
    specs: list[StreamSpec],
    obs_processor=None,
    action_processor=None,
):
    """Stream LeRobot frames from a bag file.

    Uses StreamBuffer for resampling (identical to live inference).
    Specs sharing the same key are aggregated into single tensors.

    Args:
        bag_dir: Path to bag directory.
        specs: List of StreamSpecs from contract.
        obs_processor: Optional RobotObservationProcessor to apply to observations.
        action_processor: Optional RobotActionProcessor to apply to actions.
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

    # Without an observation processor to crop/resize, image frames come out at
    # native camera resolution and must be conformed to the contract's declared
    # feature shape before add_frame (decoders no longer resize).
    image_specs = [
        s for s in specs if isinstance(s, ObservationStreamSpec) and s.is_image
    ]

    # Filter reader to only yield messages for contract topics, skipping all others
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(buffers.keys())))

    start_ns, end_ns = _get_bag_time_bounds_ns(reader)
    n_frames = max(1, int((end_ns - start_ns) // step_ns) + 1)

    current_tick_idx = 0
    current_tick_ns = start_ns
    # frames_emitted tracks how many frames have actually been yielded.
    # is_first is set on the very first emitted frame (when all buffers are populated),
    # not necessarily on tick index 0, to handle per-topic startup latency.
    frames_emitted = 0
    header_warned: set[str] = set()

    def _all_buffers_ready() -> bool:
        """Return True once every topic buffer has received at least one message."""
        return all(buf.last_ts is not None for _, buf in buffers.values())

    while reader.has_next():
        topic, data, bag_ns = reader.read_next()

        # Emit frames whose tick time has passed
        while current_tick_idx < n_frames and bag_ns >= current_tick_ns:
            # Hold back emission until all buffers have at least one sample
            if not _all_buffers_ready():
                current_tick_idx += 1
                current_tick_ns = start_ns + current_tick_idx * step_ns
                continue

            # Use processor path if processors provided, otherwise use direct sampling
            if obs_processor is not None or action_processor is not None:
                # Sample into robot-style dicts
                robot_obs, robot_action = _sample_robot_dicts(current_tick_ns, buffers)

                # Apply processors
                if obs_processor is not None:
                    robot_obs = obs_processor(robot_obs)
                if action_processor is not None:
                    robot_action = action_processor(robot_action)

                # Convert back to frame format
                frame = _robot_dicts_to_frame(robot_obs, robot_action, specs)
            else:
                # Direct sampling (original path, no processor overhead)
                frame = _sample_frame(current_tick_ns, buffers)

            if obs_processor is None:
                _conform_images_to_specs(frame, image_specs)

            frame["is_first"] = np.array([frames_emitted == 0], dtype=bool)
            frame["is_last"] = np.array([current_tick_idx == n_frames - 1], dtype=bool)
            frame["is_terminal"] = np.array(
                [current_tick_idx == n_frames - 1], dtype=bool
            )
            frame["task"] = prompt

            yield frame
            frames_emitted += 1

            current_tick_idx += 1
            current_tick_ns = start_ns + current_tick_idx * step_ns

        # Push message to buffer
        if topic in buffers:
            spec, buffer = buffers[topic]
            msg = deserialize_message(data, get_message(spec.msg_type))

            ts, used_fallback = get_message_timestamp_ns(msg, spec, bag_ns)
            if (
                spec.stamp_src == "header"
                and used_fallback
                and spec.key not in header_warned
            ):
                logging.warning(
                    "Header stamp unavailable for '%s' in %s, using bag receive time",
                    spec.key,
                    bag_dir.name,
                )
                header_warned.add(spec.key)
            val = decode_value(msg, spec)
            if val is not None:
                buffer.push(ts, val)

    # Emit remaining frames after all bag messages are consumed
    while current_tick_idx < n_frames:
        # Use processor path if processors provided
        if obs_processor is not None or action_processor is not None:
            robot_obs, robot_action = _sample_robot_dicts(current_tick_ns, buffers)
            if obs_processor is not None:
                robot_obs = obs_processor(robot_obs)
            if action_processor is not None:
                robot_action = action_processor(robot_action)
            frame = _robot_dicts_to_frame(robot_obs, robot_action, specs)
        else:
            frame = _sample_frame(current_tick_ns, buffers)

        if obs_processor is None:
            _conform_images_to_specs(frame, image_specs)

        frame["is_first"] = np.array([frames_emitted == 0], dtype=bool)
        frame["is_last"] = np.array([current_tick_idx == n_frames - 1], dtype=bool)
        frame["is_terminal"] = np.array([current_tick_idx == n_frames - 1], dtype=bool)
        frame["task"] = prompt

        yield frame
        frames_emitted += 1

        current_tick_idx += 1
        current_tick_ns = start_ns + current_tick_idx * step_ns


# ---------- Main porting function ----------


def port_bags(
    raw_dir: Path,
    repo_id: str | None,
    contract_path: Path,
    root: Path | None = None,
    push_to_hub: bool = False,
    num_shards: int | None = None,
    shard_index: int | None = None,
    vcodec: str = "libsvtav1",
    observation_processor_path: Path | None = None,
    action_processor_path: Path | None = None,
    task: str | None = None,
    user_name: str | None = None,
):
    """
    Port ROS2 bags to LeRobot dataset format.

    Args:
        raw_dir: Directory containing bag subdirectories.
        repo_id: HuggingFace repository ID (e.g., "my_org/my_dataset"). If both
            this and ``root`` are None, the dataset is saved under the shared NAS
            location using the unified naming convention (see ``DATASET_NAS_ROOT``).
        contract_path: Path to Rosetta contract YAML. A copy is saved inside the
            output dataset directory for provenance.
        root: Output directory for dataset. Defaults to ~/.cache/huggingface/lerobot.
        push_to_hub: Whether to upload to HuggingFace Hub after porting.
        num_shards: Total number of shards for parallel processing.
        shard_index: Index of this shard (0 to num_shards-1).
        vcodec: Video codec for encoding. Options: 'libsvtav1' (default, good compression),
            'libx264'/'h264' (fast), 'hevc', 'h264_nvenc' (GPU).
        observation_processor_path: Path to saved RobotObservationProcessor config directory.
        action_processor_path: Path to saved RobotActionProcessor config directory.
        task: Task name used in the unified naming convention (typically read from a
            task config YAML). Falls back to "unknown" if not provided.
        user_name: User name used in the unified naming convention. Falls back to
            "unknown" if not provided.
    """
    # Unified contracts carry record + inference + processor in one file; port
    # bags with the record-role view. Legacy single-role contracts still work.
    if is_unified_contract(contract_path):
        contract = load_unified_contract(contract_path, ROLE_RECORD)
    else:
        contract = load_contract(contract_path)
    specs = list(iter_specs(contract))
    features = _build_features(specs)

    # Load processors if paths provided
    obs_processor = None
    action_processor = None

    if observation_processor_path or action_processor_path:
        if not PROCESSORS_AVAILABLE:
            raise ImportError(
                "LeRobot processor module not available. "
                "Install lerobot with processor support to use "
                "--observation-processor or --action-processor."
            )

    if observation_processor_path:
        logging.info(
            "Loading observation processor from %s", observation_processor_path
        )
        obs_processor = RobotProcessorPipeline.from_pretrained(
            observation_processor_path,
            config_filename="robot_observation_processor.json",
            to_transition=observation_to_transition,
            to_output=transition_to_observation,
        )
    elif is_unified_contract(contract_path):
        # No explicit processor dir: use the contract's inline processor (if any)
        # so the crop/resize baked into the dataset matches what deploy applies.
        inline_spec = load_processor_spec(contract_path)
        if inline_spec is not None:
            if not PROCESSORS_AVAILABLE:
                raise ImportError(
                    "Contract has an inline processor but the LeRobot processor "
                    "module is unavailable. Install lerobot with processor support."
                )
            # Unconditionally strict (unlike the warn-by-default deploy gate):
            # processor output is baked into every frame here, and a
            # processor/contract shape drift would fail every episode at
            # add_frame — after the dataset dir was already created.
            check_processor_vs_contract(inline_spec, contract).raise_or_warn(
                strict=True
            )
            logging.info("Building observation processor from inline contract spec.")
            obs_processor = _processors.build_observation_processor(
                inline_spec,
                to_transition=observation_to_transition,
                to_output=transition_to_observation,
            )

    if action_processor_path:
        logging.info("Loading action processor from %s", action_processor_path)
        action_processor = RobotProcessorPipeline.from_pretrained(
            action_processor_path,
            config_filename="robot_action_processor.json",
            to_transition=robot_action_to_transition,
            to_output=transition_to_robot_action,
        )

    if obs_processor is None:
        n_image_specs = sum(
            1 for s in specs if isinstance(s, ObservationStreamSpec) and s.is_image
        )
        if n_image_specs:
            logging.info(
                "No observation processor given; resizing %d image stream(s) to the "
                "contract image_shape (nearest-neighbor, no crop).",
                n_image_specs,
            )

    all_bag_dirs = find_bag_dirs(raw_dir)
    total_bags = len(all_bag_dirs)
    logging.info("Found %d bags in %s", total_bags, raw_dir)

    # Select shard subset if sharding
    if num_shards is not None:
        if shard_index is None:
            raise ValueError("shard_index required when num_shards is specified")
        if shard_index >= num_shards:
            raise ValueError(
                f"shard_index ({shard_index}) >= num_shards ({num_shards})"
            )

        bag_dirs = all_bag_dirs[shard_index::num_shards]
        logging.info(
            "Shard %d/%d: processing %d bags", shard_index, num_shards, len(bag_dirs)
        )
    else:
        bag_dirs = all_bag_dirs

    if not bag_dirs:
        logging.warning("No bags to process in this shard")
        return

    # Resolve the output location.
    if repo_id is None and root is None:
        # Unified convention: no explicit root/repo-id, so save under the shared
        # NAS root with a date/time-stamped, self-describing dataset name:
        #   DATASET_NAS_ROOT/<YYYY.MM.DD>/<HH.MM.SS>_dataset_<robot>_<task>_<user>
        if num_shards is not None:
            logging.warning(
                "Unified naming convention used with sharding: each shard computes "
                "its own timestamp and will write to a different directory. Pass "
                "--root/--repo-id to keep shards together."
            )
        now = datetime.now(DATASET_TZ)
        dataset_name = (
            f"{now:%H.%M.%S}_dataset_"
            f"{_sanitize_name_part(contract.robot_type)}_"
            f"{_sanitize_name_part(task or 'unknown')}_"
            f"{_sanitize_name_part(user_name or 'unknown')}"
        )
        dataset_root = DATASET_NAS_ROOT / f"{now:%Y.%m.%d}" / dataset_name
        repo_id = dataset_name
        logging.info("No --root/--repo-id given; saving dataset to %s", dataset_root)
    else:
        repo_id = repo_id or raw_dir.name
        # LeRobot uses root directly as dataset path, so append repo_id.
        dataset_root = root / repo_id if root else None

    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_root,
        robot_type=contract.robot_type,
        fps=contract.fps,
        features=features,
        vcodec=vcodec,
    )

    # Save a copy of the contract alongside the dataset for provenance/reproducibility.
    try:
        dataset_dir = Path(lerobot_dataset.root)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        contract_copy = dataset_dir / contract_path.name
        shutil.copy2(contract_path, contract_copy)
        logging.info("Saved a copy of the contract to %s", contract_copy)
    except Exception as e:  # noqa: BLE001
        logging.warning("Failed to copy contract into dataset directory: %s", e)

    # Record the arguments used to generate this dataset for reproducibility.
    try:
        dataset_dir = Path(lerobot_dataset.root)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        args_record = {
            "raw_dir": str(raw_dir),
            "repo_id": repo_id,
            "contract": str(contract_path),
            "root": str(root) if root is not None else None,
            "push_to_hub": push_to_hub,
            "num_shards": num_shards,
            "shard_index": shard_index,
            "vcodec": vcodec,
            "observation_processor": (
                str(observation_processor_path)
                if observation_processor_path is not None
                else None
            ),
            "action_processor": (
                str(action_processor_path)
                if action_processor_path is not None
                else None
            ),
            "task": task,
            "user_name": user_name,
            "generated_at": datetime.now(DATASET_TZ).isoformat(),
        }
        args_path = dataset_dir / "args.yaml"
        with args_path.open("w") as f:
            yaml.safe_dump(args_record, f, sort_keys=False)
        logging.info("Saved generation arguments to %s", args_path)
    except Exception as e:  # noqa: BLE001
        logging.warning("Failed to write args.yaml into dataset directory: %s", e)

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
            for frame in _stream_frames_from_bag(
                bag_dir,
                specs,
                obs_processor=obs_processor,
                action_processor=action_processor,
            ):
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

    parser = argparse.ArgumentParser(description="Port ROS2 bags to LeRobot dataset")

    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing bag subdirectories",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help=(
            "HuggingFace repository ID (e.g., my_org/my_dataset). If --root is set "
            "but this is omitted, defaults to the raw-dir name. If neither --root "
            "nor --repo-id is set, the unified NAS naming convention is used."
        ),
    )
    parser.add_argument(
        "--contract", type=Path, required=True, help="Rosetta contract YAML path"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Parent directory for datasets. Dataset saved to root/repo-id. (default: ~/.cache/huggingface/lerobot)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload to HuggingFace Hub after porting",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Total number of shards for parallel processing",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Index of this shard (0 to num-shards-1)",
    )
    parser.add_argument(
        "--vcodec",
        type=str,
        default="libsvtav1",
        choices=["libsvtav1", "libx264", "h264", "hevc", "h264_nvenc"],
        help="Video codec for encoding (default: libsvtav1). Use libx264/h264 for faster encoding.",
    )
    parser.add_argument(
        "--observation-processor",
        type=Path,
        default=None,
        help="Path to saved RobotObservationProcessor config directory",
    )
    parser.add_argument(
        "--action-processor",
        type=Path,
        default=None,
        help="Path to saved RobotActionProcessor config directory",
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        default=None,
        help=(
            "Path to a task config YAML (e.g. configs/task/tossing.yaml). The "
            "'task_name' field is used in the unified dataset naming convention."
        ),
    )
    parser.add_argument(
        "--user-name",
        type=str,
        default=getpass.getuser(),
        help=(
            "User name used in the unified dataset naming convention "
            "(default: current system user)."
        ),
    )

    args = parser.parse_args()

    task = _read_task_name(args.task_config) if args.task_config else None

    try:
        port_bags(
            raw_dir=args.raw_dir,
            repo_id=args.repo_id,
            contract_path=args.contract,
            root=args.root,
            push_to_hub=args.push_to_hub,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            vcodec=args.vcodec,
            observation_processor_path=args.observation_processor,
            action_processor_path=args.action_processor,
            task=task,
            user_name=args.user_name,
        )
    except KeyboardInterrupt:
        logging.info("\nInterrupted by user")
    except Exception as e:
        logging.error("Error: %s", e)
        raise


if __name__ == "__main__":
    main()
