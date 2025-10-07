#!/usr/bin/env python3
"""
ROS 2 bag → LeRobot v3.0 exporter (tiny, unified, contract-true)

This version depends ONLY on two shared modules:
  - rosetta.common.contract_utils: Contract + SpecView + features + zero_pad
  - rosetta.common.signal_utils:   decode_value + (hold/asof/drop) + stamps

Result: the exact same preparation code-paths are used offline (conversion)
and online (live inference), minimizing train/serve skew.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# ---- LeRobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ---- Shared core (ONLY these two)
from rosetta.common.contract_utils import (
    load_contract,
    iter_specs,
    feature_from_spec,
    zero_pad as make_zero_pad,   # alias to avoid name clash with dict var
)
from rosetta.common.signal_utils import (
    decode_value,
    resample,
    stamp_from_header_ns,
)

# ---------------------------------------------------------------------------

@dataclass
class _Stream:
    spec: Any          # SpecView
    ros_type: str
    ts: List[int]
    val: List[Any]

# ---------------------------------------------------------------------------

def _read_yaml(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _topic_type_map(reader: rosbag2_py.SequentialReader) -> Dict[str, str]:
    return {t.name: t.type for t in reader.get_all_topics_and_types()}

def _nearest_resize_rgb(arr: np.ndarray, rh: int, rw: int) -> np.ndarray:
    if arr.shape[0] == rh and arr.shape[1] == rw:
        return arr
    y = np.clip(np.linspace(0, arr.shape[0] - 1, rh), 0, arr.shape[0] - 1).astype(np.int64)
    x = np.clip(np.linspace(0, arr.shape[1] - 1, rw), 0, arr.shape[1] - 1).astype(np.int64)
    return arr[y][:, x]

def _coerce_image(val: Any, hwc: Tuple[int, int, int]) -> np.ndarray:
    """
    Ensure HWC uint8 RGB and requested size. We expect decode_value() to already
    return HxWx3 uint8 RGB when the spec is an image; this is a defensive check.
    """
    arr = np.asarray(val)
    # CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and (arr.shape[2] not in (1, 3)):
        arr = np.transpose(arr, (1, 2, 0))
    # gray -> RGB
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    # dtype → uint8
    if arr.dtype != np.uint8:
        if arr.dtype.kind == "f":
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0 + 0.5).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    # size
    h, w, _ = hwc
    if arr.shape[:2] != (h, w):
        arr = _nearest_resize_rgb(arr, h, w)
    return arr

def _plan_streams(specs: Iterable[Any], tmap: Dict[str, str]) -> Tuple[Dict[str, _Stream], Dict[str, List[str]]]:
    """
    Plan streams once and build a topic -> [keys] index for fast dispatch.
    """
    streams: Dict[str, _Stream] = {}
    by_topic: Dict[str, List[str]] = {}
    for sv in specs:
        if sv.topic not in tmap:
            kind = "action" if sv.is_action else "observation"
            print(f"[WARN] Missing {kind} '{sv.key}' topic in bag: {sv.topic}")
            continue
        rt = sv.ros_type or tmap[sv.topic]
        streams[sv.key] = _Stream(spec=sv, ros_type=rt, ts=[], val=[])
        by_topic.setdefault(sv.topic, []).append(sv.key)
    if not streams:
        raise RuntimeError("No contract topics found in bag.")
    return streams, by_topic

# ---------------------------------------------------------------------------

def export_bags_to_lerobot(
    bag_dirs: List[Path],
    contract_path: Path,
    out_root: Path,
    repo_id: str = "rosbag_v30",
    use_videos: bool = True,
    image_writer_threads: int = 4,
    chunk_size: int = 1000,
    data_mb: int = 100,
    video_mb: int = 500,
    timestamp_source: str = "contract",   # 'contract' | 'bag' | 'header'
) -> None:
    # Contract + specs
    contract = load_contract(contract_path)
    fps = float(contract.rate_hz)
    if fps <= 0:
        raise ValueError("Contract rate_hz must be > 0")
    step_ns = int(round(1e9 / fps))
    specs = list(iter_specs(contract))

    # Features (also detect first image key as anchor)
    features: Dict[str, Dict[str, Any]] = {}
    primary_image_key: Optional[str] = None
    for sv in specs:
        k, ft, is_img = feature_from_spec(sv, use_videos)
        features[k] = ft
        if is_img and primary_image_key is None:
            primary_image_key = sv.key

    # Dataset
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(round(fps)),
        features=features,
        root=out_root,
        robot_type=contract.robot_type,
        use_videos=use_videos,
        image_writer_processes=0,          # keep simple & predictable
        image_writer_threads=image_writer_threads,
        batch_encoding_size=1,
    )
    ds.meta.update_chunk_settings(
        chunks_size=chunk_size,
        data_files_size_in_mb=data_mb,
        video_files_size_in_mb=video_mb,
    )

    # Precompute zero pads + shapes for fast frame assembly
    zero_pad_map = {k: make_zero_pad(ft) for k, ft in features.items()}
    write_keys = [k for k, ft in features.items() if ft["dtype"] in ("video", "image", "float32", "float64", "string")]
    shapes = {k: tuple(features[k]["shape"]) for k in write_keys}

    # Episodes
    for epi_idx, bag_dir in enumerate(bag_dirs):
        print(f"[Episode {epi_idx}] {bag_dir}")

        meta = _read_yaml(bag_dir / "metadata.yaml")
        info = meta.get("rosbag2_bagfile_information") or {}
        storage = info.get("storage_identifier") or "mcap"
        meta_dur_ns = int((info.get("duration") or {}).get("nanoseconds") or 0)

        # operator prompt (if present)
        prompt = ""
        for node in (meta, info):
            if isinstance(node, dict):
                cd = node.get("custom_data")
                if isinstance(cd, dict) and "lerobot.operator_prompt" in cd:
                    prompt = cd["lerobot.operator_prompt"] or prompt

        # Reader
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=storage),
            rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
        )
        tmap = _topic_type_map(reader)

        # Plan once
        streams, by_topic = _plan_streams(specs, tmap)

        # Counters for light diagnostics
        decoded_msgs = 0

        # Decode single pass
        while reader.has_next():
            topic, data, bag_ns = reader.read_next()
            if topic not in by_topic:
                continue
            for key in by_topic[topic]:
                st = streams[key]
                msg = deserialize_message(data, get_message(st.ros_type))
                sv = st.spec

                # Timestamp selection policy
                if timestamp_source == "bag":
                    ts_sel = int(bag_ns)
                elif timestamp_source == "header":
                    ts_sel = stamp_from_header_ns(msg) or int(bag_ns)
                else:  # 'contract' (per-spec stamp_src)
                    ts_sel = int(bag_ns)
                    if sv.stamp_src == "header":
                        hdr = stamp_from_header_ns(msg)
                        if hdr is not None:
                            ts_sel = int(hdr)

                val = decode_value(st.ros_type, msg, sv)
                if val is not None:
                    st.ts.append(ts_sel)
                    st.val.append(val)
                    decoded_msgs += 1

        if decoded_msgs == 0:
            raise RuntimeError(f"No usable messages in {bag_dir} (none decoded).")

        # Choose anchor + duration
        valid_ts = [np.asarray(st.ts, dtype=np.int64) for st in streams.values() if st.ts]
        if not valid_ts:
            raise RuntimeError(f"No usable messages in {bag_dir} (no timestamps).")

        if primary_image_key and streams.get(primary_image_key) and streams[primary_image_key].ts:
            start_ns = int(np.asarray(streams[primary_image_key].ts, dtype=np.int64).min())
        else:
            start_ns = int(min(ts.min() for ts in valid_ts))
        ts_max = int(max(ts.max() for ts in valid_ts))
        observed_dur_ns = max(0, ts_max - start_ns)
        # Prefer observed duration unless bag metadata matches within ~2 ticks
        dur_ns = observed_dur_ns if (meta_dur_ns <= 0 or abs(meta_dur_ns - observed_dur_ns) > 2 * step_ns) else meta_dur_ns

        # Ticks
        n_ticks = int(dur_ns // step_ns) + 1
        ticks_ns = start_ns + np.arange(n_ticks, dtype=np.int64) * step_ns

        # Resample onto ticks
        resampled: Dict[str, List[Any]] = {}
        for key, st in streams.items():
            if not st.ts:
                resampled[key] = [None] * n_ticks
                continue
            ts = np.asarray(st.ts, dtype=np.int64)
            pol = st.spec.resample_policy
            resampled[key] = resample(pol, ts, st.val, ticks_ns, step_ns, st.spec.asof_tol_ms)

        # Write frames
        for i in range(n_ticks):
            frame: Dict[str, Any] = {}
            for name in write_keys:
                ft = features[name]
                dtype = ft["dtype"]
                val = resampled.get(name, [None] * n_ticks)[i]

                if val is None:
                    frame[name] = zero_pad_map[name]
                    continue

                if dtype in ("video", "image"):
                    frame[name] = _coerce_image(val, shapes[name])

                elif dtype in ("float32", "float64"):
                    tgt_dt = np.float32 if dtype == "float32" else np.float64
                    arr = np.asarray(val, dtype=tgt_dt).reshape(-1)
                    exp = int(ft["shape"][0])
                    if arr.shape[0] != exp:
                        fixed = np.zeros((exp,), dtype=tgt_dt)
                        fixed[: min(exp, arr.shape[0])] = arr[: min(exp, arr.shape[0])]
                        arr = fixed
                    frame[name] = arr

                elif dtype == "string":
                    frame[name] = str(val)

                else:
                    # Fallback – should not happen with current features
                    frame[name] = val

            # Text task prompt (if any) is a first-class feature in LeRobot
            frame["task"] = prompt
            ds.add_frame(frame)

        ds.save_episode()
        print(f"  → saved {n_ticks} frames @ {int(round(fps))} FPS  | decoded_msgs={decoded_msgs}")

    print(f"\n[OK] Dataset root: {ds.root.resolve()}")
    if use_videos:
        print("  - videos/<image_key>/chunk-*/file-*.mp4")
    else:
        print("  - images/*/*.png")
    print("  - data/chunk-*/file-*.parquet")
    print("  - meta/info.json, meta/tasks.parquet, meta/stats.json, meta/episodes/*/*.parquet")

# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("ROS2 bag → LeRobot v3 (using rosetta.common.*)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bag", help="Path to a single bag directory (episode)")
    g.add_argument("--bags", nargs="+", help="Paths to multiple bag directories")
    ap.add_argument("--contract", required=True, help="Path to YAML contract")
    ap.add_argument("--out", required=True, help="Output dataset root")
    ap.add_argument("--repo-id", default="rosbag_v30", help="repo_id metadata")
    ap.add_argument("--no-videos", action="store_true", help="Store images instead of videos")
    ap.add_argument("--image-threads", type=int, default=4, help="Image writer threads")
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--data-mb", type=int, default=100)
    ap.add_argument("--video-mb", type=int, default=500)
    ap.add_argument(
        "--timestamp",
        choices=("contract", "bag", "header"),
        default="contract",
        help="Which time base to use when resampling: "
             "'contract' (per-spec), 'bag' (receive), or 'header' (message header).",
    )
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    bag_dirs = [Path(args.bag)] if args.bag else [Path(p) for p in args.bags]
    export_bags_to_lerobot(
        bag_dirs=bag_dirs,
        contract_path=Path(args.contract),
        out_root=Path(args.out),
        repo_id=args.repo_id,
        use_videos=not args.no_videos,
        image_writer_threads=args.image_threads,
        chunk_size=args.chunk_size,
        data_mb=args.data_mb,
        video_mb=args.video_mb,
        timestamp_source=args.timestamp,
    )

if __name__ == "__main__":
    main()
