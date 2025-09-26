#!/usr/bin/env python3
"""
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# contract loader (your module)
from rosetta.contract_utils import load_contract  # adjust import if needed

# LeRobot v3 writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# -------------------- Helpers --------------------

def _read_yaml(p: Path) -> Dict[str, Any]:
    with p.open("r") as f:
        return yaml.safe_load(f) or {}

def _topic_type_map(reader: rosbag2_py.SequentialReader) -> Dict[str, str]:
    return {t.name: t.type for t in reader.get_all_topics_and_types()}

def _ns_to_s(ns: int | np.ndarray) -> float | np.ndarray:
    return ns / 1e9

def _build_uniform_timeline(start_ns: int, end_ns: int, fps: float) -> np.ndarray:
    if end_ns <= start_ns:
        return np.array([start_ns], dtype=np.int64)
    step = int(round(1e9 / fps))
    n = 1 + (end_ns - start_ns) // step
    return start_ns + np.arange(n, dtype=np.int64) * step

def _resample_hold(ts_ns: np.ndarray, vals: List[Any], timeline_ns: np.ndarray) -> List[Any]:
    out, j, last = [], 0, None
    for t in timeline_ns:
        while j + 1 < len(ts_ns) and ts_ns[j + 1] <= t:
            j += 1
        if j < len(vals) and ts_ns[j] <= t:
            last = vals[j]
        out.append(last)
    return out

# --- message decoders ---

def _image_to_numpy_rgb_u8(msg, expected_encoding: str = "bgr8", resize_hw: Tuple[int, int] | None = None):
    h, w = msg.height, msg.width
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, -1)  # HWC
    enc = (expected_encoding or "bgr8").lower()
    if enc not in ("bgr8", "rgb8"):
        raise ValueError(f"Unsupported image encoding '{expected_encoding}', expected 'bgr8' or 'rgb8'")
    if enc == "bgr8":
        arr = arr[..., ::-1]  # BGR->RGB
    if resize_hw and (h, w) != tuple(resize_hw):
        try:
            import cv2
            arr = cv2.resize(arr, (resize_hw[1], resize_hw[0]), interpolation=cv2.INTER_AREA)
        except Exception:
            rh, rw = int(resize_hw[0]), int(resize_hw[1])
            y_idx = (np.linspace(0, h - 1, rh)).astype(np.int32)
            x_idx = (np.linspace(0, w - 1, rw)).astype(np.int32)
            arr = arr[y_idx][:, x_idx]
    return arr  # uint8 RGB HxWxC

def _joint_vector(msg, names: List[str], field: str) -> np.ndarray:
    idx = {n: i for i, n in enumerate(msg.name)}
    src = getattr(msg, field)
    out = [float(src[idx[n]]) if (n in idx and idx[n] < len(src)) else np.nan for n in names]
    return np.asarray(out, dtype=np.float32)

def _float32_multiarray(msg) -> np.ndarray:
    data = np.asarray(msg.data, dtype=np.float32)
    return data

def _int32_multiarray(msg) -> np.ndarray:
    data = np.asarray(msg.data, dtype=np.int32)
    return data

def _string_msg(msg) -> str:
    return str(getattr(msg, "data", ""))


# -------------------- Contract → features (v3.0) --------------------

def build_features_from_contract(contract, use_videos: bool) -> Dict[str, Dict]:
    feats: Dict[str, Dict] = {}
    # Observations
    for o in contract.observations:
        key = o.key
        if o.image:
            feats[key] = {"dtype": "video" if use_videos else "image"}
        else:
            sel = (o.selector or {})
            dim = len(sel.get("names", [])) or (o.shape[0] if getattr(o, "shape", None) else 0)
            if dim <= 0:
                raise ValueError(f"Cannot infer shape for vector feature '{key}' from contract.")
            feats[key] = {"dtype": "float32", "shape": (dim,), "names": sel.get("names", None)}
    # Tasks (strings)
    for t in contract.tasks:
        feats[t.key] = {"dtype": "string", "shape": (1,)}
    # Actions (array-like; minimal)
    for a in contract.actions:
        # We don't guess shapes deeply—writer will take vectors as-is
        # Keep the feature name exactly the action key (e.g., action.continuous)
        feats[a.key] = {"dtype": "float32"}  # default; int32 streams will be inferred at write time
    return feats


# -------------------- Core export --------------------

def export_bags_to_v30(
    bag_dirs: List[Path],
    contract_path: Path,
    out_root: Path,
    repo_id: str,
    use_videos: bool = True,
    image_threads: int = 4,
    chunk_size: int = 1000,
    data_mb: int = 100,
    video_mb: int = 500,
):
    # Contract + dataset init
    contract = load_contract(str(contract_path))
    fps = float(contract.rate_hz)
    features = build_features_from_contract(contract, use_videos=use_videos)
    robot_type = getattr(contract, "robot_type", None)

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(round(fps)),
        features=features,
        root=out_root,
        robot_type=robot_type,
        use_videos=use_videos,
        image_writer_processes=0,
        image_writer_threads=image_threads,
        batch_encoding_size=1,
    )
    ds.meta.update_chunk_settings(
        chunks_size=chunk_size,
        data_files_size_in_mb=data_mb,
        video_files_size_in_mb=video_mb,
    )

    # collect operator prompts for per-episode sidecar
    operator_prompts: Dict[int, str] = {}

    # Process every bag as one episode
    for epi_idx, bag_dir in enumerate(bag_dirs):
        print(f"[INFO] Episode {epi_idx}: {bag_dir}")

        # Read rosbag2 metadata.yaml
        meta_yaml = _read_yaml(bag_dir / "metadata.yaml")
        storage = meta_yaml.get("storage_identifier", "mcap")
        prompt = (meta_yaml.get("custom_data") or {}).get("lerobot.operator_prompt", "")

        # Open bag
        reader = rosbag2_py.SequentialReader()
        storage_opts = rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=storage)
        converter = rosbag2_py.ConverterOptions("", "")
        reader.open(storage_opts, converter)
        tmap = _topic_type_map(reader)

        # Plan streams from contract
        streams: Dict[str, Dict[str, Any]] = {}

        # Observations
        for o in contract.observations:
            topic = o.topic
            if topic not in tmap:
                print(f"[WARN] Missing observation '{o.key}' topic: {topic}")
                continue
            streams[o.key] = dict(
                topic=topic,
                ros_type=o.type or tmap[topic],
                image=o.image or None,
                selector=o.selector or None,
                ts=[], val=[],
            )

        # Tasks (strings)
        for t in contract.tasks:
            topic = t.topic
            if topic not in tmap:
                print(f"[WARN] Missing task '{t.key}' topic: {topic}")
                continue
            streams[t.key] = dict(
                topic=topic,
                ros_type=t.type or tmap[topic],
                ts=[], val=[],
                _is_task=True,
            )

        # Actions (arrays; minimal support)
        for a in contract.actions:
            topic = a.publish.topic
            if topic not in tmap:
                print(f"[WARN] Missing action '{a.key}' topic: {topic}")
                continue
            streams[a.key] = dict(
                topic=topic,
                ros_type=a.publish.type or tmap[topic],
                ts=[], val=[],
                _is_action=True,
            )

        # Decode pass
        while reader.has_next():
            topic, data, t_ns = reader.read_next()
            for key, st in streams.items():
                if st["topic"] != topic:
                    continue
                msg_type = get_message(st["ros_type"])
                msg = deserialize_message(data, msg_type)

                rtype = st["ros_type"]
                if rtype.endswith("/Image"):
                    enc = (st.get("image") or {}).get("encoding", "bgr8")
                    resize = (st.get("image") or {}).get("resize", None)
                    arr = _image_to_numpy_rgb_u8(msg, enc, resize)
                    st["ts"].append(int(t_ns)); st["val"].append(arr)
                elif rtype.endswith("/JointState") and st.get("selector"):
                    names = st["selector"].get("names", [])
                    field = st["selector"].get("field", "position")
                    vec = _joint_vector(msg, names, field)
                    st["ts"].append(int(t_ns)); st["val"].append(vec)
                elif rtype.endswith("/Float32MultiArray"):
                    vec = _float32_multiarray(msg)
                    st["ts"].append(int(t_ns)); st["val"].append(vec)
                elif rtype.endswith("/Int32MultiArray"):
                    vec = _int32_multiarray(msg)
                    st["ts"].append(int(t_ns)); st["val"].append(vec)
                elif rtype.endswith("/String"):
                    st["ts"].append(int(t_ns)); st["val"].append(_string_msg(msg))
                else:
                    # Keep script minimal: silently ignore unsupported types
                    pass

        # Build timeline
        series_ts = [np.asarray(v["ts"], dtype=np.int64) for v in streams.values() if len(v["ts"]) > 0]
        if not series_ts:
            raise RuntimeError(f"No usable messages found in {bag_dir} for the given contract.")
        start_ns = min(ts.min() for ts in series_ts)
        end_ns   = max(ts.max() for ts in series_ts)
        tl_ns    = _build_uniform_timeline(start_ns, end_ns, fps)
        T = len(tl_ns)

        # Resample (HOLD)
        resampled: Dict[str, List[Any]] = {}
        for key, st in streams.items():
            ts = np.asarray(st["ts"], dtype=np.int64)
            if len(ts) == 0:
                resampled[key] = [None] * T
            else:
                resampled[key] = _resample_hold(ts, st["val"], tl_ns)

        # Write frames
        for i in range(T):
            frame: Dict[str, Any] = {}
            frame["timestamp"] = float(_ns_to_s(tl_ns[i] - tl_ns[0]))
            for key, seq in resampled.items():
                v = seq[i]
                if v is None:
                    continue
                frame[key] = v
            ds.add_frame(frame)

        ds.save_episode()
        operator_prompts[epi_idx] = prompt or ""
        print(f"[OK] Wrote episode {epi_idx} (frames={T})")

    ds.stop_image_writer()

    # Sidecar with operator prompts (kept separate from task.* streams)
    meta_dir = out_root / repo_id / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with (meta_dir / "operator_prompts.json").open("w") as f:
        json.dump(operator_prompts, f, indent=2)

    print(f"\n[COMPLETE] v3.0 dataset at: {out_root / repo_id}")
    print("  - data/chunk-*/file-*.parquet")
    if use_videos:
        print("  - videos/<camera_key>/chunk-*/file-*.mp4")
    print("  - meta/info.json, meta/tasks.parquet, meta/stats.json, meta/episodes/chunk-*/file-*.parquet")
    print("  - meta/operator_prompts.json  # per-episode operator prompts")


# -------------------- CLI --------------------

def parse_args():
    ap = argparse.ArgumentParser(description="ROS2 bag -> LeRobot v3.0 dataset exporter (Option A)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bag", help="Path to a single episode bag directory")
    g.add_argument("--bags", nargs="+", help="Paths to multiple episode bag directories")

    ap.add_argument("--contract", required=True, help="Path to the YAML contract used for recording")
    ap.add_argument("--out", required=True, help="Output root (dataset will live under OUT/REPO_ID)")
    ap.add_argument("--repo-id", default="rosbag_v30", help="Dataset repo_id (folder name under OUT)")
    ap.add_argument("--no-videos", action="store_true", help="Store images instead of encoding videos")
    ap.add_argument("--image-threads", type=int, default=4, help="Async image writer threads")
    ap.add_argument("--chunk-size", type=int, default=1000, help="Max files per chunk directory")
    ap.add_argument("--data-mb", type=int, default=100, help="Target max size for data parquet files (MB)")
    ap.add_argument("--video-mb", type=int, default=500, help="Target max size for video files (MB)")
    return ap.parse_args()

def main():
    args = parse_args()
    bag_dirs = [Path(args.bag)] if args.bag else [Path(p) for p in args.bags]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    export_bags_to_v30(
        bag_dirs=bag_dirs,
        contract_path=Path(args.contract),
        out_root=out_root,
        repo_id=args.repo_id,
        use_videos=not args.no_videos,
        image_threads=args.image_threads,
        chunk_size=args.chunk_size,
        data_mb=args.data_mb,
        video_mb=args.video_mb,
    )

if __name__ == "__main__":
    main()
