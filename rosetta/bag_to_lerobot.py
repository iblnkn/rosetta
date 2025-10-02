#!/usr/bin/env python3
"""
ROS2 bag -> LeRobot v3.0 dataset exporter (robust 'names' handling)

- Ensures every float32 vector feature has a 'names' list in meta/info.json.
- Infers vector shapes from contract selector names or first message encountered.
- Works with image or video storage.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import yaml

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# contract loader + converters
from rosetta.contract_utils import (
    load_contract,
    decode_observation_msg,
    float32_multiarray_to_np,
    int32_multiarray_to_np,
)

# LeRobot v3 writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import write_info  # to patch meta/info.json on the fly


# -------------------- Tiny helpers --------------------

def _read_yaml(p: Path) -> Dict[str, Any]:
    with p.open("r") as f:
        return yaml.safe_load(f) or {}

def _topic_type_map(reader: rosbag2_py.SequentialReader) -> Dict[str, str]:
    return {t.name: t.type for t in reader.get_all_topics_and_types()}

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

def _gen_default_names(prefix: str, dim: int) -> List[str]:
    return [f"{prefix}.{i}" for i in range(dim)]

def _infer_vec_from_contract(obj) -> Tuple[int, Optional[List[str]]]:
    """
    For an observation/action spec, try to infer dimension and optional names from:
      - selector.names (if present)
      - shape[0] (if present)
    Returns (dim, names_or_None)
    """
    sel = getattr(obj, "selector", None) or {}
    names = sel.get("names")
    if names:
        return len(names), names
    shp = getattr(obj, "shape", None)
    if isinstance(shp, (list, tuple)) and len(shp) >= 1 and int(shp[0]) > 0:
        return int(shp[0]), None
    return 1, None  # fallback


# -------------------- Contract → features --------------------

def build_features_from_contract(contract, use_videos: bool) -> Dict[str, Dict]:
    feats: Dict[str, Dict] = {}

    # Observations
    for o in contract.observations:
        key = o.key
        if o.image:
            resize = tuple(o.image.get("resize", ())) if o.image else ()
            H, W = (int(resize[0]), int(resize[1])) if (resize and len(resize)==2) else (None, None)
            shape = (H, W, 3) if (H and W) else (224, 224, 3)  # or drop the fallback if you prefer
            feats[key] = {
                "dtype": "video" if use_videos else "image",
                "shape": shape,
                "names": ["height","width","channel"]
            }
        else:
            dim, names = _infer_vec_from_contract(o)
            if names is None:
                names = _gen_default_names(key, dim)
            feats[key] = {"dtype": "float32", "shape": (dim,), "names": names}

    # Actions — try to honor contract selector/shape; special-case /cmd_vel
    for a in contract.actions:
        if getattr(a, "publish", None) and a.publish.topic == "/cmd_vel":
            names = ["linear_x", "angular_z"]
            feats[a.key] = {"dtype": "float32", "shape": (2,), "names": names}
            continue

        dim, names = _infer_vec_from_contract(a)
        if names is None:
            names = _gen_default_names(a.key, dim)
        feats[a.key] = {"dtype": "float32", "shape": (dim,), "names": names}

    # (We keep tasks out of features; tasks are added per-frame via 'task' string.)

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
    print(f"Generated features (pre-create): {features}")
    robot_type = getattr(contract, "robot_type", None)

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(round(fps)),
        features=features,
        root=out_root,      # NOTE: dataset root == out_root (no extra subdir)
        robot_type=robot_type,
        use_videos=use_videos,
        image_writer_processes=0,
        image_writer_threads=image_threads,
        batch_encoding_size=1,
    )

    # Apply chunk/file size preferences to meta/info.json
    ds.meta.update_chunk_settings(
        chunks_size=chunk_size,
        data_files_size_in_mb=data_mb,
        video_files_size_in_mb=video_mb,
    )

    # ---- Pass 0 note: we will auto-patch any float32 features that lack 'names' (defensive) ----
    patched = False
    for k, ft in ds.meta.info["features"].items():
        if ft.get("dtype") == "float32" and "names" not in ft:
            dim = int(ft["shape"][0]) if "shape" in ft and ft["shape"] else 1
            ds.meta.info["features"][k]["names"] = _gen_default_names(k, dim)
            patched = True
    if patched:
        write_info(ds.meta.info, ds.meta.root)

    # Process every bag as one episode
    for epi_idx, bag_dir in enumerate(bag_dirs):
        print(f"[INFO] Episode {epi_idx}: {bag_dir}")

        # Read rosbag2 metadata.yaml
        meta_yaml = _read_yaml(bag_dir / "metadata.yaml")
        info = meta_yaml.get("rosbag2_bagfile_information", {}) or {}
        storage = info.get("storage_identifier") or "mcap"

        # Prompt can be top-level or nested under rosbag2_bagfile_information
        prompt = ""
        cd_top = meta_yaml.get("custom_data") or {}
        if isinstance(cd_top, dict):
            prompt = cd_top.get("lerobot.operator_prompt", "") or prompt
        cd_info = info.get("custom_data") or {}
        if isinstance(cd_info, dict):
            prompt = cd_info.get("lerobot.operator_prompt", "") or prompt

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
                spec=o,  # used by decode_observation_msg
                ros_type=o.type or tmap[topic],
                ts=[],
                val=[],
                _is_action=False,
            )

        # Use first image observation present as timeline anchor
        primary_observation_key = None
        for o in contract.observations:
            if o.image and (o.key in streams):
                primary_observation_key = o.key
                break

        # Actions
        for a in contract.actions:
            topic = a.publish.topic
            if topic not in tmap:
                print(f"[WARN] Missing action '{a.key}' topic: {topic}")
                continue
            streams[a.key] = dict(
                topic=topic,
                ros_type=a.publish.type or tmap[topic],
                ts=[],
                val=[],
                _is_action=True,
            )

        # Decode pass
        while reader.has_next():
            topic, data, t_ns = reader.read_next()
            for key, st in streams.items():
                if st["topic"] != topic:
                    continue
                msg_type = get_message(st.get("ros_type"))
                msg = deserialize_message(data, msg_type)
                rtype = st["ros_type"]

                # Observations via helper
                if not st["_is_action"]:
                    val = decode_observation_msg(st.get("spec"), msg)
                    if val is not None:
                        st["ts"].append(int(t_ns)); st["val"].append(val)
                    continue

                # Actions
                if rtype.endswith("/Float32MultiArray"):
                    st["ts"].append(int(t_ns)); st["val"].append(float32_multiarray_to_np(msg))
                elif rtype.endswith("/Int32MultiArray"):
                    st["ts"].append(int(t_ns)); st["val"].append(int32_multiarray_to_np(msg))
                elif rtype.endswith("/Twist"):
                    st["ts"].append(int(t_ns))
                    st["val"].append(np.asarray([msg.linear.x, msg.angular.z], dtype=np.float32))
                elif rtype.endswith("/String"):
                    st["ts"].append(int(t_ns)); st["val"].append(str(getattr(msg, "data", "")))
                else:
                    # Minimal script: ignore other types
                    pass

        # If we can infer action dims from the first decoded message, patch feature shapes/names accordingly
        feature_patch_needed = False
        for key, st in streams.items():
            if not st.get("_is_action"):
                continue
            vals = st["val"]
            if not vals:
                continue
            sample = vals[0]
            if isinstance(sample, (list, tuple)):
                dim = len(sample)
            elif isinstance(sample, np.ndarray):
                dim = int(sample.shape[-1]) if sample.ndim > 0 else 1
            else:
                dim = 1
            ft = ds.meta.info["features"].get(key, None)
            if ft and ft.get("dtype") == "float32":
                cur_shape = tuple(ft.get("shape", ()) or ())
                if not cur_shape or (len(cur_shape) == 1 and int(cur_shape[0]) != dim):
                    ds.meta.info["features"][key]["shape"] = (dim,)
                    ds.meta.info["features"][key]["names"] = _gen_default_names(key, dim)
                    feature_patch_needed = True

        if feature_patch_needed:
            write_info(ds.meta.info, ds.meta.root)
            # Recreate empty HF dataset with the updated schema (no frames added yet)
            ds.hf_dataset = ds.create_hf_dataset()
            print(f"[INFO] Patched action shapes/names from data. New features: {ds.meta.info['features']}")

        # Build timeline (prefer image timestamps)
        if primary_observation_key and len(streams[primary_observation_key]["ts"]) > 0:
            tl_ns = np.asarray(streams[primary_observation_key]["ts"], dtype=np.int64)
        else:
            series_ts = [np.asarray(v["ts"], dtype=np.int64) for v in streams.values() if len(v["ts"]) > 0]
            if not series_ts:
                raise RuntimeError(f"No usable messages found in {bag_dir} for the given contract.")
            start_ns = min(ts.min() for ts in series_ts)
            end_ns = max(ts.max() for ts in series_ts)
            tl_ns = _build_uniform_timeline(start_ns, end_ns, fps)

        # We only require actual data features (ignore library default meta features)
        data_keys = [k for k, ft in ds.meta.info["features"].items()
                     if ft["dtype"] in ("video", "image", "float32", "string")
                     and k in streams]  # must be present in this episode

        # Trim timeline so all required features have at least one sample
        first_seen = []
        for name in data_keys:
            st = streams.get(name)
            if st and len(st["ts"]) > 0:
                first_seen.append(int(np.asarray(st["ts"], dtype=np.int64)[0]))
        if first_seen:
            min_valid_ns = max(first_seen)
            tl_ns = tl_ns[tl_ns >= min_valid_ns]

        T = len(tl_ns)
        if T == 0:
            raise RuntimeError(f"No overlapping time range across required streams in {bag_dir}.")

        # Resample (HOLD)
        resampled: Dict[str, List[Any]] = {}
        for key, st in streams.items():
            ts = np.asarray(st["ts"], dtype=np.int64)
            if len(ts) == 0:
                resampled[key] = [None] * T
            else:
                resampled[key] = _resample_hold(ts, st["val"], tl_ns)

        # Write frames
        written = 0
        for i in range(T):
            # Skip frame if any required feature missing
            if any(resampled.get(name, [None] * T)[i] is None for name in data_keys):
                continue

            frame: Dict[str, Any] = {}
            for name in data_keys:
                frame[name] = resampled[name][i]
            frame["task"] = prompt  # LeRobotDataset.add_frame() expects this

            ds.add_frame(frame)
            written += 1

        ds.save_episode()
        print(f"[OK] Wrote episode {epi_idx} (frames={written}/{T})")

    ds.stop_image_writer()

    print(f"\n[COMPLETE] v3.0 dataset at: {ds.root.resolve()}")
    print("  - data/chunk-*/file-*.parquet")
    if use_videos:
        print("  - videos/<camera_key>/chunk-*/file-*.mp4")
    print("  - meta/info.json, meta/tasks.parquet, meta/stats.json, meta/episodes/chunk-*/file-*.parquet")


# -------------------- CLI --------------------

def parse_args():
    ap = argparse.ArgumentParser(description="ROS2 bag -> LeRobot v3.0 dataset exporter (robust)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bag", help="Path to a single episode bag directory")
    g.add_argument("--bags", nargs="+", help="Paths to multiple episode bag directories")

    ap.add_argument("--contract", required=True, help="Path to the YAML contract used for recording")
    ap.add_argument("--out", required=True, help="Output root directory for the dataset")
    ap.add_argument("--repo-id", default="rosbag_v30", help="Dataset repo_id (metadata only)")
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
