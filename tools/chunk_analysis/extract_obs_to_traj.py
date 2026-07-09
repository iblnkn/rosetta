"""Add the observed robot state (/joint_states) to a merge-timeline data JSON,
placed on the TIMESTEP axis so it can be compared against each chunk's first
point (which is generated from the observation at t_observation).

Mapping: the merge dump records (wall_time, latest_action) per event; a linear
fit gives timestep = slope*wall + intercept (slope ~= fps). /joint_states stamps
are the same epoch clock, so each state sample maps to a timestep.

Joints: each of the dump header's ``action_features`` is matched to a
/joint_states name (normalized containment), per distinct name list —
/joint_states may interleave several publishers (per-arm drivers, gripper).
Only messages whose names fully match contribute; the rest are skipped and
counted, so a partial match is never plotted as the wrong arm.

Usage:
    python extract_obs_to_traj.py <bag_dir> <merge_dump.jsonl> <data.json>

Exit codes: 0 = overlay added or cleanly skipped; 2 = unusable dump.
"""
import json
import re
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState

from merge_dump_io import load_merge_dump

bag, dump, data_path = sys.argv[1], sys.argv[2], sys.argv[3]


def _norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _match_action_to_bag(action_names, bag_names):
    """Index into bag_names for each action feature, or None.

    Normalized containment match; only a full, collision-free mapping is
    accepted — a partial/ambiguous overlay would silently plot the wrong arm,
    which is exactly the failure this guards against.
    """
    normed_bag = [_norm(b) for b in bag_names]
    idx = []
    for a in action_names:
        na = _norm(a)
        hits = [
            i for i, nb in enumerate(normed_bag)
            if nb and (nb in na or na in nb)
        ]
        if len(hits) != 1:
            return None
        idx.append(hits[0])
    return idx if len(set(idx)) == len(idx) else None


# 1. wall_time -> timestep fit from the merge dump
header, evs = load_merge_dump(dump)
if not evs:
    print(f"error: {dump} has no merge events; cannot fit wall->timestep",
          file=sys.stderr)
    sys.exit(2)
action_names = (header or {}).get("action_features")
if not action_names:
    print("overlay skipped: dump header has no action_features to map "
          "/joint_states against")
    sys.exit(0)
wt = np.array([e["wall_time"] for e in evs], dtype=float)
la = np.array([e["latest_action"] for e in evs], dtype=float)
if len(evs) >= 2:
    slope, intercept = np.polyfit(wt, la, 1)
else:
    # One event can't anchor a line; use the header's fps as the slope.
    fps = (header or {}).get("fps")
    if not fps:
        print("error: single merge event and no fps in the dump header; "
              "cannot fit wall->timestep", file=sys.stderr)
        sys.exit(2)
    slope, intercept = float(fps), float(la[0] - float(fps) * wt[0])
print(f"wall->timestep fit: slope={slope:.3f} (~fps), intercept={intercept:.1f}")

# 2. read /joint_states, remap to action order, map stamp -> timestep
reader = rosbag2_py.SequentialReader()
# storage_id="" -> auto-detect; the recorder writes mcap or sqlite3.
reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=""),
            rosbag2_py.ConverterOptions("", ""))
# Filter in the storage layer: episode bags carry camera streams whose
# payloads we would otherwise decompress just to skip.
reader.set_filter(rosbag2_py.StorageFilter(topics=["/joint_states"]))
obs_t, obs_p = [], []
idx_by_names = {}  # per-publisher name list -> action-order indices, or None
skipped = 0
while reader.has_next():
    topic, raw, t = reader.read_next()
    if topic != "/joint_states":
        continue
    m = deserialize_message(raw, JointState)
    key = tuple(m.name)
    if key not in idx_by_names:
        idx_by_names[key] = _match_action_to_bag(action_names, list(key))
    name_idx = idx_by_names[key]
    if name_idx is None or len(m.position) < len(key):
        # Different publisher (no full match) or malformed/velocity-only
        # message (positions shorter than names) — never guess.
        skipped += 1
        continue
    if m.header.stamp.sec == 0 and m.header.stamp.nanosec == 0:
        # Uninitialized stamp (same convention as ros2_utils.
        # stamp_from_header_ns): epoch 0 through the fit would land at
        # timestep ~intercept, i.e. garbage.
        continue
    stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
    ts = slope * stamp + intercept
    obs_t.append(round(float(ts), 3))
    obs_p.append([round(float(m.position[i]), 5) for i in name_idx])

if not obs_t:
    print(f"overlay skipped: no /joint_states messages in {bag} matched "
          f"the dump's action_features (name lists seen: "
          f"{[list(k) for k in idx_by_names]})")
    sys.exit(0)

# 3. merge into the data json
data = json.load(open(data_path))
data["obs_t"] = obs_t
data["obs_p"] = obs_p
json.dump(data, open(data_path, "w"))
if skipped:
    print(f"note: skipped {skipped} /joint_states messages from "
          "non-matching publishers")
print(f"added {len(obs_t)} observed-state samples "
      f"(timesteps {obs_t[0]:.1f}..{obs_t[-1]:.1f})")
