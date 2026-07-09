"""Build the merge-timeline viewer (timestep x-axis) from a merge JSONL dump.

traj_viewer.html already carries the merge-timeline text; this just runs the
exporters and injects the data (plus the page title) into its
``const DATA = null;`` slot. Paths resolve against this file's directory, so
run it from anywhere.

Usage:
    python build_merge_timeline.py <merge_dump.jsonl> <out.html> [title] [bag]
"""
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent

dump, out = sys.argv[1], sys.argv[2]
title = sys.argv[3] if len(sys.argv) > 3 else "merge run"
bag = sys.argv[4] if len(sys.argv) > 4 else None  # optional: overlay /joint_states

# 1. dump -> timestep-axis traj JSON (chunks carry ndrop)
data_path = out + ".data.json"
rc = subprocess.run(
    [sys.executable, str(HERE / "export_merge_to_traj.py"), dump, data_path]
).returncode
if rc:
    sys.exit(rc)  # the exporter already printed why

# 1b. optional: add observed state (/joint_states) on the timestep axis
if bag:
    rc = subprocess.run(
        [sys.executable, str(HERE / "extract_obs_to_traj.py"), bag, dump, data_path]
    ).returncode
    if rc:
        sys.exit(rc)

# 2. inject the data; the title rides along inside DATA
data = json.load(open(data_path))
data["title"] = title
# <-escape so a "</script>" inside any string (title, joint name) cannot
# terminate the inline <script> element; str.replace never interprets escapes
# in the payload, and exact-match on the sentinel cannot over-match.
data_js = json.dumps(data).replace("<", "\\u003c")
sentinel = "const DATA = null;"
tmpl = (HERE / "traj_viewer.html").read_text()
if tmpl.count(sentinel) != 1:
    sys.exit(f"error: traj_viewer.html must contain exactly one {sentinel!r} "
             f"slot; found {tmpl.count(sentinel)}")
h = tmpl.replace(sentinel, "const DATA = " + data_js + ";")

open(out, "w").write(h)
print(f"wrote {out} ({len(h)} bytes)")
