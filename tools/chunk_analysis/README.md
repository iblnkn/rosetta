# Merge-timeline viewer

Builds the interactive "Merge timeline (timestep axis)" HTML — chunks on the
merge-timestep grid, per joint: black = realized executed command
(reconstructed from the merged queue), color = each chunk anchored at
t_observation, red dashed = dropped already-passed prefix, cyan = observed
state, plus a merge scrubber to isolate the chunks live in the queue at one
instant.

## Producing the input

The dump comes from the runtime hook in `rosetta/common/chunk_debug.py`:
set the client node's `merge_dump_dir` param to a directory, and every
RunPolicy goal writes its own file there — `merge_<stamp>_<pid>_<task-slug>.jsonl`,
one goal = one episode = one dump. The first line is a `{"header": ...}`
provenance record (task, policy, contract, chunk config, and
`action_features` — the joint order of every pose vector); each following
line is one action-queue merge (`existing` / `incoming` / `merged` per
timestep, plus the full pre-drop chunk). Optionally record `/joint_states`
in a bag during the same run for the observed-state overlay.

## Building the viewer

Runs from anywhere — helpers and the template resolve against the script's
own directory:

```bash
python3 build_merge_timeline.py <dump.jsonl> <out.html> [title] [bag]
```

- `title` — heading suffix (e.g. `"plating_enc (right arm)"`)
- `bag` — optional: overlay `/joint_states` from a rosbag on the timestep
  axis (needs a ROS-sourced shell; without a bag, plain Python). Joints map
  by name-matching the header's `action_features`; on any mismatch the
  overlay is skipped loudly instead of plotted wrong.

## Files

- `build_merge_timeline.py` — entry point; runs the two exporters and
  injects the data (title included) into the template's
  `const DATA = null;` slot.
- `merge_dump_io.py` — shared loader: splits the header record from events,
  derives joint labels from the header's `action_features` (fallback
  `dim0..N`).
- `export_merge_to_traj.py` — dump → timestep-axis viewer JSON (chunks with
  dropped-prefix counts + the realized executed command). Errors out clearly
  on header-only dumps (goal died before the first chunk).
- `extract_obs_to_traj.py` — adds `/joint_states` from a bag onto the
  timestep axis (linear wall-time→timestep fit against the dump; falls back
  to header fps when only one merge event exists).
- `traj_viewer.html` — the merge-timeline page itself (headings, legend,
  scrubber); ships with `const DATA = null;` and renders once a builder
  injects data.

Generated HTML is self-contained — open locally or publish as an artifact.
