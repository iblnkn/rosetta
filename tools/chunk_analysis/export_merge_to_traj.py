"""Convert a merge-event JSONL dump into the trajectory-viewer JSON format,
but on the TIMESTEP axis (the grid the merge uses) instead of received time.

  - chunks[k].t = the incoming chunk's integer timesteps for merge event k
                  (exactly the points used during that merge; dropped prefix
                  already excluded by the client filter)
  - cmd_t / cmd_p = the realized executed command, reconstructed from the
                  merged queue: each timestep's value = the merged value from
                  the last event that still held it as a future action (i.e.
                  the final blended value right before it was consumed).
"""
import json
import sys

from merge_dump_io import joint_names, load_merge_dump

dump = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "traj_data_merge.json"

header, events = load_merge_dump(dump)
if not events:
    kind = "header only" if header else "no records"
    print(
        f"error: {dump} has no merge events ({kind}); nothing to export "
        "(goal failed/cancelled before the first chunk arrived?)",
        file=sys.stderr,
    )
    sys.exit(2)

# Reconstruct the executed command on the timestep grid. merged always holds
# future timesteps (> latest_action); processing events in order and letting
# later events overwrite yields each timestep's final value before consumption.
executed = {}
for e in events:
    for ts, pose in e["merged"].items():
        executed[int(ts)] = pose

cmd_ts = sorted(executed)
if not cmd_ts:
    print(
        f"error: {dump} events carry no future actions; nothing to export",
        file=sys.stderr,
    )
    sys.exit(2)
cmd_t = [float(ts) for ts in cmd_ts]
cmd_p = [executed[ts] for ts in cmd_ts]

# Each merge event's chunk, anchored at its true merge timesteps. When the dump
# has the full pre-drop chunk (incoming_full), start at t_observation (i_0) and
# tag how many leading points are the dropped already-passed prefix (timesteps
# <= latest_action). Otherwise fall back to the post-drop incoming (ndrop=0).
chunks = []
for e in events:
    src = e.get("incoming_full") or e["incoming"]
    ks = sorted(int(k) for k in src)
    if not ks:
        continue
    # Dropped = full chunk minus what actually entered the queue (the kept set is
    # always a suffix of the full chunk). This counts both the already-passed
    # prefix and any extra drop_chunk_prefix, so it stays correct for any K.
    if e.get("incoming_full"):
        ndrop = len(ks) - len(e["incoming"])
    else:
        ndrop = sum(1 for ts in ks if ts <= e["latest_action"])
    chunks.append({
        "t": [float(ts) for ts in ks],
        "p": [src[str(ts)] for ts in ks],
        "ndrop": ndrop,
    })

data = {
    "joints": joint_names(header, len(cmd_p[0])),
    "cmd_t": cmd_t,
    "cmd_p": cmd_p,
    "chunks": chunks,
}
with open(out, "w") as f:
    json.dump(data, f)
print(f"wrote {out}: {len(cmd_t)} executed timesteps "
      f"({cmd_ts[0]}..{cmd_ts[-1]}), {len(chunks)} chunks")
