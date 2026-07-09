"""Shared loader for per-goal merge-event JSONL dumps.

A dump starts with a ``{"header": {...}}`` provenance record (task, policy,
contract, chunk config, action_features) followed by one merge-event record
per line. Both exporters go through this module so the header convention
lives in one place.
"""
import json


def load_merge_dump(path):
    """Return ``(header_or_None, events)`` for a merge-dump JSONL file."""
    header, events = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "header" in rec:
                if header is None:
                    header = rec["header"]
                continue
            events.append(rec)
    return header, events


def joint_names(header, dim):
    """Joint labels for a ``dim``-wide dump.

    The header's ``action_features`` is the true pose-vector order; fall
    back to ``dim0..dimN`` when it is absent (the node failed to resolve it).
    """
    names = (header or {}).get("action_features") or None
    if names and len(names) == dim:
        return [str(n) for n in names]
    return [f"dim{i}" for i in range(dim)]
