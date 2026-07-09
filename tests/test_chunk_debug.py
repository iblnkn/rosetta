"""Per-goal merge-dump semantics: one writer = one file = one goal.

Uses duck-typed actions (the tensor chain _ts_pose expects) so the tests
run without torch, matching the rest of the ROS-free suite.
"""

import json
import os

from rosetta.common.chunk_debug import MergeDumpWriter, new_dump_path


class FakeTensor:
    def __init__(self, vals):
        self._vals = [float(v) for v in vals]

    def detach(self):
        return self

    def to(self, _device):
        return self

    def flatten(self):
        return self

    def tolist(self):
        return list(self._vals)


class FakeAction:
    def __init__(self, ts, vals):
        self._ts = ts
        self._tensor = FakeTensor(vals)

    def get_timestep(self):
        return self._ts

    def get_action(self):
        return self._tensor


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_writer_header_then_events_with_per_file_counter(tmp_path):
    path = tmp_path / "merge_a.jsonl"
    w = MergeDumpWriter(str(path), header={"task": "toss", "fps": 30})
    for latest in (1, 3):
        w.dump(
            latest,
            existing={5: FakeAction(5, [0.5, 1.5])},
            incoming={6: FakeAction(6, [1.5, 2.5])},
            merged={
                5: FakeAction(5, [0.5, 1.5]),
                6: FakeAction(6, [1.5, 2.5]),
            },
            incoming_full=[FakeAction(4, [-1.0, 0.0]), FakeAction(6, [1.5, 2.5])],
        )
    records = _read_jsonl(path)
    assert records[0] == {"header": {"task": "toss", "fps": 30}}
    assert [r["event"] for r in records[1:]] == [0, 1]
    assert records[1]["latest_action"] == 1 and records[2]["latest_action"] == 3
    assert records[1]["existing"] == {"5": [0.5, 1.5]}
    assert records[1]["incoming_full"] == {"4": [-1.0, 0.0], "6": [1.5, 2.5]}
    assert "wall_time" in records[1]


def test_second_writer_starts_a_fresh_file_and_counter(tmp_path):
    # Two goals: independent files, both counters starting at 0.
    a = MergeDumpWriter(str(tmp_path / "a.jsonl"))
    b = MergeDumpWriter(str(tmp_path / "b.jsonl"))
    event = dict(
        existing={},
        incoming={1: FakeAction(1, [0.0])},
        merged={1: FakeAction(1, [0.0])},
        incoming_full=[FakeAction(1, [0.0])],
    )
    a.dump(0, **event)
    a.dump(0, **event)
    b.dump(0, **event)
    assert [r["event"] for r in _read_jsonl(tmp_path / "a.jsonl")] == [0, 1]
    assert [r["event"] for r in _read_jsonl(tmp_path / "b.jsonl")] == [0]


def test_new_dump_path_slugs_task_and_never_collides(tmp_path):
    p1 = new_dump_path(str(tmp_path / "dumps"), "Pick & place the cup!")
    assert "Pick-place-the-cup" in p1 and p1.endswith(".jsonl")
    # The name is claimed atomically (file pre-created): PIDs can coincide
    # across container PID namespaces, so O_EXCL is the real guarantee.
    assert os.path.exists(p1)
    p2 = new_dump_path(str(tmp_path / "dumps"), "Pick & place the cup!")
    assert p2 != p1

    # Empty/garbage tasks still produce a usable name.
    p3 = new_dump_path(str(tmp_path / "dumps"), "  ??? ")
    assert "_task" in p3
