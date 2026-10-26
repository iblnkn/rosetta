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

"""write_dataset's per-episode failure isolation.

A mid-episode failure must discard buffered frames, not leak them: frames
already add_frame()'d would otherwise blend into the next successful episode
(LeRobotDataset keeps its episode buffer across the exception). And a port
where EVERY episode failed must raise without finalizing — an empty dataset
is a failure, not a product.

No monkeypatching: write_dataset takes its writer and episodes as arguments,
so the tests hand it a recording fake and literal frame generators.
"""

import pytest

from rosetta.robots.ros2.offline.port import write_dataset


class _StubWriter:
    """Records the call sequence write_dataset drives."""

    def __init__(self):
        self.calls: list[str] = []

    def open(self, **_kw):
        self.calls.append("open")

    def add_frame(self, _frame):
        self.calls.append("add_frame")

    def save_episode(self):
        self.calls.append("save_episode")

    def discard_episode(self):
        self.calls.append("discard_episode")

    def finalize(self):
        self.calls.append("finalize")


def _failing_frames():
    yield {"observation.state": [0.0]}
    raise ValueError("decode failed mid-episode")


def _good_frames():
    yield {"observation.state": [1.0]}
    yield {"observation.state": [2.0]}


def test_failed_episode_discarded_before_next(tmp_path):
    writer = _StubWriter()

    write_dataset(
        writer,
        [("ep_bad", _failing_frames()), ("ep_good", _good_frames())],
        contract=None,  # opaque to the stub writer
        repo_id="repo",
        contract_path=tmp_path / "contract.yaml",
    )

    assert writer.calls == [
        "open",
        "add_frame",  # ep_bad's partial frame...
        "discard_episode",  # ...dropped, never saved
        "add_frame",
        "add_frame",
        "save_episode",  # ep_good saves cleanly
        "finalize",
    ]


def test_all_episodes_failed_raises_without_finalize(tmp_path):
    """An all-failed port is an error, and the writer must not finalize an
    empty dataset as if it were a product."""
    writer = _StubWriter()

    with pytest.raises(RuntimeError, match="All 2 bags failed"):
        write_dataset(
            writer,
            [("ep_bad_1", _failing_frames()), ("ep_bad_2", _failing_frames())],
            contract=None,
            repo_id="repo",
            contract_path=tmp_path / "contract.yaml",
        )

    assert "finalize" not in writer.calls
    assert writer.calls.count("discard_episode") == 2


def test_failed_episode_iterator_closed_before_next_episode(tmp_path):
    """A writer failure leaves the episode's generator suspended at a yield —
    holding its bag reader open. write_dataset must close it at failure time,
    not leave it to end-of-run GC (write_dataset holds the episodes list, so
    every failed bag's file handle would otherwise stay open for the run).

    The probe episode records whether the failed one was already closed when
    the NEXT episode starts — end-of-function GC can't fake that.
    """
    state = {"suspended_closed": False, "closed_before_next": None}

    def _suspended_frames():
        try:
            yield {"observation.state": [0.0]}
            yield {"observation.state": [1.0]}  # never reached: writer raises first
        finally:
            state["suspended_closed"] = True

    def _probe_frames():
        state["closed_before_next"] = state["suspended_closed"]
        yield {"observation.state": [2.0]}

    class _FirstFrameBomb(_StubWriter):
        def add_frame(self, frame):
            super().add_frame(frame)
            if self.calls.count("add_frame") == 1:
                raise ValueError("writer rejected frame")

    write_dataset(
        _FirstFrameBomb(),
        [("ep_suspended", _suspended_frames()), ("ep_probe", _probe_frames())],
        contract=None,
        repo_id="repo",
        contract_path=tmp_path / "contract.yaml",
    )

    assert state["closed_before_next"] is True
