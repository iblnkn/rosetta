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

"""A mid-episode failure must discard buffered frames, not leak them.

port() feeds frames to the writer episode by episode; if iter_bag_frames
raises partway through an episode, the frames already add_frame()'d would
otherwise blend into the next successful episode (LeRobotDataset keeps its
episode buffer across the exception).
"""

from pathlib import Path
from types import SimpleNamespace

from rosetta.robots.ros2.offline import port as port_mod


class _StubWriter:
    """Records the call sequence port() drives."""

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


class _StubSource:
    def __init__(self, *_a, **_k):
        pass

    def bag_dirs(self):
        return [Path("ep_bad"), Path("ep_good")]


def _stub_iter_bag_frames(bag_dir, _specs, warmup_keys=None):
    _ = warmup_keys
    if bag_dir.name == "ep_bad":
        yield {"observation.state": [0.0]}
        raise ValueError("decode failed mid-episode")
    yield {"observation.state": [1.0]}
    yield {"observation.state": [2.0]}


def test_failed_episode_discarded_before_next(monkeypatch, tmp_path):
    writer = _StubWriter()
    monkeypatch.setattr(port_mod, "load_contract", lambda _p: SimpleNamespace(tasks=[]))
    monkeypatch.setattr(port_mod, "iter_specs", lambda _c: iter(()))
    monkeypatch.setattr(port_mod, "iter_observation_specs", lambda _c: iter(()))
    monkeypatch.setattr(port_mod, "BagFrameSource", _StubSource)
    monkeypatch.setattr(port_mod, "load_dataset_writer", lambda _b: writer)
    monkeypatch.setattr(port_mod, "iter_bag_frames", _stub_iter_bag_frames)

    port_mod.port(tmp_path, "repo", tmp_path / "contract.yaml")

    assert writer.calls == [
        "open",
        "add_frame",  # ep_bad's partial frame...
        "discard_episode",  # ...dropped, never saved
        "add_frame",
        "add_frame",
        "save_episode",  # ep_good saves cleanly
        "finalize",
    ]
