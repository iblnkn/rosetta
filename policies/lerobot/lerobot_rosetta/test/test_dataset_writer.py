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

"""LeRobotDatasetWriter against the installed lerobot.

Runs the real create/add_frame/save_episode/finalize path in a tmp dir, so a
LeRobotDataset signature change fails here instead of mid-port, and pins the
writer's own guarantees: the contract sidecar, discarded episodes, private uploads.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_rosetta.dataset_writer import LeRobotDatasetWriter
from rosetta.contract.schema import load_contract
from rosetta.frames.codecs import discover_codecs

discover_codecs()  # register the built-in codecs

CONTRACT_YAML = """
robot_type: smoke_bot
robot_interface: ros2
fps: 10
observations:
  observation.images.cam:
    channel: {topic: /cam, type: sensor_msgs/msg/CompressedImage}
    align: {strategy: hold, timeline: receive}
    apply: [{resize: [32, 32]}]
  observation.state:
    channel: {topic: /js, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.j1, position.j2]
actions:
  action:
    channel: {topic: /cmd, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.j1, position.j2]
"""


@pytest.fixture
def contract_path(tmp_path: Path) -> Path:
    path = tmp_path / "smoke.yaml"
    path.write_text(CONTRACT_YAML)
    return path


@pytest.fixture
def ds_root(tmp_path: Path) -> Path:
    return tmp_path / "out" / "smoke/ds"


def _open_writer(yaml_path: Path, **open_kwargs) -> LeRobotDatasetWriter:
    writer = LeRobotDatasetWriter()
    writer.open(
        contract=load_contract(yaml_path),
        repo_id="smoke/ds",
        root=yaml_path.parent / "out",
        **open_kwargs,
    )
    return writer


def _frame(i: int) -> dict:
    return {
        "observation.images.cam": np.full((32, 32, 3), i * 10, dtype=np.uint8),
        # dtype must match the declared features (JointState resolves to
        # float64) — exactly what FrameLayout.assemble emits when porting.
        "observation.state": np.array([0.1 * i, 0.2 * i], dtype=np.float64),
        "action": np.array([1.0 * i, 2.0 * i], dtype=np.float64),
        "is_first": np.array([i == 0], dtype=bool),
        "is_last": np.array([i == 1], dtype=bool),
        "is_terminal": np.array([i == 1], dtype=bool),
        "task": "smoke",
    }


@pytest.mark.parametrize("streaming_encoding", [False, True])
def test_writer_end_to_end(contract_path: Path, ds_root: Path, streaming_encoding: bool):
    writer = _open_writer(contract_path, vcodec="libsvtav1", streaming_encoding=streaming_encoding)
    for i in range(2):
        writer.add_frame(_frame(i))
    writer.save_episode()
    writer.finalize()

    assert (ds_root / "meta" / "info.json").exists()
    assert any((ds_root / "data").rglob("*.parquet"))
    assert any((ds_root / "videos").rglob("*.mp4"))


def test_discarded_episode_does_not_leak(contract_path: Path, ds_root: Path):
    """Frames buffered before a mid-episode failure must not reach the dataset."""
    writer = _open_writer(contract_path)

    writer.add_frame(_frame(0))  # partial episode, then failure upstream
    writer.discard_episode()

    for i in range(2):
        writer.add_frame(_frame(i))
    writer.save_episode()
    writer.finalize()

    info = json.loads((ds_root / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 2  # the discarded frame is gone


def test_push_to_hub_private_by_default(contract_path: Path, monkeypatch):
    """Publishing a dataset publicly must be an explicit choice."""
    pushed = {}
    monkeypatch.setattr(LeRobotDataset, "push_to_hub", lambda _self, **kw: pushed.update(kw))
    writer = _open_writer(contract_path, push_to_hub=True)
    writer.add_frame(_frame(0))
    writer.save_episode()
    writer.finalize()
    assert pushed["private"] is True


@pytest.mark.parametrize("open_kwargs", [{"embed_contract": True}, {}], ids=["embed_contract", "default"])
def test_embed_contract_writes_sidecar(contract_path: Path, ds_root: Path, open_kwargs):
    _open_writer(contract_path, contract_path=contract_path, **open_kwargs)
    assert (ds_root / "meta" / "rosetta_contract.yaml").read_text() == contract_path.read_text()


@pytest.mark.parametrize(
    "open_kwargs",
    [{"embed_contract": False}, {"contract_path": None}],
    ids=["embed_contract=false", "no_contract_path"],
)
def test_no_sidecar_without_contract_or_opt_in(contract_path: Path, ds_root: Path, open_kwargs):
    _open_writer(contract_path, **{"contract_path": contract_path, **open_kwargs})
    assert not (ds_root / "meta" / "rosetta_contract.yaml").exists()
