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

"""End-to-end smoke test for LeRobotDatasetWriter against the installed lerobot.

Migration guard: LeRobotDataset.create() dropped the `vcodec` kwarg in
lerobot 0.6.0 (codecs are configured via RGBEncoderConfig). This exercises
the real create/add_frame/save_episode/finalize path in a tmp dir so a future
signature change fails here instead of mid-port.
"""

import json
from pathlib import Path

import numpy as np
import rosetta.robots.ros2.offline.bag_frames  # noqa: F401  register codecs
from lerobot_robot_rosetta.dataset_writer import LeRobotDatasetWriter
from rosetta.contract.schema import load_contract
from rosetta.contract.specs import (
    iter_action_specs,
    iter_observation_specs,
)

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


def test_writer_end_to_end(tmp_path: Path):
    contract_path = tmp_path / "smoke.yaml"
    contract_path.write_text(CONTRACT_YAML)
    contract = load_contract(contract_path)
    specs = list(iter_observation_specs(contract)) + list(iter_action_specs(contract))

    writer = LeRobotDatasetWriter()
    writer.open(
        contract=contract,
        specs=specs,
        repo_id="smoke/ds",
        root=tmp_path / "out",
        vcodec="libsvtav1",
    )
    for i in range(2):
        writer.add_frame(_frame(i))
    writer.save_episode()
    writer.finalize()

    ds_root = tmp_path / "out" / "smoke/ds"
    assert (ds_root / "meta" / "info.json").exists()
    assert any((ds_root / "data").rglob("*.parquet"))


def test_discarded_episode_does_not_leak(tmp_path: Path):
    """Frames buffered before a mid-episode failure must not reach the dataset."""
    contract_path = tmp_path / "smoke.yaml"
    contract_path.write_text(CONTRACT_YAML)
    contract = load_contract(contract_path)
    specs = list(iter_observation_specs(contract)) + list(iter_action_specs(contract))

    writer = LeRobotDatasetWriter()
    writer.open(contract=contract, specs=specs, repo_id="smoke/ds", root=tmp_path / "out")

    writer.add_frame(_frame(0))  # partial episode, then failure upstream
    writer.discard_episode()

    for i in range(2):
        writer.add_frame(_frame(i))
    writer.save_episode()
    writer.finalize()

    ds_root = tmp_path / "out" / "smoke/ds"
    info = json.loads((ds_root / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 2  # the discarded frame is gone


def test_push_to_hub_private_by_default(tmp_path: Path, monkeypatch):
    """Publishing a dataset publicly must be an explicit choice."""
    contract_path = tmp_path / "smoke.yaml"
    contract_path.write_text(CONTRACT_YAML)
    contract = load_contract(contract_path)
    specs = list(iter_observation_specs(contract)) + list(iter_action_specs(contract))

    writer = LeRobotDatasetWriter()
    writer.open(
        contract=contract,
        specs=specs,
        repo_id="smoke/ds",
        root=tmp_path / "out",
        push_to_hub=True,
    )
    writer.add_frame(_frame(0))
    writer.save_episode()

    pushed = {}
    monkeypatch.setattr(type(writer._ds), "push_to_hub", lambda _self, **kw: pushed.update(kw))
    writer.finalize()
    assert pushed["private"] is True
