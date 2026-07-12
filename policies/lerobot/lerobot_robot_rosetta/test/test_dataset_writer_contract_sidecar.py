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

"""LeRobotDatasetWriter.open() copies the contract into meta/rosetta_contract.yaml
when embed_contract is on, and leaves the dataset untouched when it's off."""

from pathlib import Path

import rosetta.robots.ros2.offline.bag_frames  # noqa: F401  register codecs
from rosetta.contract.schema import load_contract
from rosetta.contract.specs import iter_action_specs, iter_observation_specs

from lerobot_robot_rosetta.dataset_writer import LeRobotDatasetWriter

CONTRACT_YAML = """
robot_type: smoke_bot
robot_interface: ros2
fps: 10
observations:
  observation.state:
    channel: {topic: /js, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.j1]
actions:
  action:
    channel: {topic: /cmd, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.j1]
"""


def _open_writer(tmp_path: Path, **open_kwargs) -> tuple[LeRobotDatasetWriter, Path]:
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
        contract_path=contract_path,
        **open_kwargs,
    )
    return writer, contract_path


def test_embed_contract_writes_sidecar(tmp_path: Path):
    _writer, contract_path = _open_writer(tmp_path, embed_contract=True)
    sidecar = tmp_path / "out" / "smoke/ds" / "meta" / "rosetta_contract.yaml"
    assert sidecar.exists()
    assert sidecar.read_text() == contract_path.read_text()


def test_embed_contract_false_skips_sidecar(tmp_path: Path):
    _open_writer(tmp_path, embed_contract=False)
    sidecar = tmp_path / "out" / "smoke/ds" / "meta" / "rosetta_contract.yaml"
    assert not sidecar.exists()


def test_default_embeds_contract_when_path_given(tmp_path: Path):
    """embed_contract defaults to True."""
    _open_writer(tmp_path)
    sidecar = tmp_path / "out" / "smoke/ds" / "meta" / "rosetta_contract.yaml"
    assert sidecar.exists()


def test_no_contract_path_means_no_sidecar(tmp_path: Path):
    contract_path = tmp_path / "smoke.yaml"
    contract_path.write_text(CONTRACT_YAML)
    contract = load_contract(contract_path)
    specs = list(iter_observation_specs(contract)) + list(iter_action_specs(contract))

    writer = LeRobotDatasetWriter()
    writer.open(contract=contract, specs=specs, repo_id="smoke/ds", root=tmp_path / "out")

    sidecar = tmp_path / "out" / "smoke/ds" / "meta" / "rosetta_contract.yaml"
    assert not sidecar.exists()
