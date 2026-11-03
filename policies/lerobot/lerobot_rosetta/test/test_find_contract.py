# Copyright 2026 Isaac Blankenau
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

"""The LeRobot runner finds a checkpoint's contract through its training dataset.

The chase itself (each link local or on the Hub, every miss a warning) is
covered in lerobot_dataset_rosetta's test_contract_fallback_chase.py. This pins
that the runner is a PolicyRunner and hands the chase the checkpoint unchanged.
"""

import json

from lerobot_dataset_rosetta.sidecar import LEROBOT_CONTRACT_SIDECAR_PATH

from lerobot_rosetta.policy_runner import LeRobotPolicyRunner
from rosetta.policies import PolicyRunner


def test_runner_implements_the_protocol():
    assert isinstance(LeRobotPolicyRunner(), PolicyRunner)


def test_find_contract_follows_the_checkpoint_to_its_dataset(tmp_path):
    sidecar = tmp_path / "dataset" / LEROBOT_CONTRACT_SIDECAR_PATH
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("robot_type: x\n")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "train_config.json").write_text(json.dumps({"dataset": {"root": str(tmp_path / "dataset")}}))

    warnings = []
    assert LeRobotPolicyRunner().find_contract(str(checkpoint), warn=warnings.append) == sidecar
    assert warnings == []


def test_find_contract_warns_and_returns_none_without_a_train_config(tmp_path):
    warnings = []
    assert LeRobotPolicyRunner().find_contract(str(tmp_path), warn=warnings.append) is None
    assert any("train_config.json not found" in w for w in warnings), warnings
