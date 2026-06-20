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

"""Tests for contract loading and schema validation."""

from pathlib import Path

import pytest

from rosetta.core.contract import (
    ContractValidationError,
    is_valid_lerobot_dtype,
    load_contract,
)

CONTRACTS_DIR = Path(__file__).resolve().parent.parent / 'contracts'

VALID_CONTRACT = """
robot_type: test
fps: 30
observations:
  - key: observation.state
    topic: /joint_states
    type: sensor_msgs/msg/JointState
    select: [position.j1, position.j2]
    apply: [rad2deg]
actions:
  - key: action
    topic: /cmd
    type: sensor_msgs/msg/JointState
    select: [position.j1, position.j2]
    apply: [rad2deg]
"""

# Old schema: action publish block + selector, no top-level `topic`.
OLD_SYNTAX_CONTRACT = """
robot_type: test
fps: 30
actions:
  - key: action
    type: sensor_msgs/msg/JointState
    publish:
      topic: /cmd
    selector:
      names: [position.j1]
"""


@pytest.mark.parametrize('name', ['so_101.yaml', 'so_101_hil.yaml', 'turtlebot3.yaml'])
def test_bundled_contracts_load(name):
    contract = load_contract(CONTRACTS_DIR / name)
    assert contract.robot_type
    assert contract.fps > 0
    # Every bundled contract defines at least one observation and one action.
    assert len(contract.observations) >= 1
    assert len(contract.actions) >= 1


def test_valid_contract_round_trips(tmp_path):
    p = tmp_path / 'c.yaml'
    p.write_text(VALID_CONTRACT)
    contract = load_contract(p)
    assert contract.robot_type == 'test'
    assert contract.observations[0].key == 'observation.state'
    assert contract.actions[0].key == 'action'


def test_old_syntax_action_rejected(tmp_path):
    # The new parser requires a top-level `topic` on actions; the deprecated
    # `publish:`/`selector:` form is missing it and must fail to load.
    p = tmp_path / 'old.yaml'
    p.write_text(OLD_SYNTAX_CONTRACT)
    with pytest.raises(ContractValidationError):
        load_contract(p)


@pytest.mark.parametrize(
    'dtype',
    ['float32', 'float64', 'int32', 'int64', 'bool', 'uint8', 'video', 'image', 'string'],
)
def test_valid_dtypes_accepted(dtype):
    assert is_valid_lerobot_dtype(dtype)


@pytest.mark.parametrize('dtype', ['not_a_dtype', 'flot32', ''])
def test_invalid_dtypes_rejected(dtype):
    assert not is_valid_lerobot_dtype(dtype)


def test_missing_robot_type_rejected(tmp_path):
    p = tmp_path / 'bad.yaml'
    p.write_text('fps: 30\n')
    with pytest.raises(ContractValidationError):
        load_contract(p)
