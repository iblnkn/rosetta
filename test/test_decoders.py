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

"""Tests for the decode path (decoder + forward op pipeline).

Uses duck-typed mock messages so it needs no ROS message classes -- the
decoders read message fields by attribute access.
"""

import types

import numpy as np

# Import for the @register_decoder side effects that populate the registry.
import rosetta.ros2.decoders  # noqa: F401
from rosetta.core.contract import load_contract
from rosetta.core.contract_utils import iter_observation_specs
from rosetta.core.converters import decode_value


def _obs_spec(tmp_path, apply_block):
    yaml = f"""
robot_type: test
fps: 30
observations:
  - key: observation.state
    topic: /joint_states
    type: sensor_msgs/msg/JointState
    select: [position.j1, position.j2]
    apply: {apply_block}
actions:
  - key: action
    topic: /cmd
    type: sensor_msgs/msg/JointState
    select: [position.j1]
"""
    p = tmp_path / 'c.yaml'
    p.write_text(yaml)
    contract = load_contract(p)
    return next(iter(iter_observation_specs(contract)))


def _joint_state(names, positions):
    return types.SimpleNamespace(name=list(names), position=list(positions))


def test_decode_joint_state_selects_by_name(tmp_path):
    spec = _obs_spec(tmp_path, '[]')
    # Out-of-order names: decoder must select by joint name, not index.
    msg = _joint_state(['j2', 'j1'], [2.0, 1.0])
    out = decode_value(msg, spec)
    assert np.allclose(out, [1.0, 2.0])


def test_decode_applies_rad2deg(tmp_path):
    spec = _obs_spec(tmp_path, '[rad2deg]')
    msg = _joint_state(['j1', 'j2'], [np.pi / 2, np.pi])
    out = decode_value(msg, spec)
    assert np.allclose(out, [90.0, 180.0])


def test_decode_applies_rad2deg_then_clamp(tmp_path):
    # Ops run front-to-back on decode: rad2deg then clamp.
    spec = _obs_spec(tmp_path, '[rad2deg, {clamp: [0, 90]}]')
    msg = _joint_state(['j1', 'j2'], [np.pi / 2, np.pi])
    out = decode_value(msg, spec)
    # [90, 180] -> clamp[0,90] -> [90, 90]
    assert np.allclose(out, [90.0, 90.0])
