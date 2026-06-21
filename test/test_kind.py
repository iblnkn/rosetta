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

"""Tests for the ``kind`` semantic tag and its load-time validation."""

import warnings

import pytest

import rosetta.ros2.bag_frames  # noqa: F401  register decoders
from rosetta.core.contract import (
    Contract,
    ContractValidationError,
    ObservationSpec,
    _validate_kind,
)
from rosetta.core.contract_utils import iter_observation_specs


def _contract(obs):
    return Contract(
        robot_type='r', fps=30, observations=obs, actions=[], tasks=[],
        recording={}, adjunct=[], rewards=[], signals=[], info=[],
        complementary_data=[],
    )


# _validate_kind returns (representation, absolute)

def test_default_is_continuous_absolute():
    assert _validate_kind(None, ['a', 'b'], 'ctx') == ('continuous', True)


def test_invalid_kind_errors():
    with pytest.raises(ContractValidationError):
        _validate_kind('rotation_matrix', ['a'], 'ctx')


def test_dim_count_enforced():
    assert _validate_kind('quaternion', ['x', 'y', 'z', 'w'], 'ctx') == ('quaternion', True)
    assert _validate_kind('euler_rpy', ['r', 'p', 'y'], 'ctx') == ('euler_rpy', True)
    assert _validate_kind('rotation_6d', list('abcdef'), 'ctx') == ('rotation_6d', True)
    with pytest.raises(ContractValidationError):
        _validate_kind('quaternion', ['x', 'y', 'z'], 'ctx')  # 3 != 4
    with pytest.raises(ContractValidationError):
        _validate_kind('euler_rpy', ['x', 'y', 'z', 'w'], 'ctx')  # 4 != 3


def test_binary_and_continuous_any_dim():
    assert _validate_kind('binary', ['g'], 'ctx') == ('binary', True)
    assert _validate_kind('continuous', list('abcdefghij'), 'ctx') == ('continuous', True)


def test_frame_axis_absolute_vs_delta():
    # a lone frame tag gives continuous plus that frame
    assert _validate_kind('delta', ['a', 'b'], 'ctx') == ('continuous', False)
    assert _validate_kind('absolute', ['a'], 'ctx') == ('continuous', True)
    # compound: representation plus frame, in any order
    assert _validate_kind(['quaternion', 'delta'], ['x', 'y', 'z', 'w'], 'ctx') == ('quaternion', False)
    assert _validate_kind(['delta', 'binary'], ['g'], 'ctx') == ('binary', False)
    assert _validate_kind(['continuous', 'absolute'], ['a'], 'ctx') == ('continuous', True)


def test_compound_kind_errors():
    with pytest.raises(ContractValidationError):  # two representations
        _validate_kind(['quaternion', 'binary'], ['x', 'y', 'z', 'w'], 'ctx')
    with pytest.raises(ContractValidationError):  # two frames
        _validate_kind(['absolute', 'delta'], ['a'], 'ctx')
    with pytest.raises(ContractValidationError):  # unknown tag
        _validate_kind(['quaternion', 'bogus'], ['x', 'y', 'z', 'w'], 'ctx')


def test_rotation_smell_warns_when_untagged():
    with pytest.warns(UserWarning):
        _validate_kind('continuous', ['orientation.x', 'orientation.y', 'orientation.z', 'orientation.w'], 'ctx')
    with pytest.warns(UserWarning):
        _validate_kind(None, ['pose.quat_w'], 'ctx')


def test_no_warn_for_plain_joints():
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # any warning becomes a failure
        _validate_kind('continuous', ['position.j1', 'position.j2', 'position.j3'], 'ctx')


def test_no_warn_when_tagged_quaternion():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _validate_kind('quaternion', ['orientation.x', 'orientation.y', 'orientation.z', 'orientation.w'], 'ctx')


# propagation to resolved StreamSpec

def test_kind_propagates_to_stream_spec():
    obs = ObservationSpec(
        key='observation.state', topic='/imu', type='sensor_msgs/msg/Imu',
        select=['orientation.x', 'orientation.y', 'orientation.z', 'orientation.w'],
        kind='quaternion', absolute=False,
    )
    specs = list(iter_observation_specs(_contract([obs])))
    assert specs[0].kind == 'quaternion'
    assert specs[0].absolute is False


def test_default_kind_on_stream_spec():
    obs = ObservationSpec(
        key='observation.state', topic='/joints', type='sensor_msgs/msg/JointState',
        select=['position.j1', 'position.j2'],
    )
    specs = list(iter_observation_specs(_contract([obs])))
    assert specs[0].kind == 'continuous'
    assert specs[0].absolute is True
