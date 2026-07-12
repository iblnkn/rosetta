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

import rosetta.robots.ros2.offline.bag_frames  # noqa: F401  register decoders
from rosetta.contract.errors import ContractValidationError
from rosetta.contract.schema import (
    Align,
    Channel,
    Contract,
    FrameEntry,
    Source,
    _validate_kind,
)
from rosetta.contract.specs import iter_observation_specs


def _contract(obs):
    return Contract(
        robot_type="r",
        robot_interface="ros2",
        fps=30,
        observations=obs,
        actions=[],
        tasks=[],
        adjunct=[],
        rewards=[],
        signals=[],
        info=[],
        complementary_data=[],
    )


# _validate_kind returns the representation


def test_default_is_continuous():
    assert _validate_kind(None, ["a", "b"], "ctx") == "continuous"


def test_invalid_kind_errors():
    with pytest.raises(ContractValidationError):
        _validate_kind("rotation_matrix", ["a"], "ctx")


def test_dim_count_enforced():
    assert _validate_kind("quaternion", ["x", "y", "z", "w"], "ctx") == "quaternion"
    assert _validate_kind("euler_rpy", ["r", "p", "y"], "ctx") == "euler_rpy"
    assert _validate_kind("rotation_6d", list("abcdef"), "ctx") == "rotation_6d"
    with pytest.raises(ContractValidationError):
        _validate_kind("quaternion", ["x", "y", "z"], "ctx")  # 3 != 4
    with pytest.raises(ContractValidationError):
        _validate_kind("euler_rpy", ["x", "y", "z", "w"], "ctx")  # 4 != 3


def test_binary_and_continuous_any_dim():
    assert _validate_kind("binary", ["g"], "ctx") == "binary"
    assert _validate_kind("continuous", list("abcdefghij"), "ctx") == "continuous"


def test_rotation_smell_warns_when_untagged():
    with pytest.warns(UserWarning):
        _validate_kind("continuous", ["orientation.x", "orientation.y", "orientation.z", "orientation.w"], "ctx")
    with pytest.warns(UserWarning):
        _validate_kind(None, ["pose.quat_w"], "ctx")


def test_no_warn_for_plain_joints():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a failure
        _validate_kind("continuous", ["position.j1", "position.j2", "position.j3"], "ctx")


def test_no_warn_when_tagged_quaternion():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _validate_kind("quaternion", ["orientation.x", "orientation.y", "orientation.z", "orientation.w"], "ctx")


# propagation to resolved StreamSpec


def test_kind_propagates_to_stream_spec():
    entry = FrameEntry(
        key="observation.state",
        sources=(
            Source(
                channel=Channel(topic="/imu", type="sensor_msgs/msg/Imu"),
                align=Align("hold", "receive"),
                select=["orientation.x", "orientation.y", "orientation.z", "orientation.w"],
                kind="quaternion",
            ),
        ),
    )
    specs = list(iter_observation_specs(_contract([entry])))
    assert specs[0].source.kind == "quaternion"


def test_default_kind_on_stream_spec():
    entry = FrameEntry(
        key="observation.state",
        sources=(
            Source(
                channel=Channel(topic="/joints", type="sensor_msgs/msg/JointState"),
                align=Align("hold", "receive"),
                select=["position.j1", "position.j2"],
            ),
        ),
    )
    specs = list(iter_observation_specs(_contract([entry])))
    assert specs[0].source.kind == "continuous"
