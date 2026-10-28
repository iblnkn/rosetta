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

"""The ros2 decode path: built-in decoders, selection, and the forward operator pipeline.

Specs come from real contracts (contract load needs the message packages);
messages are duck-typed, since the decoders read fields by attribute access.
"""

import types

import numpy as np
import pytest
from codec_helpers import load_specs, require_message

from rosetta.contract.errors import ContractValidationError
from rosetta.frames.codecs import decode_value


def _obs_spec(
    tmp_path,
    apply_block="[]",
    *,
    msg_type="sensor_msgs/msg/JointState",
    select=("position.j1", "position.j2"),
):
    """Observation spec; ``select=None`` leaves select out."""
    return load_specs(tmp_path, obs_type=msg_type, obs_select=select, obs_apply=apply_block)[1]


def _joint_state(names, positions):
    return types.SimpleNamespace(name=list(names), position=list(positions))


def test_decode_joint_state_selects_by_name(tmp_path):
    spec = _obs_spec(tmp_path, "[]")
    # Out-of-order names: decoder must select by joint name, not index.
    msg = _joint_state(["j2", "j1"], [2.0, 1.0])
    out = decode_value(msg, spec)
    assert np.allclose(out, [1.0, 2.0])


# --- per-type decode behavior ------------------------------------------------


def _ns(**fields):
    return types.SimpleNamespace(**fields)


@pytest.mark.parametrize(
    ("msg_type", "select", "msg", "expected"),
    [
        (
            "sensor_msgs/msg/Imu",
            ["orientation.x", "angular_velocity.z"],
            _ns(orientation=_ns(x=0.1, y=0.2, z=0.3, w=0.4), angular_velocity=_ns(x=1.0, y=2.0, z=3.0)),
            [0.1, 3.0],
        ),
        (
            "nav_msgs/msg/Odometry",
            ["pose.pose.position.x", "twist.twist.angular.z"],
            _ns(pose=_ns(pose=_ns(position=_ns(x=1.5))), twist=_ns(twist=_ns(angular=_ns(z=-0.5)))),
            [1.5, -0.5],
        ),
    ],
    ids=["imu", "odometry"],
)
def test_decode_selects_dotted_paths(tmp_path, msg_type, select, msg, expected):
    spec = _obs_spec(tmp_path, msg_type=msg_type, select=select)
    assert np.allclose(decode_value(msg, spec), expected)


@pytest.mark.parametrize(
    ("msg_type", "data", "expected_dtype"),
    [
        ("std_msgs/msg/Float32", 1.25, np.float32),
        ("std_msgs/msg/Float64", 1.5, np.float64),
        ("std_msgs/msg/Int32", 7, np.int32),
        ("std_msgs/msg/Int64", 9, np.int64),
        ("std_msgs/msg/Bool", True, np.bool_),
    ],
)
def test_decode_scalar_types(tmp_path, msg_type, data, expected_dtype):
    spec = _obs_spec(tmp_path, msg_type=msg_type, select=None)
    out = decode_value(types.SimpleNamespace(data=data), spec)
    assert out.shape == (1,)
    assert out.dtype == expected_dtype
    assert out[0] == data


def test_decode_string(tmp_path):
    spec = _obs_spec(tmp_path, msg_type="std_msgs/msg/String", select=None)
    assert decode_value(types.SimpleNamespace(data="pick up the cube"), spec) == "pick up the cube"


# --- requires-select: selector-driven codecs have no select-less default ---

_SELECTOR_DECODED = [
    "sensor_msgs/msg/JointState",
    "sensor_msgs/msg/Imu",
    "nav_msgs/msg/Odometry",
    "geometry_msgs/msg/Twist",
    "geometry_msgs/msg/TwistStamped",
    "control_msgs/msg/MultiDOFCommand",
    "trajectory_msgs/msg/JointTrajectory",
    "sensor_msgs/msg/Joy",
]
_SELECTOR_ENCODED = [t for t in _SELECTOR_DECODED if t not in ("sensor_msgs/msg/Imu", "nav_msgs/msg/Odometry")]


@pytest.mark.parametrize(
    ("section", "msg_type"),
    [("obs", t) for t in _SELECTOR_DECODED] + [("act", t) for t in _SELECTOR_ENCODED],
)
def test_selector_codecs_require_select(tmp_path, section, msg_type):
    # select declares a stream's width and field order; a select-less channel
    # of a selector-driven type fails at CONTRACT LOAD (requires_select
    # registration), not per-message at ingest, where the drop would
    # silently zero-fill the stream, nor at first publish.
    require_message(msg_type)
    with pytest.raises(ContractValidationError, match="add 'select:'"):
        load_specs(tmp_path, **{f"{section}_type": msg_type, f"{section}_select": None})


# --- fail-fast on bad selectors and degenerate messages ----------------------


def test_joint_state_unknown_field_raises(tmp_path):
    spec = _obs_spec(tmp_path, select=["torque.j1"])
    with pytest.raises(ValueError, match="Unknown JointState field 'torque'"):
        decode_value(_joint_state(["j1"], [1.0]), spec)


def test_multidof_unknown_field_raises(tmp_path):
    # "velocity.j1" must raise, not be read as a DOF literally named
    # "velocity.j1".
    require_message("control_msgs/msg/MultiDOFCommand")
    spec = _obs_spec(tmp_path, msg_type="control_msgs/msg/MultiDOFCommand", select=["velocity.j1"])
    msg = types.SimpleNamespace(dof_names=["j1"], values=[1.0], values_dot=[])
    with pytest.raises(ValueError, match="Unknown MultiDOFCommand field 'velocity'"):
        decode_value(msg, spec)


def test_joint_trajectory_empty_points_raises(tmp_path):
    # An empty trajectory (a common "cancel" convention) must fail at decode,
    # where ingest drops it with message context — not poison the stream with
    # a wrong-width value that detonates later at sample time.
    spec = _obs_spec(tmp_path, msg_type="trajectory_msgs/msg/JointTrajectory", select=["position.j1"])
    msg = types.SimpleNamespace(joint_names=["j1"], points=[])
    with pytest.raises(ValueError, match="no points"):
        decode_value(msg, spec)
