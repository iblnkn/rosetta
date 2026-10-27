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
from codec_helpers import require_message

from rosetta.contract.errors import ContractValidationError
from rosetta.contract.schema import load_contract
from rosetta.contract.specs import iter_observation_specs
from rosetta.frames.codecs import decode_value


def _obs_spec(
    tmp_path,
    apply_block="[]",
    *,
    msg_type="sensor_msgs/msg/JointState",
    select="[position.j1, position.j2]",
):
    select_line = f"    select: {select}\n" if select is not None else ""
    yaml = f"""
robot_type: test
robot_interface: ros2
fps: 30
observations:
  observation.state:
    channel: {{topic: /obs, type: {msg_type}}}
    align: {{strategy: hold, timeline: receive}}
{select_line}    apply: {apply_block}
actions:
  action:
    channel: {{topic: /cmd, type: sensor_msgs/msg/JointState}}
    align: {{strategy: hold, timeline: receive}}
    select: [position.j1]
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml)
    contract = load_contract(p)
    return next(iter(iter_observation_specs(contract)))


def _joint_state(names, positions):
    return types.SimpleNamespace(name=list(names), position=list(positions))


def test_decode_joint_state_selects_by_name(tmp_path):
    spec = _obs_spec(tmp_path, "[]")
    # Out-of-order names: decoder must select by joint name, not index.
    msg = _joint_state(["j2", "j1"], [2.0, 1.0])
    out = decode_value(msg, spec)
    assert np.allclose(out, [1.0, 2.0])


def test_decode_applies_rad2deg_then_clamp(tmp_path):
    # Operators run front-to-back on decode: rad2deg then clamp.
    spec = _obs_spec(tmp_path, "[rad2deg, {clamp: {min: 0, max: 90}}]")
    msg = _joint_state(["j1", "j2"], [np.pi / 2, np.pi])
    out = decode_value(msg, spec)
    # [90, 180] -> clamp[0,90] -> [90, 90]
    assert np.allclose(out, [90.0, 90.0])


# --- per-type decode behavior ------------------------------------------------


def _ns(**fields):
    return types.SimpleNamespace(**fields)


@pytest.mark.parametrize(
    ("msg_type", "select", "msg", "expected"),
    [
        (
            "sensor_msgs/msg/Imu",
            "[orientation.x, angular_velocity.z]",
            _ns(orientation=_ns(x=0.1, y=0.2, z=0.3, w=0.4), angular_velocity=_ns(x=1.0, y=2.0, z=3.0)),
            [0.1, 3.0],
        ),
        (
            "nav_msgs/msg/Odometry",
            "[pose.pose.position.x, twist.twist.angular.z]",
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


# --- requires-select: selector-driven decoders have no select-less default ---


@pytest.mark.parametrize(
    "msg_type",
    [
        "sensor_msgs/msg/JointState",
        "sensor_msgs/msg/Imu",
        "nav_msgs/msg/Odometry",
        "geometry_msgs/msg/Twist",
        "geometry_msgs/msg/TwistStamped",
        "control_msgs/msg/MultiDOFCommand",
        "trajectory_msgs/msg/JointTrajectory",
        "sensor_msgs/msg/Joy",
    ],
)
def test_selector_decoders_require_select(tmp_path, msg_type):
    # select declares a stream's width and field order; a select-less channel
    # of a selector-driven type fails at CONTRACT LOAD (requires_select
    # registration) — not per-message at ingest, where the drop would
    # silently zero-fill the stream.
    require_message(msg_type)
    with pytest.raises(ContractValidationError, match="add 'select:'"):
        _obs_spec(tmp_path, msg_type=msg_type, select=None)


# --- fail-fast on bad selectors and degenerate messages ----------------------


def test_joint_state_unknown_field_raises(tmp_path):
    spec = _obs_spec(tmp_path, select="[torque.j1]")
    with pytest.raises(ValueError, match="Unknown JointState field 'torque'"):
        decode_value(_joint_state(["j1"], [1.0]), spec)


def test_multidof_unknown_field_raises(tmp_path):
    # "velocity.j1" must raise, not be read as a DOF literally named
    # "velocity.j1".
    require_message("control_msgs/msg/MultiDOFCommand")
    spec = _obs_spec(tmp_path, msg_type="control_msgs/msg/MultiDOFCommand", select="[velocity.j1]")
    msg = types.SimpleNamespace(dof_names=["j1"], values=[1.0], values_dot=[])
    with pytest.raises(ValueError, match="Unknown MultiDOFCommand field 'velocity'"):
        decode_value(msg, spec)


def test_joint_trajectory_empty_points_raises(tmp_path):
    # An empty trajectory (a common "cancel" convention) must fail at decode,
    # where ingest drops it with message context — not poison the stream with
    # a wrong-width value that detonates later at sample time.
    spec = _obs_spec(tmp_path, msg_type="trajectory_msgs/msg/JointTrajectory", select="[position.j1]")
    msg = types.SimpleNamespace(joint_names=["j1"], points=[])
    with pytest.raises(ValueError, match="no points"):
        decode_value(msg, spec)


# --- decode_value width gate --------------------------------------------------


def test_decode_value_selectless_stream_must_be_scalar(tmp_path):
    # Built-in MultiArray decoders return the whole array; without select the
    # spec is 1-wide, so a multi-element message is refused at the choke point.
    spec = _obs_spec(tmp_path, msg_type="std_msgs/msg/Float64MultiArray", select=None)
    with pytest.raises(ValueError, match="select-less stream is a scalar"):
        decode_value(types.SimpleNamespace(data=[1.0, 2.0, 3.0]), spec)
