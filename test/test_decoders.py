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

"""Tests for the decode path (decoder + forward operator pipeline).

Uses duck-typed mock messages so it needs no ROS message classes -- the
decoders read message fields by attribute access.
"""

import types

import numpy as np
import pytest
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


def _require_msg_pkg(msg_type):
    """Contract load introspects timelines via rosidl, so it needs the message package."""
    from rosidl_runtime_py.utilities import get_message

    try:
        get_message(msg_type)
    except Exception:
        pytest.skip(f"{msg_type} message package unavailable")


def test_decode_joint_state_selects_by_name(tmp_path):
    spec = _obs_spec(tmp_path, "[]")
    # Out-of-order names: decoder must select by joint name, not index.
    msg = _joint_state(["j2", "j1"], [2.0, 1.0])
    out = decode_value(msg, spec)
    assert np.allclose(out, [1.0, 2.0])


def test_decode_applies_rad2deg(tmp_path):
    spec = _obs_spec(tmp_path, "[rad2deg]")
    msg = _joint_state(["j1", "j2"], [np.pi / 2, np.pi])
    out = decode_value(msg, spec)
    assert np.allclose(out, [90.0, 180.0])


def test_decode_applies_rad2deg_then_clamp(tmp_path):
    # Operators run front-to-back on decode: rad2deg then clamp.
    spec = _obs_spec(tmp_path, "[rad2deg, {clamp: {min: 0, max: 90}}]")
    msg = _joint_state(["j1", "j2"], [np.pi / 2, np.pi])
    out = decode_value(msg, spec)
    # [90, 180] -> clamp[0,90] -> [90, 90]
    assert np.allclose(out, [90.0, 90.0])


def _str_decoder(msg, spec):
    """Custom decoder that lies about its dtype (declared float64, returns str)."""
    return "not an array"


@pytest.mark.parametrize(
    ("selector", "expected"),
    [("axes.0", ("axes", 0)), ("buttons.3", ("buttons", 3)), ("2", ("axes", 2))],
)
def test_joy_selector_parses(selector, expected):
    from rosetta.robots.ros2.field_access import parse_joy_selector

    assert parse_joy_selector(selector) == expected


@pytest.mark.parametrize(
    ("selector", "match"),
    [
        ("axes.-1", "non-negative"),  # Python wraparound would silently read the wrong element
        ("axes.x", "must be an integer"),
        ("triggers.0", "Unknown Joy field"),
    ],
)
def test_joy_selector_rejects_bad_input(selector, match):
    from rosetta.robots.ros2.field_access import parse_joy_selector

    with pytest.raises(ValueError, match=match):
        parse_joy_selector(selector)


def test_decode_value_rejects_non_array_with_operators(tmp_path):
    # Spec resolution rejects string-DTYPE streams with apply, but a custom
    # decoder whose declared dtype is wrong slips past load-time validation;
    # the runtime backstop must raise instead of silently skipping the
    # declared pipeline.
    yaml = """
robot_type: test
robot_interface: ros2
fps: 30
observations:
  observation.state:
    channel: {topic: /joint_states, type: sensor_msgs/msg/JointState, decoder: 'test_decoders:_str_decoder'}
    align: {strategy: hold, timeline: receive}
    select: [position.j1]
    apply: [{clamp: {min: 0, max: 1}}]
actions:
  action:
    channel: {topic: /cmd, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.j1]
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml)
    contract = load_contract(p)
    spec = next(iter(iter_observation_specs(contract)))
    with pytest.raises(ValueError, match="numpy array"):
        decode_value(_joint_state(["j1"], [1.0]), spec)


# --- parse_field_selector (shared parallel-array selector grammar) ----------


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ("velocity.elbow", ("velocities", "elbow")),  # singular alias
        ("velocities.elbow", ("velocities", "elbow")),  # plural spelling
        ("elbow", ("positions", "elbow")),  # bare name -> default field
    ],
)
def test_field_selector_parses_trajectory_aliases(selector, expected):
    from rosetta.robots.ros2.field_access import JOINT_TRAJECTORY_FIELDS, parse_field_selector

    parsed = parse_field_selector(
        selector, JOINT_TRAJECTORY_FIELDS, default="position", msg_type="JointTrajectoryPoint"
    )
    assert parsed == expected


def test_field_selector_rejects_unknown_field():
    from rosetta.robots.ros2.field_access import JOINT_STATE_FIELDS, parse_field_selector

    with pytest.raises(ValueError, match="Valid fields: effort, position, velocity"):
        parse_field_selector("torque.elbow", JOINT_STATE_FIELDS, default="position", msg_type="JointState")


# --- per-type decode behavior ------------------------------------------------


def test_decode_imu_selects_paths(tmp_path):
    spec = _obs_spec(tmp_path, msg_type="sensor_msgs/msg/Imu", select="[orientation.x, angular_velocity.z]")
    msg = types.SimpleNamespace(
        orientation=types.SimpleNamespace(x=0.1, y=0.2, z=0.3, w=0.4),
        angular_velocity=types.SimpleNamespace(x=1.0, y=2.0, z=3.0),
    )
    assert np.allclose(decode_value(msg, spec), [0.1, 3.0])


def test_decode_odometry_selects_paths(tmp_path):
    spec = _obs_spec(
        tmp_path,
        msg_type="nav_msgs/msg/Odometry",
        select="[pose.pose.position.x, twist.twist.angular.z]",
    )
    msg = types.SimpleNamespace(
        pose=types.SimpleNamespace(pose=types.SimpleNamespace(position=types.SimpleNamespace(x=1.5))),
        twist=types.SimpleNamespace(twist=types.SimpleNamespace(angular=types.SimpleNamespace(z=-0.5))),
    )
    assert np.allclose(decode_value(msg, spec), [1.5, -0.5])


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
    _require_msg_pkg(msg_type)
    with pytest.raises(ContractValidationError, match="add 'select:'"):
        _obs_spec(tmp_path, msg_type=msg_type, select=None)


def test_decoder_requires_select_runtime_backstop():
    # The load-time check covers contracts; a hand-built spec still hits the
    # decoder's own raise.
    from rosetta.frames.codecs import DECODERS, discover_codecs

    discover_codecs()
    spec = types.SimpleNamespace(names=())
    with pytest.raises(ValueError, match="requires select"):
        DECODERS["sensor_msgs/msg/Imu"](types.SimpleNamespace(), spec)


# --- fail-fast on bad selectors and degenerate messages ----------------------


def test_joint_state_unknown_field_raises(tmp_path):
    spec = _obs_spec(tmp_path, select="[torque.j1]")
    with pytest.raises(ValueError, match="Unknown JointState field 'torque'"):
        decode_value(_joint_state(["j1"], [1.0]), spec)


def test_multidof_unknown_field_raises(tmp_path):
    # Regression: "velocity.j1" used to be treated as a DOF literally named
    # "velocity.j1" (and silently published as one on encode).
    _require_msg_pkg("control_msgs/msg/MultiDOFCommand")
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


def _two_wide_decoder(msg, spec):
    """Custom decoder that returns more values than the spec declares."""
    return np.array([1.0, 2.0])


def test_decode_value_rejects_width_mismatch(tmp_path):
    # select declares the width; a custom decoder that disagrees is refused at
    # the choke point (the only guard on the direct decode->encode HIL path).
    yaml = """
robot_type: test
robot_interface: ros2
fps: 30
observations:
  observation.state:
    channel: {topic: /joint_states, type: sensor_msgs/msg/JointState, decoder: 'test_decoders:_two_wide_decoder'}
    align: {strategy: hold, timeline: receive}
    select: [position.j1]
actions:
  action:
    channel: {topic: /cmd, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.j1]
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml)
    contract = load_contract(p)
    spec = next(iter(iter_observation_specs(contract)))
    with pytest.raises(ValueError, match="2 values for its 1-wide"):
        decode_value(_joint_state(["j1"], [1.0]), spec)


def test_decode_value_selectless_stream_must_be_scalar(tmp_path):
    # Built-in MultiArray decoders return the whole array; without select the
    # spec is 1-wide, so a multi-element message is refused at the choke point.
    spec = _obs_spec(tmp_path, msg_type="std_msgs/msg/Float64MultiArray", select=None)
    with pytest.raises(ValueError, match="select-less stream is a scalar"):
        decode_value(types.SimpleNamespace(data=[1.0, 2.0, 3.0]), spec)


def _none_decoder(msg, spec):
    """Custom decoder that violates the decoder contract by returning None."""


def test_decode_value_rejects_none_result(tmp_path):
    # A None decode must raise, not flow into the stream buffer: pushing None
    # marks the stream warmed-up while sampling still yields "missing".
    yaml = """
robot_type: test
robot_interface: ros2
fps: 30
observations:
  observation.state:
    channel: {topic: /joint_states, type: sensor_msgs/msg/JointState, decoder: 'test_decoders:_none_decoder'}
    align: {strategy: hold, timeline: receive}
    select: [position.j1]
actions:
  action:
    channel: {topic: /cmd, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.j1]
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml)
    contract = load_contract(p)
    spec = next(iter(iter_observation_specs(contract)))
    with pytest.raises(ValueError, match="returned None"):
        decode_value(_joint_state(["j1"], [1.0]), spec)
