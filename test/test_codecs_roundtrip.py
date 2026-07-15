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

"""Full encode->decode round-trip tests for every codec pair.

encode_value builds real ROS message instances via
rosidl_runtime_py.get_message. Individual pairs whose message package is
missing (e.g. control_msgs) skip per-parameter.

Coverage is enforced: ``test_every_pair_has_a_sample`` fails if a registered
encoder/decoder pair has no sample here, so a new pair cannot be added without
a round-trip case -- closing the historical 1-of-N gap for good.
"""

import numpy as np
import pytest
from rosetta.contract.schema import load_contract
from rosetta.contract.specs import (
    iter_action_specs,
    iter_observation_specs,
)
from rosetta.frames.codecs import (
    DECODERS,
    ENCODERS,
    NonFiniteActionError,
    decode_value,
    discover_codecs,
    encode_value,
)
from rosidl_runtime_py.utilities import get_message

# Every type that has BOTH an encoder and a decoder is round-trippable and must
# have a sample below. type -> (select, sample_vector, atol).
# atol > 0 for float32-backed codecs (storage precision); 0 means bit-exact.
SAMPLES: dict[str, tuple[list[str], list[float], float]] = {
    "sensor_msgs/msg/JointState": (["position.j1", "position.j2"], [0.5, -0.3], 0.0),
    "geometry_msgs/msg/Twist": (
        ["linear.x", "linear.y", "linear.z", "angular.x", "angular.y", "angular.z"],
        [0.1, 0.2, 0.3, -0.1, -0.2, -0.3],
        0.0,
    ),
    "geometry_msgs/msg/TwistStamped": (["twist.linear.x", "twist.angular.z"], [0.7, -0.4], 0.0),
    "std_msgs/msg/Float64": (["v"], [1.5], 0.0),
    "std_msgs/msg/Float32": (["v"], [1.25], 1e-6),
    "std_msgs/msg/Float64MultiArray": (["a", "b", "c"], [0.1, 0.2, 0.3], 0.0),
    "std_msgs/msg/Float32MultiArray": (["a", "b", "c"], [0.25, 0.5, -0.75], 1e-6),
    "std_msgs/msg/Int32MultiArray": (["a", "b", "c"], [1, 2, 3], 0.0),
    "sensor_msgs/msg/Joy": (["axes.0", "axes.1", "buttons.0"], [0.5, -0.25, 1.0], 1e-6),
    "trajectory_msgs/msg/JointTrajectory": (
        ["position.j1", "position.j2"],
        [0.5, -0.3],
        0.0,
    ),
    "control_msgs/msg/MultiDOFCommand": (["values.d1", "values.d2"], [0.5, -0.3], 0.0),
}


def _pairs() -> list[str]:
    """Message types that have both an encoder and a decoder."""
    discover_codecs()  # deterministic registration; no side-effect import needed
    return sorted(set(DECODERS) & set(ENCODERS))


def _specs(tmp_path, msg_type: str, select: list[str], apply_block: str = "[]"):
    """Build (action_spec, observation_spec) for a type from a tiny contract."""
    sel = "[" + ", ".join(select) + "]"
    yaml = f"""
robot_type: test
robot_interface: ros2
fps: 30
observations:
  observation.x:
    channel: {{topic: /t, type: {msg_type}}}
    align: {{strategy: hold, timeline: receive}}
    select: {sel}
    apply: {apply_block}
actions:
  action:
    channel: {{topic: /t, type: {msg_type}}}
    align: {{strategy: hold, timeline: receive}}
    select: {sel}
    apply: {apply_block}
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml)
    contract = load_contract(p)
    return (
        next(iter(iter_action_specs(contract))),
        next(iter(iter_observation_specs(contract))),
    )


def test_every_pair_has_a_sample():
    """Coverage guard: no registered codec pair may lack a round-trip sample."""
    missing = sorted(set(_pairs()) - set(SAMPLES))
    assert not missing, (
        f"Codec pairs without a round-trip sample in SAMPLES: {missing}. Add a sample so the pair is round-trip tested."
    )


@pytest.mark.parametrize("msg_type", _pairs())
def test_codec_pair_round_trips(msg_type, tmp_path):
    """decode(encode(v)) == v for every encoder/decoder pair."""
    try:
        get_message(msg_type)
    except Exception:
        pytest.skip(f"{msg_type} message package unavailable")

    assert msg_type in SAMPLES, f"No sample for {msg_type}"
    select, vec, atol = SAMPLES[msg_type]
    action_spec, obs_spec = _specs(tmp_path, msg_type, select)
    v = np.asarray(vec, dtype=np.float64)

    out1 = np.asarray(decode_value(encode_value(vec, action_spec), obs_spec), dtype=np.float64)
    assert np.allclose(out1, v, rtol=0, atol=atol), f"{msg_type}: {v} -> {out1}"


def test_roundtrip_rad2deg(tmp_path):
    """Round trip through the operator pipeline (deg<->rad), not just the raw codec."""
    action_spec, obs_spec = _specs(
        tmp_path,
        "sensor_msgs/msg/JointState",
        ["position.j1", "position.j2"],
        apply_block="[rad2deg]",
    )
    values = [90.0, 45.0]
    out = decode_value(encode_value(values, action_spec), obs_spec)
    assert np.allclose(out, values)


# ---------------------------------------------------------------------------
# Parallel-array field coverage (JointState / JointTrajectory / MultiDOFCommand)
#
# Regression guards: unselected fields used to be fabricated as full zero
# arrays (JointState/JointTrajectory: a controller reads commanded zero
# velocity/effort), and MultiDOFCommand emitted RAGGED values/values_dot
# misaligned to dof_names whenever the two DOF selections differed —
# silent cross-joint command corruption, even against rosetta's own decoder.
# ---------------------------------------------------------------------------


def _action_spec(tmp_path, msg_type, select):
    return _specs(tmp_path, msg_type, select)[0]


def test_joint_state_unselected_fields_stay_empty(tmp_path):
    spec = _action_spec(tmp_path, "sensor_msgs/msg/JointState", ["position.j1", "position.j2"])
    msg = encode_value([0.5, -0.3], spec)
    assert list(msg.name) == ["j1", "j2"]
    assert list(msg.position) == [0.5, -0.3]
    assert list(msg.velocity) == []  # unspecified, not fabricated zeros
    assert list(msg.effort) == []


def test_joint_state_full_mixed_fields_align(tmp_path):
    spec = _action_spec(
        tmp_path,
        "sensor_msgs/msg/JointState",
        ["position.j1", "position.j2", "velocity.j1", "velocity.j2"],
    )
    msg = encode_value([0.1, 0.2, 1.1, 1.2], spec)
    assert list(msg.name) == ["j1", "j2"]
    assert list(msg.position) == [0.1, 0.2]
    assert list(msg.velocity) == [1.1, 1.2]
    assert list(msg.effort) == []


def test_joint_state_partial_field_coverage_raises(tmp_path):
    spec = _action_spec(
        tmp_path,
        "sensor_msgs/msg/JointState",
        ["position.j1", "position.j2", "velocity.j2"],
    )
    with pytest.raises(ValueError, match="field 'velocity'.*missing.*j1"):
        encode_value([0.1, 0.2, 1.2], spec)


def test_joint_trajectory_partial_field_coverage_raises(tmp_path):
    try:
        get_message("trajectory_msgs/msg/JointTrajectory")
    except Exception:
        pytest.skip("trajectory_msgs unavailable")
    spec = _action_spec(
        tmp_path,
        "trajectory_msgs/msg/JointTrajectory",
        ["position.j1", "position.j2", "velocity.j1"],
    )
    with pytest.raises(ValueError, match="field 'velocities'.*missing.*j2"):
        encode_value([0.1, 0.2, 1.1], spec)


def test_multidof_identical_sets_align(tmp_path):
    try:
        get_message("control_msgs/msg/MultiDOFCommand")
    except Exception:
        pytest.skip("control_msgs unavailable")
    spec = _action_spec(
        tmp_path,
        "control_msgs/msg/MultiDOFCommand",
        ["values.d1", "values.d2", "values_dot.d1", "values_dot.d2"],
    )
    msg = encode_value([0.1, 0.2, 1.1, 1.2], spec)
    assert list(msg.dof_names) == ["d1", "d2"]
    assert list(msg.values) == [0.1, 0.2]
    assert list(msg.values_dot) == [1.1, 1.2]


def test_multidof_divergent_sets_raise(tmp_path):
    try:
        get_message("control_msgs/msg/MultiDOFCommand")
    except Exception:
        pytest.skip("control_msgs unavailable")
    spec = _action_spec(
        tmp_path,
        "control_msgs/msg/MultiDOFCommand",
        ["values.d1", "values_dot.d2"],
    )
    with pytest.raises(ValueError, match="MultiDOFCommand.*missing"):
        encode_value([0.1, 1.2], spec)


# ---------------------------------------------------------------------------
# Requires-select and per-type encoder rules
# ---------------------------------------------------------------------------


def _selectless_action_spec(tmp_path, msg_type):
    yaml = f"""
robot_type: test
robot_interface: ros2
fps: 30
observations:
  observation.x:
    channel: {{topic: /obs, type: sensor_msgs/msg/JointState}}
    align: {{strategy: hold, timeline: receive}}
    select: [position.j1]
actions:
  action:
    channel: {{topic: /t, type: {msg_type}}}
    align: {{strategy: hold, timeline: receive}}
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml)
    return next(iter(iter_action_specs(load_contract(p))))


@pytest.mark.parametrize(
    "msg_type",
    [
        "geometry_msgs/msg/Twist",
        "geometry_msgs/msg/TwistStamped",
        "sensor_msgs/msg/JointState",
        "trajectory_msgs/msg/JointTrajectory",
        "sensor_msgs/msg/Joy",
        "control_msgs/msg/MultiDOFCommand",
    ],
)
def test_selector_encoders_require_select(tmp_path, msg_type):
    # select declares which field each value lands in; a select-less
    # multi-field action has no defensible default. (The encoder raises before
    # message construction, but contract load introspects the type.)
    try:
        get_message(msg_type)
    except Exception:
        pytest.skip(f"{msg_type} message package unavailable")
    spec = _selectless_action_spec(tmp_path, msg_type)
    with pytest.raises(ValueError, match="requires select"):
        encode_value([0.5], spec)


def test_scalar_encoder_rejects_multiwide_spec(tmp_path):
    # The vector width matches the spec (so the choke point passes); only the
    # encoder knows the wire type carries a single value. This used to
    # silently publish arr[0] and drop the rest.
    spec = _action_spec(tmp_path, "std_msgs/msg/Float64", ["a", "b", "c"])
    with pytest.raises(ValueError, match="encodes a single value"):
        encode_value([1.0, 2.0, 3.0], spec)


def test_int32_multiarray_encoder_rounds(tmp_path):
    # rint, not truncation toward zero: 0.9 must publish as 1 (and match the
    # Joy encoder's button rounding).
    spec = _action_spec(tmp_path, "std_msgs/msg/Int32MultiArray", ["a", "b"])
    msg = encode_value([0.9, -0.6], spec)
    assert list(msg.data) == [1, -1]


def test_multidof_unknown_field_selector_raises(tmp_path):
    try:
        get_message("control_msgs/msg/MultiDOFCommand")
    except Exception:
        pytest.skip("control_msgs unavailable")
    # Regression: "velocity.j1" used to be silently published as a DOF
    # literally named "velocity.j1".
    spec = _action_spec(tmp_path, "control_msgs/msg/MultiDOFCommand", ["velocity.j1"])
    with pytest.raises(ValueError, match="Unknown MultiDOFCommand field 'velocity'"):
        encode_value([0.5], spec)


# ---------------------------------------------------------------------------
# Finiteness gate: encode_value is the single choke point before the wire, so
# it refuses NaN/Inf commands (e.g. from a diverged policy) instead of
# scattering them into message fields. clamp does NOT scrub NaN (np.clip
# propagates it) -- the gate is the guarantee, on every action channel.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_encode_refuses_non_finite_command(tmp_path, bad):
    spec = _action_spec(tmp_path, "sensor_msgs/msg/JointState", ["position.j1", "position.j2"])
    with pytest.raises(NonFiniteActionError, match="position.j2"):
        encode_value([0.5, bad], spec)


def test_encode_refuses_nan_even_through_clamp(tmp_path):
    # A clamp on the action does not launder NaN into a bounded value; the
    # gate still fires after the inverse pipeline.
    spec, _ = _specs(
        tmp_path,
        "sensor_msgs/msg/JointState",
        ["position.j1"],
        apply_block="[{clamp: {min: -1.0, max: 1.0}}]",
    )
    with pytest.raises(NonFiniteActionError):
        encode_value([float("nan")], spec)


def test_encode_finite_command_passes_gate(tmp_path):
    spec = _action_spec(tmp_path, "sensor_msgs/msg/JointState", ["position.j1"])
    msg = encode_value([0.5], spec)
    assert list(msg.position) == [0.5]


def test_encode_rejects_width_mismatch(tmp_path):
    # select declares the vector width; a mismatched vector is refused at the
    # choke point (as a structural ValueError, not a finiteness error) before
    # any encoder scatters it — this also covers direct decode->encode
    # callers like the HIL teleop passthrough, which bypass FrameLayout.
    spec = _action_spec(tmp_path, "sensor_msgs/msg/JointState", ["position.j1"])
    with pytest.raises(ValueError, match="2 values for its 1-wide"):
        encode_value([0.5, float("nan")], spec)


def test_split_output_reaches_gate_with_nan_intact(tmp_path):
    # Regression (gate-before-cast ordering): FrameLayout.split used to cast
    # the action frame to the key's dtype BEFORE this gate — on an int32
    # action key that turned NaN into INT_MIN, finite garbage the gate waved
    # through onto the wire. split now emits float64, so the policy's NaN
    # arrives here intact and the frame is refused.
    from rosetta.frames.layout import FrameLayout

    spec = _action_spec(tmp_path, "std_msgs/msg/Int32MultiArray", ["a", "b"])
    assert spec.dtype == "int32"  # native codec dtype; the dangerous case
    (part,) = FrameLayout([spec]).split({"action": np.array([float("nan"), 1.0])})
    with pytest.raises(NonFiniteActionError):
        encode_value(part, spec)
