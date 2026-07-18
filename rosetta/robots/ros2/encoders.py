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

"""
ROS message encoders for converting numpy arrays to ROS messages.

Each encoder is self-contained and registered with @register_encoder.
If you need to encode a message type that isn't here, add a new encoder.

Encoder signature: (action_vec, spec, stamp_ns=None) -> ROS message
- action_vec: numpy array of action values. The operator pipeline is already
  applied and encode_value has already checked the width against spec.dim.
- spec: ActionStreamSpec. Encoders read spec.names (selector paths) and
  spec.dim (vector width). The wire message type is hard-coded per encoder,
  not read from the spec.
- stamp_ns: optional timestamp in nanoseconds

Value transforms (deg2rad, ...) run in encode_value's operator pipeline before
the encoder is called. Encoders only scatter finished values into fields.

Selector-driven encoders (Twist, TwistStamped, JointState, JointTrajectory,
Joy, MultiDOFCommand) require ``select`` in the contract. They are registered
with ``requires_select=True``, so a select-less channel of these types fails at
contract load. Scalar std_msgs types need no select; a MultiArray still
needs one unless the stream is genuinely 1-wide, because select declares the
vector width (see encode_value's width gate).
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np

from rosetta.frames.codecs import register_encoder
from rosetta.robots.ros2.field_access import (
    JOINT_STATE_FIELDS,
    JOINT_TRAJECTORY_FIELDS,
    MULTIDOF_FIELDS,
    build_field_map,
    dot_set,
    parse_joy_selector,
)

if TYPE_CHECKING:
    from rosetta.contract.specs import ActionStreamSpec

# This module must stay importable without ROS: codec discovery
# (rosetta.frames.codecs.discover_codecs) imports it during ROS-less contract
# loading. rosidl is a call-time dependency of the encoder bodies only.

# =============================================================================
# Helper Functions
# =============================================================================


@lru_cache(maxsize=None)
def _get_message(type_str: str):
    """Resolve a ROS message class lazily; rosidl is not an import-time dependency."""
    from rosidl_runtime_py.utilities import get_message

    return get_message(type_str)


def _set_header_stamp(msg, stamp_ns: int | None) -> None:
    """Stamp the message header (the caller guarantees the type has one)."""
    if stamp_ns is None:
        return
    msg.header.stamp.sec = stamp_ns // 1_000_000_000
    msg.header.stamp.nanosec = stamp_ns % 1_000_000_000


def _require_scalar_spec(msg_type: str, spec: ActionStreamSpec) -> None:
    """Refuse a multi-wide spec on a single-value wire type.

    encode_value guarantees the vector width matches spec.dim, but only the
    encoder knows the wire type holds exactly one value. Without this check,
    ``select: [a, b, c]`` on a Float64 channel would silently publish only
    the first value.
    """
    if spec.dim != 1:
        raise ValueError(
            f"{msg_type} encodes a single value; got a {spec.dim}-wide spec — "
            f"use a MultiArray type or drop the extra select fields"
        )


# =============================================================================
# Dotted-Path Encoders (Twist / TwistStamped)
# =============================================================================


def _make_dotted_encoder(msg_type: str, example: str, *, stamped: bool):
    """Build an encoder whose selector names are root-relative dotted paths.

    Root-relative like every dotted-path codec. A stamped wrapper is NOT
    transparent, so contracts name the wrapped field explicitly
    (``twist.linear.x``).
    """
    short = msg_type.rsplit("/", 1)[-1]

    def encode(action_vec: np.ndarray, spec: ActionStreamSpec, stamp_ns: int | None = None) -> Any:
        if not spec.names:
            raise ValueError(f"{short} encoder requires select (e.g. [{example}])")
        msg = _get_message(msg_type)()
        if stamped:
            _set_header_stamp(msg, stamp_ns)
        arr = np.asarray(action_vec, dtype=np.float64).flatten()
        for i, path in enumerate(spec.names):
            dot_set(msg, path, arr[i])
        return msg

    encode.__doc__ = f"Encode to {msg_type}: selector names are root-relative dotted paths (e.g. [{example}])."
    return encode


register_encoder("geometry_msgs/msg/Twist", requires_select=True)(
    _make_dotted_encoder("geometry_msgs/msg/Twist", "linear.x, angular.z", stamped=False)
)
register_encoder("geometry_msgs/msg/TwistStamped", requires_select=True)(
    _make_dotted_encoder("geometry_msgs/msg/TwistStamped", "twist.linear.x, twist.angular.z", stamped=True)
)


# =============================================================================
# Scalar Encoders
# =============================================================================


@register_encoder("std_msgs/msg/Float32")
def _enc_float32(action_vec: np.ndarray, spec: ActionStreamSpec, stamp_ns: int | None = None) -> Any:
    """Encode to std_msgs/Float32 (single value; the spec must be 1-wide)."""
    _ = stamp_ns  # Unused - message type has no header
    _require_scalar_spec("std_msgs/msg/Float32", spec)
    msg = _get_message("std_msgs/msg/Float32")()
    msg.data = float(np.asarray(action_vec, dtype=np.float32).flatten()[0])
    return msg


@register_encoder("std_msgs/msg/Float64")
def _enc_float64(action_vec: np.ndarray, spec: ActionStreamSpec, stamp_ns: int | None = None) -> Any:
    """Encode to std_msgs/Float64 (single value; the spec must be 1-wide)."""
    _ = stamp_ns  # Unused - message type has no header
    _require_scalar_spec("std_msgs/msg/Float64", spec)
    msg = _get_message("std_msgs/msg/Float64")()
    msg.data = float(np.asarray(action_vec, dtype=np.float64).flatten()[0])
    return msg


# =============================================================================
# Array Encoders
# =============================================================================


@register_encoder("std_msgs/msg/Float32MultiArray")
def _enc_float32_array(action_vec: np.ndarray, spec: ActionStreamSpec, stamp_ns: int | None = None) -> Any:
    """Encode to std_msgs/Float32MultiArray."""
    _ = stamp_ns  # Unused - message type has no header
    msg = _get_message("std_msgs/msg/Float32MultiArray")()
    msg.data = np.asarray(action_vec, dtype=np.float32).flatten().tolist()
    return msg


@register_encoder("std_msgs/msg/Float64MultiArray")
def _enc_float64_array(action_vec: np.ndarray, spec: ActionStreamSpec, stamp_ns: int | None = None) -> Any:
    """Encode to std_msgs/Float64MultiArray."""
    _ = stamp_ns  # Unused - message type has no header
    msg = _get_message("std_msgs/msg/Float64MultiArray")()
    msg.data = np.asarray(action_vec, dtype=np.float64).flatten().tolist()
    return msg


@register_encoder("std_msgs/msg/Int32MultiArray")
def _enc_int32_array(action_vec: np.ndarray, spec: ActionStreamSpec, stamp_ns: int | None = None) -> Any:
    """Encode to std_msgs/Int32MultiArray (values rounded to nearest integer)."""
    _ = stamp_ns  # Unused - message type has no header
    msg = _get_message("std_msgs/msg/Int32MultiArray")()
    # rint, not a bare int cast: truncation toward zero would turn a policy's
    # 0.9 into 0 silently (and disagree with the Joy encoder's rounding).
    msg.data = np.rint(np.asarray(action_vec, dtype=np.float64)).astype(np.int32).flatten().tolist()
    return msg


def _require_full_field_coverage(
    msg_type: str, field_to_names: dict[str, dict[str, int]], name_order: list[str]
) -> None:
    """Validate per-field coverage for parallel-array messages.

    These messages pair one name list with per-field value arrays aligned
    1:1 to it. A field selected for only some names cannot be encoded
    faithfully: padding the gaps would fabricate zero commands, and ragged
    arrays silently misalign consumers that index by name position. Each
    selected field must therefore cover every name (unselected fields stay
    empty = unspecified).
    """
    for field, name_map in field_to_names.items():
        missing = [n for n in name_order if n not in name_map]
        if missing:
            raise ValueError(
                f"{msg_type} field '{field}' is selected for {sorted(name_map)} "
                f"but the message name set is {name_order} (missing {missing}). "
                f"Parallel arrays must align 1:1 with the name list; select "
                f"'{field}' for every name or for none."
            )


# =============================================================================
# JointState Encoder
# =============================================================================


@register_encoder("sensor_msgs/msg/JointState", requires_select=True)
def _enc_joint_state(action_vec: np.ndarray, spec: ActionStreamSpec, stamp_ns: int | None = None) -> Any:
    """
    Encode to sensor_msgs/JointState.

    Requires select paths like ['position.joint1', 'velocity.joint2']. Values
    map to the named field by joint name. A bare name means position.
    """
    if not spec.names:
        raise ValueError("JointState encoder requires select (e.g. [position.joint1, velocity.joint2])")

    msg = _get_message("sensor_msgs/msg/JointState")()
    _set_header_stamp(msg, stamp_ns)
    arr = np.asarray(action_vec, dtype=np.float64).flatten()

    field_to_joints, joint_order = build_field_map(
        spec.names, JOINT_STATE_FIELDS, default="position", msg_type="JointState"
    )
    _require_full_field_coverage("JointState", field_to_joints, joint_order)

    msg.name = joint_order

    # Coverage is validated above. Unselected fields stay empty, which
    # JointState reads as "unspecified". Zero-filling them would fabricate
    # zero velocity/effort commands.
    for field in ("position", "velocity", "effort"):
        joint_map = field_to_joints.get(field)
        values = [float(arr[joint_map[j]]) for j in joint_order] if joint_map else []
        setattr(msg, field, values)

    return msg


# =============================================================================
# JointTrajectory Encoder
# =============================================================================


@register_encoder("trajectory_msgs/msg/JointTrajectory", requires_select=True)
def _enc_joint_trajectory(action_vec: np.ndarray, spec: ActionStreamSpec, stamp_ns: int | None = None) -> Any:
    """
    Encode to trajectory_msgs/JointTrajectory (single-point trajectory).

    Requires select paths like ['position.joint1', 'velocity.joint2']. Values
    map to the named field by joint name. A bare name means position.

    time_from_start defaults to 0 (execute immediately).

    Field name aliases (both singular and plural are accepted):
      position / positions, velocity / velocities,
      acceleration / accelerations, effort
    """
    if not spec.names:
        raise ValueError("JointTrajectory encoder requires select (e.g. [position.joint1, velocity.joint2])")

    msg = _get_message("trajectory_msgs/msg/JointTrajectory")()
    _set_header_stamp(msg, stamp_ns)
    point = _get_message("trajectory_msgs/msg/JointTrajectoryPoint")()  # time_from_start is zero-initialized
    arr = np.asarray(action_vec, dtype=np.float64).flatten()

    field_to_joints, joint_order = build_field_map(
        spec.names, JOINT_TRAJECTORY_FIELDS, default="position", msg_type="JointTrajectoryPoint"
    )
    _require_full_field_coverage("JointTrajectory", field_to_joints, joint_order)

    msg.joint_names = joint_order

    # Coverage is validated above. Unselected point arrays stay empty
    # (unspecified) rather than fabricated zeros.
    for attr, joint_map in field_to_joints.items():
        setattr(point, attr, [float(arr[joint_map[j]]) for j in joint_order])

    msg.points = [point]
    return msg


# =============================================================================
# Joy Encoder
# =============================================================================


@register_encoder("sensor_msgs/msg/Joy", requires_select=True)
def _enc_joy(action_vec: np.ndarray, spec: ActionStreamSpec, stamp_ns: int | None = None) -> Any:
    """
    Encode to sensor_msgs/Joy.

    Requires select paths like ['axes.0', 'axes.1', 'buttons.0']: values map
    to axes/buttons by index. Joy arrays are dense by index, so each array is
    sized to the highest selected index and unselected slots below it are
    filled with 0.0 / 0 (the neutral Joy value). Select contiguously from
    index 0 if a consumer reads the gaps. Button values are rounded to the
    nearest integer.
    """
    if not spec.names:
        raise ValueError("Joy encoder requires select (e.g. [axes.0, buttons.0])")

    msg = _get_message("sensor_msgs/msg/Joy")()
    _set_header_stamp(msg, stamp_ns)
    arr = np.asarray(action_vec, dtype=np.float32).flatten()

    axes_map: dict[int, int] = {}  # axis_idx -> arr_idx
    buttons_map: dict[int, int] = {}  # button_idx -> arr_idx

    for i, path in enumerate(spec.names):
        field, idx = parse_joy_selector(path)
        if field == "axes":
            axes_map[idx] = i
        else:
            buttons_map[idx] = i

    if axes_map:
        axes = [0.0] * (max(axes_map) + 1)
        for axis_idx, arr_idx in axes_map.items():
            axes[axis_idx] = float(arr[arr_idx])
        msg.axes = axes

    if buttons_map:
        buttons = [0] * (max(buttons_map) + 1)
        for btn_idx, arr_idx in buttons_map.items():
            buttons[btn_idx] = round(arr[arr_idx])
        msg.buttons = buttons

    return msg


# =============================================================================
# MultiDOFCommand Encoder
# =============================================================================


@register_encoder("control_msgs/msg/MultiDOFCommand", requires_select=True)
def _enc_multidof_command(action_vec: np.ndarray, spec: ActionStreamSpec, stamp_ns: int | None = None) -> Any:
    """
    Encode to control_msgs/MultiDOFCommand.

    Requires select paths like ['values.joint1', 'values_dot.joint1']. Values
    map to the named field by DOF name. A bare name means values.
    """
    _ = stamp_ns  # Unused - message type has no header
    if not spec.names:
        raise ValueError("MultiDOFCommand encoder requires select (e.g. [values.joint1, values_dot.joint1])")

    msg = _get_message("control_msgs/msg/MultiDOFCommand")()
    arr = np.asarray(action_vec, dtype=np.float64).flatten()

    field_to_dofs, dof_order = build_field_map(
        spec.names, MULTIDOF_FIELDS, default="values", msg_type="MultiDOFCommand"
    )
    _require_full_field_coverage("MultiDOFCommand", field_to_dofs, dof_order)

    msg.dof_names = dof_order

    # Coverage is validated above. An unselected field stays empty
    # (unspecified). control_msgs requires values/values_dot to align 1:1
    # with dof_names, so a partial array would silently cross-wire consumers
    # (including our own decoder) that index by DOF position.
    values_map = field_to_dofs.get("values")
    values_dot_map = field_to_dofs.get("values_dot")
    msg.values = [float(arr[values_map[d]]) for d in dof_order] if values_map else []
    msg.values_dot = [float(arr[values_dot_map[d]]) for d in dof_order] if values_dot_map else []

    return msg
