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
- action_vec: numpy array of action values
- spec: ActionStreamSpec with names, clamp, msg_type
- stamp_ns: optional timestamp in nanoseconds
"""

from __future__ import annotations

from typing import Any

import numpy as np
from rosidl_runtime_py.utilities import get_message

from .converters import register_encoder
from .contract import ActionStreamSpec
from .ros2_utils import dot_set


# =============================================================================
# Helper Functions
# =============================================================================


def _apply_clamp(arr: np.ndarray, clamp: tuple[float, float] | None) -> np.ndarray:
    """Apply optional clamping to array."""
    if clamp:
        return np.clip(arr, clamp[0], clamp[1])
    return arr


def _set_header_stamp(msg, stamp_ns: int | None) -> None:
    """Set header timestamp if the message has a header."""
    if stamp_ns is None:
        return
    try:
        msg.header.stamp.sec = stamp_ns // 1_000_000_000
        msg.header.stamp.nanosec = stamp_ns % 1_000_000_000
    except AttributeError:
        pass  # Message doesn't have a header


# =============================================================================
# Twist Encoder
# =============================================================================


@register_encoder("geometry_msgs/msg/Twist")
def _enc_twist(
    action_vec: np.ndarray, spec: ActionStreamSpec, stamp_ns: int | None = None
) -> Any:
    """Encode to geometry_msgs/Twist.

    Requires selector.names like ['linear.x', 'angular.z'].
    """
    if not spec.names:
        raise ValueError(
            "Twist encoder requires selector.names "
            "(e.g., ['linear.x', 'angular.z'])"
        )

    msg_cls = get_message("geometry_msgs/msg/Twist")
    msg = msg_cls()

    arr = _apply_clamp(np.asarray(action_vec, dtype=np.float64).flatten(), spec.clamp)

    if len(spec.names) != len(arr):
        raise ValueError(f"names length ({len(spec.names)}) != action length ({len(arr)})")

    for i, path in enumerate(spec.names):
        dot_set(msg, path, arr[i])

    return msg


# =============================================================================
# TwistStamped Encoder
# =============================================================================


@register_encoder("geometry_msgs/msg/TwistStamped")
def _enc_twist_stamped(
    action_vec: np.ndarray, spec: ActionStreamSpec, stamp_ns: int | None = None
) -> Any:
    """Encode to geometry_msgs/TwistStamped.

    Requires selector.names like ['linear.x', 'angular.z'].
    Same selector syntax as Twist - the stamped wrapper is transparent.
    """
    if not spec.names:
        raise ValueError(
            "TwistStamped encoder requires selector.names "
            "(e.g., ['linear.x', 'angular.z'])"
        )

    msg_cls = get_message("geometry_msgs/msg/TwistStamped")
    msg = msg_cls()
    _set_header_stamp(msg, stamp_ns)

    arr = _apply_clamp(np.asarray(action_vec, dtype=np.float64).flatten(), spec.clamp)

    if len(spec.names) != len(arr):
        raise ValueError(f"names length ({len(spec.names)}) != action length ({len(arr)})")

    for i, path in enumerate(spec.names):
        dot_set(msg.twist, path, arr[i])

    return msg


# =============================================================================
# Scalar Encoders
# =============================================================================

# Be carefull. Float 32 and Float64 have no header
@register_encoder("std_msgs/msg/Float32")
def _enc_float32(
    action_vec: np.ndarray, spec: ActionStreamSpec, stamp_ns: int | None = None
) -> Any:
    """Encode to std_msgs/Float32 (scalar)."""
    _ = stamp_ns  # Unused - message type has no header
    msg_cls = get_message("std_msgs/msg/Float32")
    msg = msg_cls()
    arr = _apply_clamp(np.asarray(action_vec, dtype=np.float32).flatten(), spec.clamp)
    msg.data = float(arr[0])
    return msg


@register_encoder("std_msgs/msg/Float64")
def _enc_float64(
    action_vec: np.ndarray, spec: ActionStreamSpec, stamp_ns: int | None = None
) -> Any:
    """Encode to std_msgs/Float64 (scalar)."""
    _ = stamp_ns  # Unused - message type has no header
    msg_cls = get_message("std_msgs/msg/Float64")
    msg = msg_cls()
    arr = _apply_clamp(np.asarray(action_vec, dtype=np.float64).flatten(), spec.clamp)
    msg.data = float(arr[0])
    return msg
    
    
# =============================================================================
# Array Encoders
# =============================================================================


@register_encoder("std_msgs/msg/Float32MultiArray")
def _enc_float32_array(
    action_vec: np.ndarray, spec: ActionStreamSpec, stamp_ns: int | None = None
) -> Any:
    """Encode to std_msgs/Float32MultiArray."""
    _ = stamp_ns  # Unused - message type has no header
    msg_cls = get_message("std_msgs/msg/Float32MultiArray")
    msg = msg_cls()

    arr = _apply_clamp(np.asarray(action_vec, dtype=np.float32).flatten(), spec.clamp)
    msg.data = arr.tolist()

    return msg


@register_encoder("std_msgs/msg/Float64MultiArray")
def _enc_float64_array(
    action_vec: np.ndarray, spec: ActionStreamSpec, stamp_ns: int | None = None
) -> Any:
    """Encode to std_msgs/Float64MultiArray."""
    _ = stamp_ns  # Unused - message type has no header
    msg_cls = get_message("std_msgs/msg/Float64MultiArray")
    msg = msg_cls()

    arr = _apply_clamp(np.asarray(action_vec, dtype=np.float64).flatten(), spec.clamp)
    msg.data = arr.tolist()

    return msg


@register_encoder("std_msgs/msg/Int32MultiArray")
def _enc_int32_array(
    action_vec: np.ndarray, spec: ActionStreamSpec, stamp_ns: int | None = None
) -> Any:
    """Encode to std_msgs/Int32MultiArray."""
    _ = stamp_ns  # Unused - message type has no header
    msg_cls = get_message("std_msgs/msg/Int32MultiArray")
    msg = msg_cls()

    arr = _apply_clamp(np.asarray(action_vec, dtype=np.int32).flatten(), spec.clamp)
    msg.data = arr.tolist()

    return msg


# =============================================================================
# JointState Encoder
# =============================================================================


@register_encoder("sensor_msgs/msg/JointState")
def _enc_joint_state(
    action_vec: np.ndarray, spec: ActionStreamSpec, stamp_ns: int | None = None
) -> Any:
    """Encode to sensor_msgs/JointState.

    With selector.names like ['position.joint1', 'velocity.joint2']:
      - Maps values to specified fields by joint name
    Without names:
      - Maps action vector to positions with auto-generated names
    """
    msg_cls = get_message("sensor_msgs/msg/JointState")
    msg = msg_cls()
    _set_header_stamp(msg, stamp_ns)

    arr = _apply_clamp(np.asarray(action_vec, dtype=np.float64).flatten(), spec.clamp)

    if not spec.names:
        # Default: all values go to position
        msg.name = [f"joint_{i}" for i in range(len(arr))]
        msg.position = arr.tolist()
        msg.velocity = []
        msg.effort = []
        return msg

    if len(spec.names) != len(arr):
        raise ValueError(f"names length ({len(spec.names)}) != action length ({len(arr)})")

    # Parse names like "position.shoulder_pan", "velocity.elbow"
    field_to_joints: dict[str, dict[str, int]] = {}  # field -> {joint_name -> arr_index}
    joint_order: list[str] = []
    seen_joints: set[str] = set()

    for i, path in enumerate(spec.names):
        if "." in path:
            field, joint_name = path.split(".", 1)
        else:
            field, joint_name = "position", path

        field_to_joints.setdefault(field, {})[joint_name] = i

        if joint_name not in seen_joints:
            joint_order.append(joint_name)
            seen_joints.add(joint_name)

    msg.name = joint_order
    n_joints = len(joint_order)
    joint_to_idx = {name: i for i, name in enumerate(joint_order)}

    # Initialize arrays
    msg.position = [0.0] * n_joints
    msg.velocity = [0.0] * n_joints
    msg.effort = [0.0] * n_joints

    # Fill arrays
    for field, joint_map in field_to_joints.items():
        if field == "position":
            target = msg.position
        elif field == "velocity":
            target = msg.velocity
        elif field == "effort":
            target = msg.effort
        else:
            raise ValueError(f"Unknown JointState field '{field}'")

        for joint_name, arr_idx in joint_map.items():
            target[joint_to_idx[joint_name]] = float(arr[arr_idx])

    return msg


# =============================================================================
# JointTrajectory Encoder
# =============================================================================


@register_encoder("trajectory_msgs/msg/JointTrajectory")
def _enc_joint_trajectory(
    action_vec: np.ndarray, spec: ActionStreamSpec, stamp_ns: int | None = None
) -> Any:
    """Encode to trajectory_msgs/JointTrajectory (single-point trajectory).

    With selector.names like ['position.joint1', 'velocity.joint2']:
      - Maps values to specified fields by joint name
    Without names:
      - Maps action vector to positions with auto-generated names

    time_from_start defaults to 0 (execute immediately).

    Field name aliases (both singular and plural are accepted):
      position / positions, velocity / velocities,
      acceleration / accelerations, effort
    """
    traj_cls = get_message("trajectory_msgs/msg/JointTrajectory")
    point_cls = get_message("trajectory_msgs/msg/JointTrajectoryPoint")
    msg = traj_cls()
    _set_header_stamp(msg, stamp_ns)

    arr = _apply_clamp(np.asarray(action_vec, dtype=np.float64).flatten(), spec.clamp)
    point = point_cls()  # time_from_start is zero-initialized

    if not spec.names:
        msg.joint_names = [f"joint_{i}" for i in range(len(arr))]
        point.positions = arr.tolist()
        msg.points = [point]
        return msg

    if len(spec.names) != len(arr):
        raise ValueError(f"names length ({len(spec.names)}) != action length ({len(arr)})")

    _FIELD_MAP = {
        "position": "positions",
        "positions": "positions",
        "velocity": "velocities",
        "velocities": "velocities",
        "acceleration": "accelerations",
        "accelerations": "accelerations",
        "effort": "effort",
    }

    # Collect joint order and per-field assignments
    joint_order: list[str] = []
    seen_joints: set[str] = set()
    field_to_joints: dict[str, dict[str, int]] = {}  # attr -> {joint_name -> arr_idx}

    for i, path in enumerate(spec.names):
        if "." in path:
            field, joint_name = path.split(".", 1)
        else:
            field, joint_name = "position", path

        attr = _FIELD_MAP.get(field)
        if attr is None:
            raise ValueError(
                f"Unknown JointTrajectoryPoint field '{field}'. "
                f"Valid fields: position, velocity, acceleration, effort"
            )

        field_to_joints.setdefault(attr, {})[joint_name] = i
        if joint_name not in seen_joints:
            joint_order.append(joint_name)
            seen_joints.add(joint_name)

    msg.joint_names = joint_order
    n_joints = len(joint_order)
    joint_to_idx = {name: i for i, name in enumerate(joint_order)}

    for attr, joint_map in field_to_joints.items():
        values = [0.0] * n_joints
        for joint_name, arr_idx in joint_map.items():
            values[joint_to_idx[joint_name]] = float(arr[arr_idx])
        setattr(point, attr, values)

    msg.points = [point]
    return msg


# =============================================================================
# Joy Encoder
# =============================================================================


@register_encoder("sensor_msgs/msg/Joy")
def _enc_joy(
    action_vec: np.ndarray, spec: ActionStreamSpec, stamp_ns: int | None = None
) -> Any:
    """Encode to sensor_msgs/Joy.

    With selector.names like ['axes.0', 'axes.1', 'buttons.0']:
      - Maps values to axes/buttons by index
    Without names:
      - Maps action vector to axes

    Button values are rounded to the nearest integer.
    Selector syntax: "<field>.<index>" where field is "axes" or "buttons".
    """
    msg_cls = get_message("sensor_msgs/msg/Joy")
    msg = msg_cls()
    _set_header_stamp(msg, stamp_ns)

    arr = _apply_clamp(np.asarray(action_vec, dtype=np.float32).flatten(), spec.clamp)

    if not spec.names:
        msg.axes = arr.tolist()
        msg.buttons = []
        return msg

    if len(spec.names) != len(arr):
        raise ValueError(f"names length ({len(spec.names)}) != action length ({len(arr)})")

    axes_map: dict[int, int] = {}    # axis_idx -> arr_idx
    buttons_map: dict[int, int] = {}  # button_idx -> arr_idx

    for i, path in enumerate(spec.names):
        if "." in path:
            field, idx_str = path.split(".", 1)
        else:
            field, idx_str = "axes", path

        try:
            idx = int(idx_str)
        except ValueError:
            raise ValueError(
                f"Joy selector index must be an integer, got '{idx_str}' "
                f"in selector '{path}'"
            )

        if field == "axes":
            axes_map[idx] = i
        elif field == "buttons":
            buttons_map[idx] = i
        else:
            raise ValueError(
                f"Unknown Joy field '{field}'. Valid fields: axes, buttons"
            )

    if axes_map:
        axes = [0.0] * (max(axes_map) + 1)
        for axis_idx, arr_idx in axes_map.items():
            axes[axis_idx] = float(arr[arr_idx])
        msg.axes = axes

    if buttons_map:
        buttons = [0] * (max(buttons_map) + 1)
        for btn_idx, arr_idx in buttons_map.items():
            buttons[btn_idx] = int(round(arr[arr_idx]))
        msg.buttons = buttons

    return msg


# =============================================================================
# MultiDOFCommand Encoder
# =============================================================================


@register_encoder("control_msgs/msg/MultiDOFCommand")
def _enc_multidof_command(
    action_vec: np.ndarray, spec: ActionStreamSpec, stamp_ns: int | None = None
) -> Any:
    """Encode to control_msgs/MultiDOFCommand.

    With selector.names like ['values.joint1', 'values_dot.joint1']:
      - Maps values to specified DOF fields by name
    Without names:
      - Maps action vector to values with auto-generated names
    """
    _ = stamp_ns  # Unused - message type has no header
    msg_cls = get_message("control_msgs/msg/MultiDOFCommand")
    msg = msg_cls()

    arr = _apply_clamp(np.asarray(action_vec, dtype=np.float64).flatten(), spec.clamp)

    if not spec.names:
        msg.dof_names = [f"dof_{i}" for i in range(len(arr))]
        msg.values = arr.tolist()
        msg.values_dot = []
        return msg

    if len(spec.names) != len(arr):
        raise ValueError(f"names length ({len(spec.names)}) != action length ({len(arr)})")

    # Parse names: "values.foo" -> values[foo], "values_dot.bar" -> values_dot[bar]
    values_map: dict[str, int] = {}  # dof_name -> arr_index
    values_dot_map: dict[str, int] = {}
    dof_order: list[str] = []
    seen_dofs: set[str] = set()

    for i, name in enumerate(spec.names):
        if name.startswith("values_dot."):
            dof_name = name[11:]
            values_dot_map[dof_name] = i
        elif name.startswith("values."):
            dof_name = name[7:]
            values_map[dof_name] = i
        else:
            dof_name = name
            values_map[dof_name] = i

        if dof_name not in seen_dofs:
            dof_order.append(dof_name)
            seen_dofs.add(dof_name)

    msg.dof_names = dof_order

    # Build arrays - require explicit specification for each DOF
    msg.values = []
    msg.values_dot = []
    for d in dof_order:
        if d in values_map:
            msg.values.append(float(arr[values_map[d]]))
        if d in values_dot_map:
            msg.values_dot.append(float(arr[values_dot_map[d]]))

    return msg


# =============================================================================
# J1939 Encoders
# =============================================================================


@register_encoder("j1939_msgs/msg/EngineRPM")
def _enc_engine_rpm(
    action_vec: np.ndarray, spec: ActionStreamSpec, stamp_ns: int | None = None
) -> Any:
    """Encode to j1939_msgs/EngineRPM.

    With selector names like ["engine_rpm", "engine_on"]:
      - Maps values to specified fields
    Without names:
      - First value -> engine_rpm, second value (if present) -> engine_on

    engine_on is treated as bool (nonzero = True).
    Valid field names: engine_rpm, engine_on
    """
    msg_cls = get_message("j1939_msgs/msg/EngineRPM")
    msg = msg_cls()
    _set_header_stamp(msg, stamp_ns)

    arr = _apply_clamp(np.asarray(action_vec, dtype=np.float64).flatten(), spec.clamp)

    if not spec.names:
        if len(arr) >= 1:
            msg.engine_rpm = float(arr[0])
        if len(arr) >= 2:
            msg.engine_on = bool(arr[1])
        return msg

    if len(spec.names) != len(arr):
        raise ValueError(f"names length ({len(spec.names)}) != action length ({len(arr)})")

    for i, name in enumerate(spec.names):
        if name == "engine_rpm":
            msg.engine_rpm = float(arr[i])
        elif name == "engine_on":
            msg.engine_on = bool(arr[i])
        else:
            raise ValueError(
                f"Unknown EngineRPM field '{name}'. "
                f"Valid fields: engine_rpm, engine_on"
            )

    return msg
