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
