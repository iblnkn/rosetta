# rosetta/common/encoders.py
from __future__ import annotations

"""
ROS message encoders for converting numpy arrays to ROS messages.

This module contains all registered encoders for converting policy outputs
(numpy arrays) into ROS messages for publishing. Encoders are registered using
the @register_encoder decorator and called via encode_value() in processing_utils.py.

Supported message types:
- geometry_msgs/msg/Twist: Convert to Twist messages for robot control
- std_msgs/msg/Float32MultiArray: Convert to Float32MultiArray messages
- std_msgs/msg/Int32MultiArray: Convert to Int32MultiArray messages
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from rosidl_runtime_py.utilities import get_message

from rosetta.common.contract_utils import register_encoder


# ---------- Helper functions ----------


def dot_set(obj, path: str, value: float):
    """Set a dotted attribute on a pre-shaped ROS message (no new objects)."""
    cur = obj
    parts = path.split(".")
    for p in parts[:-1]:
        cur = getattr(cur, p)
    setattr(cur, parts[-1], float(value))


def _encode_via_dotted_paths(
    ros_type: str,
    names: List[str],
    action_vec: Sequence[float],
    clamp: Optional[Tuple[float, float]] = None,
) -> Any:
    """Fallback: explicit dotted-path assignment into a freshly-constructed ROS message."""
    if not names:
        raise ValueError(
            f"encode_value: no encoder registered for '{ros_type}' and no dotted-path names provided"
        )
    msg_cls = get_message(ros_type)
    msg = msg_cls()
    arr = np.asarray(action_vec, dtype=np.float32).reshape(-1)
    if clamp:
        arr = np.clip(arr, clamp[0], clamp[1])
    if len(names) > arr.size:
        raise ValueError(
            f"encode_value: names length ({len(names)}) exceeds action vector length ({arr.size})"
        )
    for i, path in enumerate(names):
        v = float(arr[i]) if i < arr.size else 0.0
        dot_set(msg, path, v)
    return msg


# ---------- Geometry encoders ----------


@register_encoder("geometry_msgs/msg/Twist")
def _enc_twist(
    names: List[str], action_vec: Sequence[float], clamp: Optional[Tuple[float, float]]
):
    """Twist encoder with sensible defaults when names are absent."""
    if names:
        return _encode_via_dotted_paths(
            "geometry_msgs/msg/Twist", names, action_vec, clamp
        )

    # Default mapping when no names specified
    msg_cls = get_message("geometry_msgs/msg/Twist")
    msg = msg_cls()

    # Apply clamping if specified
    arr = np.asarray(action_vec, dtype=np.float32).reshape(-1)
    if clamp:
        arr = np.clip(arr, clamp[0], clamp[1])

    # Map to twist fields (linear.x, linear.y, linear.z, angular.x, angular.y, angular.z)
    if len(arr) >= 1:
        msg.linear.x = float(arr[0])
    if len(arr) >= 2:
        msg.angular.z = float(arr[1])  # Common pattern: linear.x, angular.z
    if len(arr) >= 3:
        msg.linear.y = float(arr[2])
    if len(arr) >= 4:
        msg.linear.z = float(arr[3])
    if len(arr) >= 5:
        msg.angular.x = float(arr[4])
    if len(arr) >= 6:
        msg.angular.y = float(arr[5])

    return msg


# ---------- Array encoders ----------


@register_encoder("std_msgs/msg/Float32MultiArray")
def _enc_f32_array(
    names: List[str], action_vec: Sequence[float], clamp: Optional[Tuple[float, float]]
):
    if names:
        return _encode_via_dotted_paths(
            "std_msgs/msg/Float32MultiArray", names, action_vec, clamp
        )
    msg_cls = get_message("std_msgs/msg/Float32MultiArray")
    msg = msg_cls()
    arr = np.asarray(action_vec, dtype=np.float32).reshape(-1)
    if clamp:
        arr = np.clip(arr, clamp[0], clamp[1])
    msg.data = [float(x) for x in arr.tolist()]
    return msg


@register_encoder("std_msgs/msg/Int32MultiArray")
def _enc_i32_array(
    names: List[str], action_vec: Sequence[float], clamp: Optional[Tuple[float, float]]
):
    if names:
        return _encode_via_dotted_paths(
            "std_msgs/msg/Int32MultiArray", names, action_vec, clamp
        )
    msg_cls = get_message("std_msgs/msg/Int32MultiArray")
    msg = msg_cls()
    arr = np.asarray(action_vec, dtype=np.float32).reshape(-1)
    if clamp:
        arr = np.clip(arr, clamp[0], clamp[1])
    msg.data = [int(round(float(x))) for x in arr.tolist()]
    return msg
