# rosetta/common/decoders.py
from __future__ import annotations

"""
ROS message decoders for converting ROS messages to numpy arrays and Python types.

This module contains all registered decoders for converting ROS messages into
forms suitable for policy inference. Decoders are registered using the
@register_decoder decorator and called via decode_value() in processing_utils.py.

Supported message types:
- sensor_msgs/msg/Image: Convert to HxWx3 uint8 RGB arrays
- std_msgs/msg/Float32MultiArray: Convert to float32 numpy arrays
- std_msgs/msg/Int32MultiArray: Convert to int32 numpy arrays  
- std_msgs/msg/String: Convert to Python strings
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from rosetta.common.contract_utils import register_decoder
import tempfile
import subprocess

import os
import cv2  

from rclpy.serialization import deserialize_message
from foxglove_msgs.msg import CompressedVideo
# ---------- Helper functions ----------


def dot_get(obj, path: str):
    """
    Resolve a dotted attribute path on a ROS message or nested object.

    Supports a special JointState-style pattern: "<field>.<joint_name>".
    Example:
        path = "position.elbow_joint"
        -> looks up index of "elbow_joint" inside msg.name and returns position[idx]
    """
    parts = path.split(".")
    # JointState-like fast path
    if len(parts) == 2 and hasattr(obj, "name") and hasattr(obj, parts[0]):
        field, key = parts
        try:
            idx = list(obj.name).index(key)
            return getattr(obj, field)[idx]
        except Exception:
            raise

    # Generic nested getattr walk
    cur = obj
    for p in parts:
        cur = getattr(cur, p)
    return cur


def _nearest_resize_rgb(img: np.ndarray, rh: int, rw: int) -> np.ndarray:
    """Pure-numpy nearest-neighbor resize for HxWxC arrays (uint8)."""
    if img.shape[0] == rh and img.shape[1] == rw:
        return img
    y = np.clip(np.linspace(0, img.shape[0] - 1, rh), 0, img.shape[0] - 1).astype(
        np.int64
    )
    x = np.clip(np.linspace(0, img.shape[1] - 1, rw), 0, img.shape[1] - 1).astype(
        np.int64
    )
    return img[y][:, x]


def _nearest_resize_any(img: np.ndarray, rh: int, rw: int) -> np.ndarray:
    # img is HxW or HxWxC
    H, W = img.shape[:2]
    if H == rh and W == rw:
        return img
    y = np.clip(np.linspace(0, H - 1, rh), 0, H - 1).astype(np.int64)
    x = np.clip(np.linspace(0, W - 1, rw), 0, W - 1).astype(np.int64)
    if img.ndim == 2:
        return img[np.ix_(y, x)]
    else:  # HxWxC
        return img[y][:, x, :]


def decode_ros_image(
    msg,
    expected_encoding: Optional[str] = None,
    resize_hw: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Decode ROS image to numpy array in HWC format.

    For depth images:
        - Returns normalized depth [0,1] for valid measurements (capped at 50m)
        - Preserves REP 117 special values: -Inf (too close), NaN (invalid), +Inf (no return)
        - 3 channels (HxWx3) replicated for LeRobot compatibility
    For color images: returns [0,1] normalized RGB, 3 channels
    For grayscale images: returns [0,1] normalized, 1 channel

    Returns:
        np.ndarray: Shape (H, W, C) with dtype float32
    """
    h, w = int(msg.height), int(msg.width)
    enc = (getattr(msg, "encoding", None) or expected_encoding or "bgr8").lower()
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    step = int(getattr(msg, "step", 0))

    # --- Depth: canonical float32 meters ---
    if enc in ("32fc1", "32fc"):
        data32 = raw.view(np.float32)
        row_elems = (step // 4) if step else w
        arr = data32.reshape(h, row_elems)[:, :w].reshape(h, w)  # HxW float32 meters
        hwc = arr[..., None]  # HxWx1
        if resize_hw:
            rh, rw = int(resize_hw[0]), int(resize_hw[1])
            hwc = _nearest_resize_any(hwc, rh, rw)
        # Normalize depth to [0,1] while preserving REP 117 special values (NaN, ±Inf)
        hwc_normalized = np.where(
            np.isfinite(hwc),
            np.clip(hwc, 0, 50) / 50,  # Cap at 50m, normalize to [0,1]
            hwc,  # Preserve NaN, -Inf, +Inf
        )
        hwc_3ch = np.repeat(hwc_normalized, 3, axis=-1)  # H'xW'x3
        return hwc_3ch.astype(np.float32)

    # --- Depth: OpenNI raw 16UC1 (millimeters) ---
    elif enc in ("16uc1", "mono16"):
        data16 = raw.view(np.uint16)
        row_elems = (step // 2) if step else w
        arr16 = data16.reshape(h, row_elems)[:, :w].reshape(h, w)
        # 0 -> invalid depth -> NaN
        arr_m = arr16.astype(np.float32)
        arr_m[arr16 == 0] = np.nan
        arr_m[arr16 != 0] *= 1.0 / 1000.0  # mm -> m
        hwc = arr_m[..., None]  # HxWx1
        if resize_hw:
            rh, rw = int(resize_hw[0]), int(resize_hw[1])
            hwc = _nearest_resize_any(hwc, rh, rw)
        # Normalize depth to [0,1] while preserving REP 117 special values (NaN, ±Inf)
        hwc_normalized = np.where(
            np.isfinite(hwc),
            np.clip(hwc, 0, 10) / 10,  # Cap at 50m, normalize to [0,1]
            hwc,  # Preserve NaN, -Inf, +Inf
        )
        hwc_3ch = np.repeat(hwc_normalized, 3, axis=-1)  # H'xW'x3
        return hwc_3ch.astype(np.float32)

    # --- Grayscale 8-bit ---
    elif enc in ("mono8", "8uc1", "uint8"):
        if not step:
            step = max(w, 1)
        arr = raw.reshape(h, step)[:, :w].reshape(h, w)
        # keep intensity in [0,255] -> normalize to [0,1] float for vision models
        hwc = (arr.astype(np.float32) / 255.0)[..., None]  # HxWx1
        if resize_hw:
            rh, rw = int(resize_hw[0]), int(resize_hw[1])
            hwc = _nearest_resize_any(hwc, rh, rw)
        # Replicate to 3 channels for LeRobot compatibility (like depth images)
        hwc_3ch = np.repeat(hwc, 3, axis=-1)  # H'xW'x3
        return hwc_3ch.astype(np.float32)

    # --- Color paths (unchanged behavior) ---
    elif enc in ("rgb8", "bgr8"):
        ch = 3
        row = raw.reshape(h, step)[:, : w * ch]
        arr = row.reshape(h, w, ch)
        hwc_rgb = arr if enc == "rgb8" else arr[..., ::-1]
    elif enc in ("rgba8", "bgra8"):
        ch = 4
        row = raw.reshape(h, step)[:, : w * ch]
        arr = row.reshape(h, w, ch)
        rgb = arr[..., :3]
        hwc_rgb = rgb if enc == "rgba8" else rgb[..., ::-1]
    else:
        raise ValueError(f"Unsupported image encoding '{enc}'")

    # Color processing: resize and normalize to [0,1]
    if resize_hw:
        rh, rw = int(resize_hw[0]), int(resize_hw[1])
        hwc_rgb = _nearest_resize_rgb(hwc_rgb, rh, rw)

    # Normalize to [0,1] and keep HWC format for LeRobot compatibility
    hwc_float = hwc_rgb.astype(np.float32) / 255.0  # uint8 [0,255] -> float32 [0,1]

    return hwc_float

import pandas as pd
_timestamp_map = None
def _load_timestamp_map(csv_path): #to load csv just once
    global _timestamp_map
    if _timestamp_map is None:
        df = pd.read_csv(csv_path)
        _timestamp_map = df
    return _timestamp_map

def _find_closest_image(images_dir, ts, df, tolerance=0.2):
    """
    Busca la imagen cuyo timestamp esté más cerca de 'ts'.
    tolerance en segundos.
    """
    df["diff"] = np.abs(df["timestamp"] - ts)
    closest = df.loc[df["diff"].idxmin()]
    if closest["diff"] > tolerance:
        print(f" No close image found for {ts:.6f}, closest diff={closest['diff']:.3f}s")
    return os.path.join(images_dir, closest["image_name"])

def flatten_ros_message(msg, prefix=""):
    """Recursively flatten ROS2 messages into a flat dict."""
    result = {}
    # Check if the message has fields and field types (to handle complex types)
    if not hasattr(msg, "get_fields_and_field_types"):
        # Handle simple types (should not happen if called correctly, but for robustness)
        return {}

    for field_name, field_type in msg.get_fields_and_field_types().items():
        value = getattr(msg, field_name)
        key = f"{prefix}{field_name}"
        if hasattr(value, "get_fields_and_field_types"):
            # Recurse for nested messages
            result.update(flatten_ros_message(value, prefix=key + "."))
        elif isinstance(value, (list, tuple, np.ndarray)):
            # Handle arrays/lists/tuples
            result[key] = list(value)
        else:
            # Handle primitive types
            result[key] = value
    return result

@register_decoder("foxglove_msgs/msg/CompressedVideo")
def _dec_foxglove_image(msg, spec):
    """
    Custome decoder for Compressed Images. Search image in image folder based on timestamp
    """

    topic = spec.topic.strip("/").replace("/", "_")

    IMAGES_DIR = f"/workspace/data/monday_trimmed_dir/data_0/images/{topic}/images"
    CSV_PATH = f"/workspace/data/monday_trimmed_dir/data_0/images/{topic}/timestamps.csv"
    
    ts = msg.timestamp.sec + msg.timestamp.nanosec * 1e-9

    df = _load_timestamp_map(CSV_PATH)
    # no puedo asumir que para cada timestamp va a haber una imagen
    image_path = _find_closest_image(IMAGES_DIR, ts, df)

    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError(f"Could not load image at {image_path}")

    # 4️Convertir a RGB y normalizar [0,1]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_norm = frame_rgb.astype(np.float32) / 255.0

    return frame_norm



# ---------- Image decoders ----------


@register_decoder("sensor_msgs/msg/Image")
def _dec_image(msg, spec):
    """Image decoder: try dotted names first, then decode as image."""
    if spec.names:
        return _decode_via_names(msg, spec.names)
    return decode_ros_image(msg, spec.image_encoding, spec.image_resize)


# ---------- Array decoders ----------


@register_decoder("std_msgs/msg/Float32MultiArray")
def _dec_f32(msg, spec):
    """Float32MultiArray decoder: try dotted names first, then use data field."""
    if spec.names:
        return _decode_via_names(msg, spec.names)
    return np.asarray(msg.data, dtype=np.float32)


@register_decoder("std_msgs/msg/Int32MultiArray")
def _dec_i32(msg, spec):
    """Int32MultiArray decoder: try dotted names first, then use data field."""
    if spec.names:
        return _decode_via_names(msg, spec.names)
    return np.asarray(msg.data, dtype=np.int32)


# ---------- String decoders ----------


@register_decoder("std_msgs/msg/String")
def _dec_str(msg, spec):
    """String decoder: try dotted names first, then use data field."""
    if spec.names:
        return _decode_via_names(msg, spec.names)
    return str(getattr(msg, "data", ""))


# ---------- Joint state decoder ----------


@register_decoder("sensor_msgs/msg/JointState")
def _dec_joint_state(msg, spec):
    """JointState decoder: try dotted names first, then use default behavior."""
    if spec.names:
        return _decode_via_names(msg, spec.names)
    # Default: return position data if available, otherwise empty array
    if hasattr(msg, "position") and msg.position:
        return np.asarray(msg.position, dtype=np.float32)
    return np.array([], dtype=np.float32)


@register_decoder("sensor_msgs/msg/Imu")
def _dec_imu(msg, spec):
    """IMU decoder: try dotted names first, then use default behavior."""
    if spec.names:
        return _decode_via_names(msg, spec.names)
    # Default: return orientation quaternion + angular velocity + linear acceleration
    return np.concatenate(
        [
            np.asarray(
                [
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                    msg.orientation.w,
                ],
                dtype=np.float32,
            ),
            np.asarray(
                [
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z,
                ],
                dtype=np.float32,
            ),
            np.asarray(
                [
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z,
                ],
                dtype=np.float32,
            ),
        ]
    )


@register_decoder("nav_msgs/msg/Odometry")
def _dec_odometry(msg, spec):
    """Odometry decoder: try dotted names first, then use default behavior."""
    if spec.names:
        return _decode_via_names(msg, spec.names)
    # Default: return position + orientation quaternion
    return np.concatenate(
        [
            np.asarray(
                [
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                ],
                dtype=np.float32,
            ),
            np.asarray(
                [
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w,
                ],
                dtype=np.float32,
            ),
        ]
    )


@register_decoder("geometry_msgs/msg/Twist")
def _dec_twist(msg, spec):
    """Twist decoder: try dotted names first, then use default behavior."""
    if spec.names:
        return _decode_via_names(msg, spec.names)
    # Default: return linear and angular velocities
    return np.concatenate(
        [
            np.asarray([msg.linear.x, msg.linear.y, msg.linear.z], dtype=np.float32),
            np.asarray([msg.angular.x, msg.angular.y, msg.angular.z], dtype=np.float32),
        ]
    )


@register_decoder("control_msgs/msg/MultiDOFCommand")
def _dec_multidof_command(msg, spec):
    """MultiDOFCommand decoder: try dotted names first, then use default behavior."""
    if spec.names:
        return _decode_via_names(msg, spec.names)

    # Default: return values and values_dot concatenated
    values = (
        np.asarray(msg.values, dtype=np.float32)
        if msg.values
        else np.array([], dtype=np.float32)
    )
    values_dot = (
        np.asarray(msg.values_dot, dtype=np.float32)
        if msg.values_dot
        else np.array([], dtype=np.float32)
    )
    return np.concatenate([values, values_dot])


# ---------- Generic fallback decoder ----------


def _decode_via_names(msg, names: List[str]) -> Optional[np.ndarray]:
    """Fallback: sample scalar fields using dotted selectors into a float32 vector."""
    if not names:
        return None

    # Special handling for MultiDOFCommand messages
    if (
        hasattr(msg, "dof_names")
        and hasattr(msg, "values")
        and hasattr(msg, "values_dot")
    ):
        return _decode_multidof_via_names(msg, names)

    out: List[float] = []
    for name in names:
        try:
            out.append(float(dot_get(msg, name)))
        except Exception:
            out.append(float("nan"))
    return np.asarray(out, dtype=np.float32)


def _decode_multidof_via_names(msg, names: List[str]) -> np.ndarray:
    """Special decoder for MultiDOFCommand messages with values. and values_dot. prefixes."""
    out: List[float] = []

    for name in names:
        try:
            if name.startswith("values_dot."):
                # Extract DOF name from "values_dot.dof_name"
                dof_name = name[11:]  # Remove "values_dot." prefix
                if dof_name in msg.dof_names:
                    idx = msg.dof_names.index(dof_name)
                    if idx < len(msg.values_dot):
                        out.append(float(msg.values_dot[idx]))
                    else:
                        out.append(0.0)
                else:
                    out.append(0.0)
            elif name.startswith("values."):
                # Extract DOF name from "values.dof_name"
                dof_name = name[7:]  # Remove "values." prefix
                if dof_name in msg.dof_names:
                    idx = msg.dof_names.index(dof_name)
                    if idx < len(msg.values):
                        out.append(float(msg.values[idx]))
                    else:
                        out.append(0.0)
                else:
                    out.append(0.0)
            else:
                # Default to values field
                if name in msg.dof_names:
                    idx = msg.dof_names.index(name)
                    if idx < len(msg.values):
                        out.append(float(msg.values[idx]))
                    else:
                        out.append(0.0)
                else:
                    out.append(0.0)
        except Exception:
            out.append(float("nan"))

    return np.asarray(out, dtype=np.float32)


# Note: All decoders now handle dotted names internally, so no generic decoder needed
