# rosetta/common/decoders.py
from __future__ import annotations

"""
ROS message decoders for converting ROS messages to numpy arrays and Python types.

This module contains all registered decoders for converting ROS messages into
forms suitable for policy inference. Decoders are registered using the
@register_decoder decorator and called via decode_value() in processing_utils.py.

Image resizing uses NumPy nearest-neighbor interpolation, which handles all dtypes
and special values (NaN, Inf) correctly.

Supported message types:
- sensor_msgs/msg/Image: Convert to HxWx3 uint8 RGB arrays
- sensor_msgs/msg/CompressedImage: Decompress and convert to HxWx3 RGB arrays (requires opencv-python)
- std_msgs/msg/Float32MultiArray: Convert to float32 numpy arrays
- std_msgs/msg/Int32MultiArray: Convert to int32 numpy arrays  
- std_msgs/msg/String: Convert to Python strings
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Guarded import for OpenCV (used for CompressedImage)
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

from rosetta.common.contract_utils import register_decoder


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


def _nearest_resize_numpy(img: np.ndarray, rh: int, rw: int) -> np.ndarray:
    """Pure-numpy nearest-neighbor resize fallback.
    
    Args:
        img: Image array in HxW or HxWxC format
        rh: Target height
        rw: Target width
        
    Returns:
        Resized image array
    """
    H, W = img.shape[:2]
    if H == rh and W == rw:
        return img
    y = np.clip(np.linspace(0, H - 1, rh), 0, H - 1).astype(np.int64)
    x = np.clip(np.linspace(0, W - 1, rw), 0, W - 1).astype(np.int64)
    if img.ndim == 2:
        return img[np.ix_(y, x)]
    else:  # HxWxC
        return img[y][:, x, :]


def resize_image(img: np.ndarray, rh: int, rw: int, interpolation: str = "nearest") -> np.ndarray:
    """
    Resize image using NumPy nearest-neighbor interpolation.
    
    Args:
        img: Image array in HxW or HxWxC format (any dtype)
        rh: Target height
        rw: Target width
        interpolation: Interpolation mode (only 'nearest' is supported)
            
    Returns:
        Resized image array with same dtype and channel count as input
    """
    # Only supports nearest-neighbor interpolation
    if interpolation != "nearest":
        import warnings
        warnings.warn(
            f"NumPy resize only supports 'nearest' interpolation, "
            f"got '{interpolation}'. Using nearest-neighbor.",
            UserWarning
        )
    return _nearest_resize_numpy(img, rh, rw)


# Backward compatibility aliases (using the new pluggable resize)
def _nearest_resize_rgb(img: np.ndarray, rh: int, rw: int) -> np.ndarray:
    """Legacy alias: RGB resize using pluggable fast path."""
    return resize_image(img, rh, rw, interpolation="nearest")


def _nearest_resize_any(img: np.ndarray, rh: int, rw: int) -> np.ndarray:
    """Legacy alias: generic resize using pluggable fast path."""
    return resize_image(img, rh, rw, interpolation="nearest")


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
            hwc  # Preserve NaN, -Inf, +Inf
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
            np.clip(hwc, 0, 50) / 50,  # Cap at 50m, normalize to [0,1]
            hwc  # Preserve NaN, -Inf, +Inf
        )
        hwc_3ch = np.repeat(hwc_normalized, 3, axis=-1)  # H'xW'x3
        return hwc_3ch.astype(np.float32)

    # --- Grayscale 8-bit ---
    elif enc in ("mono8", "8uc1", "uint8"):
        if not step: step = max(w, 1)
        arr = raw.reshape(h, step)[:, :w].reshape(h, w)
        # keep intensity in [0,255] -> normalize to [0,1] float for vision models
        hwc = (arr.astype(np.float32) / 255.0)[..., None]  # HxWx1
        if resize_hw:
            rh, rw = int(resize_hw[0]), int(resize_hw[1])
            hwc = _nearest_resize_any(hwc, rh, rw)
        # Replicate to 3 channels for LeRobot compatibility (like depth images)
        hwc_3ch = np.repeat(hwc, 3, axis=-1)  # H'xW'x3
        return hwc_3ch.astype(np.float32)

    # --- Color paths ---
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
    # --- Bayer encodings: convert to grayscale (demosaic not yet implemented) ---
    elif enc.startswith("bayer_"):
        # Bayer encodings are single-channel raw mosaics; convert to grayscale
        if not step: step = max(w, 1)
        arr = raw.reshape(h, step)[:, :w].reshape(h, w)
        hwc = (arr.astype(np.float32) / 255.0)[..., None]  # HxWx1
        if resize_hw:
            rh, rw = int(resize_hw[0]), int(resize_hw[1])
            hwc = _nearest_resize_any(hwc, rh, rw)
        # Replicate to 3 channels for LeRobot compatibility
        hwc_3ch = np.repeat(hwc, 3, axis=-1)  # H'xW'x3
        return hwc_3ch.astype(np.float32)
    # --- Unknown encodings: attempt grayscale fallback ---
    else:
        # Fallback: try to interpret as grayscale 8-bit
        if not step: step = max(w, 1)
        arr = raw.reshape(h, step)[:, :w].reshape(h, w)
        hwc = (arr.astype(np.float32) / 255.0)[..., None]  # HxWx1
        if resize_hw:
            rh, rw = int(resize_hw[0]), int(resize_hw[1])
            hwc = _nearest_resize_any(hwc, rh, rw)
        # Replicate to 3 channels for LeRobot compatibility
        hwc_3ch = np.repeat(hwc, 3, axis=-1)  # H'xW'x3
        return hwc_3ch.astype(np.float32)

    # Color processing: resize and normalize to [0,1]
    if resize_hw:
        rh, rw = int(resize_hw[0]), int(resize_hw[1])
        hwc_rgb = _nearest_resize_rgb(hwc_rgb, rh, rw)

    # Normalize to [0,1] and keep HWC format for LeRobot compatibility
    hwc_float = hwc_rgb.astype(np.float32) / 255.0  # uint8 [0,255] -> float32 [0,1]

    return hwc_float


# ---------- Image decoders ----------


@register_decoder("sensor_msgs/msg/Image")
def _dec_image(msg, spec):
    """Image decoder: try dotted names first, then decode as image."""
    if spec.names:
        return _decode_via_names(msg, spec.names)
    return decode_ros_image(msg, spec.image_encoding, spec.image_resize)


@register_decoder("sensor_msgs/msg/CompressedImage")
def _dec_compressed_image(msg, spec):
    """CompressedImage decoder: decompress using cv2 if available."""
    if spec.names:
        return _decode_via_names(msg, spec.names)
    
    if not _CV2_AVAILABLE:
        raise RuntimeError(
            "sensor_msgs/CompressedImage requires OpenCV (cv2). "
            "Install with: pip install opencv-python"
        )
    
    # Decompress image data
    compressed_data = np.frombuffer(msg.data, dtype=np.uint8)
    # cv2.imdecode expects a NumPy array and returns BGR format
    # Type ignore needed because cv2 is conditionally imported
    img_bgr = cv2.imdecode(compressed_data, cv2.IMREAD_COLOR)  # type: ignore[attr-defined]
    
    if img_bgr is None:
        raise ValueError(
            f"Failed to decode CompressedImage with format '{msg.format}'"
        )
    
    # Convert BGR to RGB for consistency with other decoders
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # type: ignore[attr-defined]
    
    # Resize if needed (use pluggable resize for consistency)
    if spec.image_resize:
        rh, rw = int(spec.image_resize[0]), int(spec.image_resize[1])
        img_rgb = resize_image(img_rgb, rh, rw, interpolation="nearest")
    
    # Normalize to [0,1] and return HWC format
    hwc_float = img_rgb.astype(np.float32) / 255.0
    return hwc_float


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
    return np.concatenate([
        np.asarray([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w], dtype=np.float32),
        np.asarray([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], dtype=np.float32),
        np.asarray([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z], dtype=np.float32)
    ])


@register_decoder("nav_msgs/msg/Odometry")
def _dec_odometry(msg, spec):
    """Odometry decoder: try dotted names first, then use default behavior."""
    if spec.names:
        return _decode_via_names(msg, spec.names)
    # Default: return position + orientation quaternion
    return np.concatenate([
        np.asarray([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z], dtype=np.float32),
        np.asarray([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w], dtype=np.float32)
    ])


@register_decoder("geometry_msgs/msg/Twist")
def _dec_twist(msg, spec):
    """Twist decoder: try dotted names first, then use default behavior."""
    if spec.names:
        return _decode_via_names(msg, spec.names)
    # Default: return linear and angular velocities
    return np.concatenate([
        np.asarray([msg.linear.x, msg.linear.y, msg.linear.z], dtype=np.float32),
        np.asarray([msg.angular.x, msg.angular.y, msg.angular.z], dtype=np.float32)
    ])


@register_decoder("control_msgs/msg/MultiDOFCommand")
def _dec_multidof_command(msg, spec):
    """MultiDOFCommand decoder: try dotted names first, then use default behavior."""
    if spec.names:
        return _decode_via_names(msg, spec.names)
    
    # Default: return values and values_dot concatenated
    values = np.asarray(msg.values, dtype=np.float32) if msg.values else np.array([], dtype=np.float32)
    values_dot = np.asarray(msg.values_dot, dtype=np.float32) if msg.values_dot else np.array([], dtype=np.float32)
    return np.concatenate([values, values_dot])


# ---------- Generic fallback decoder ----------


def _decode_via_names(msg, names: List[str]) -> Optional[np.ndarray]:
    """Fallback: sample scalar fields using dotted selectors into a float32 vector."""
    if not names:
        return None
    
    # Special handling for MultiDOFCommand messages
    if hasattr(msg, 'dof_names') and hasattr(msg, 'values') and hasattr(msg, 'values_dot'):
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
