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
import os
import cv2  
import tempfile
from pathlib import Path

import pandas as pd

import av
from sensor_msgs.msg import Image
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
    #print("decode ros image")
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
        #print("reshaping")
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
    #hwc_float = hwc_rgb.astype(np.float32) / 255.0  # uint8 [0,255] -> float32 [0,1]
    hwc_float = hwc_rgb.astype(np.uint8)
    #print("returning hwc float")
    return hwc_float




_decoder_state = {}

def create_dummy_image(msg, spec, output_encoding='rgb8'):
    print(f"[FoxgloveDecoder] Skipping frame  (warming up)")
    # Return a black dummy image of reasonable size (e.g., 480x640)
    dummy = Image()
    dummy.header.stamp = msg.timestamp
    dummy.header.frame_id = msg.frame_id

    if spec.image_resize:
        h, w = spec.image_resize
    else:
        h, w = 720, 1280  

    dummy.height = int(h)
    dummy.width = int(w)
    dummy.encoding = output_encoding
    dummy.is_bigendian = 0
    dummy.step = dummy.width * 3
    dummy.data = (np.zeros((dummy.height, dummy.width, 3), dtype=np.uint8)).tobytes()
    return dummy

def decode_foxglove_compressed_video(msg, spec, output_encoding='rgb8', warmup_frames=30):
    """
    Decode foxglove_msgs/CompressedVideo into a sensor_msgs/Image message using PyAV.
    """
    codec_name = getattr(msg, "format", "h264")
    topic = spec.topic
    

    if topic not in _decoder_state:
        try:
            codec_ctx = av.codec.CodecContext.create(codec_name, 'r')
            _decoder_state[topic] = {
                'codec_ctx': codec_ctx, 
                'prev_image': None,
            }
            print(f"[FoxgloveDecoder] Initialized decoder for topic {spec.topic}")
        except Exception as e:
            print(f"[FoxgloveDecoder] Failed to initialize decoder for {codec_name}: {e}")
            return None

    state = _decoder_state[topic]
    codec_ctx = state['codec_ctx']
    prev_image = state['prev_image']

    
    # Wrap raw bytes from msg.data as a PyAV packet
    try:
        packet = av.packet.Packet(bytes(msg.data))
    except Exception as e:
        print(f'Failed to create AV packet: {e}')
        return prev_image
    
    try:
        frames = codec_ctx.decode(packet)
    except Exception as e:
        #dummy = create_dummy_image(msg) if prev_image is None else prev_image
        dummy = create_dummy_image(msg, spec)
        print(f'Decode error: {e}')
        return dummy
    

    if not frames:
        print("[FoxgloveDecoder] No frames decoded (maybe waiting for keyframe)")
        dummy = create_dummy_image(msg, spec)
        return dummy

    # Usually only one frame per packet
    frame = frames[0]
    try:
        img_rgb = frame.to_ndarray(format='rgb24')
        #print("valid frame")
    except Exception as e:
        print(f"[FoxgloveDecoder] Failed to convert frame: {e}")
        return None

    h, w, ch = img_rgb.shape
    if ch != 3:
        print(f"[FoxgloveDecoder] Unexpected channel count {ch}, expecting 3 (rgb24)")

    ros_img = Image()
    ros_img.header.stamp = msg.timestamp
    ros_img.header.frame_id = msg.frame_id
    ros_img.height = h
    ros_img.width = w
    ros_img.encoding = output_encoding
    ros_img.is_bigendian = 0
    ros_img.step = w * 3
    ros_img.data = img_rgb.tobytes()

    prev_image = ros_img
    return ros_img

@register_decoder("foxglove_msgs/msg/CompressedVideo")
def _dec_foxglove_image(msg, spec):
    """Foxglove CompressedVideo decoder: decode H.264 then use image decoder.
    
    This decoder:
    1. Decodes the H.264 frame from the Foxglove message
    2. Passes the resulting image to the standard image decoder (decode_ros_image)
    3. Returns the same normalized float32 array as sensor_msgs/Image decoding
    
    This ensures consistent processing pipeline between Foxglove and standard ROS images.
    """

    decoded_image = decode_foxglove_compressed_video(msg, spec, output_encoding='rgb8')
    
    normalized_image = decode_ros_image(decoded_image)

    return normalized_image


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

@register_decoder("sensor_msgs/msg/NavSatFix")
def _dec_navsatfix(msg, spec):
    """NavSatFix decoder: try dotted names first, then default (lat, lon, alt)."""
    if spec.names:
        return _decode_via_names(msg, spec.names)

    return np.asarray(
        [
            msg.latitude,
            msg.longitude,
            msg.altitude,
        ],
        dtype=np.float32,
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

@register_decoder("geometry_msgs/msg/TwistStamped")
def _dec_twist_stamped(msg, spec):
    """
    TwistStamped decoder: Extrae los 6 valores de velocidad (linear y angular) 
    del campo 'twist'. El 'header' se maneja por separado para el resampling.
    """
    # 1. Manejar la decodificación personalizada si se especifican dotted names
    if spec.names:
        # Aquí, 'msg' es el mensaje TwistStamped completo
        # La función _decode_via_names debe ser capaz de manejar paths como 
        # 'twist.linear.x'
        return _decode_via_names(msg, spec.names)

    # 2. Decodificación por defecto: Concatenar los 6 valores (linear + angular)
    #    accediendo al campo 'twist' del mensaje.
    
    # Referencia al mensaje Twist incrustado
    twist_msg = msg.twist

    # Concatenar velocidades lineales (x, y, z) y angulares (x, y, z)
    return np.concatenate(
        [
            np.asarray([twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z], dtype=np.float32),
            np.asarray([twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z], dtype=np.float32),
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

# ----- Motors -----

@register_decoder("can_msgs/msg/MotorsRPM")
def _dec_motors_rpm(msg, spec):
    """MotorsRPM decoder: try dotted names first, then default (FR, RR, RL, FL)."""
    
    if spec.names:
        return _decode_via_names(msg, spec.names)

    return np.asarray(
        [
            msg.rpms_fr,
            msg.rpms_rr,
            msg.rpms_rl,
            msg.rpms_fl,
        ],
        dtype=np.float32,
    )


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
