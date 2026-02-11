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
ROS message decoders for converting ROS messages to numpy arrays.

Each decoder is self-contained and registered with @register_decoder.
If you need to decode a message type that isn't here, add a new decoder.

Decoders declare their output dtype at registration time. This is the
single source of truth for what LeRobot dtype the decoder produces.

Image Encoding Support
----------------------
All images are normalized to HWC uint8 RGB format for LeRobot compatibility.

Supported encodings (from sensor_msgs/image_encodings.h):
  - rgb8, bgr8: 8-bit 3-channel color
  - rgba8, bgra8: 8-bit 4-channel color (alpha dropped)
  - mono8, 8uc1: 8-bit grayscale (replicated to 3 channels)

Depth encodings (mono16, 16uc1, 32fc1) are NOT supported and will raise
DepthEncodingNotSupported. LeRobot does not currently have proper depth
image handling - it forces all images through RGB conversion which causes
precision loss. See DEPTH_ENCODINGS for details.

To add a new encoding:
  1. Add it to the appropriate category in IMAGE_ENCODINGS
  2. Add decoding logic in _decode_image_by_encoding()
  3. Test with actual sensor data
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np

from .converters import register_decoder
from .contract import DEPTH_ENCODINGS, ObservationStreamSpec
from .ros2_utils import dot_get

# Optional cv2 for compressed image decoding (falls back to PIL)
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    _HAS_CV2 = False

# PIL is the fallback for compressed images
from PIL import Image


# =============================================================================
# Image Encoding Configuration
# =============================================================================

# Supported image encodings grouped by category
# To add support for a new encoding, add it here and implement in _decode_image_by_encoding
IMAGE_ENCODINGS = {
    # 8-bit color (3 channels)
    "color_8bit_3ch": {"rgb8", "bgr8"},
    # 8-bit color with alpha (4 channels, alpha dropped)
    "color_8bit_4ch": {"rgba8", "bgra8"},
    # 8-bit grayscale (1 channel, replicated to 3)
    "mono_8bit": {"mono8", "8uc1"},
}

# Flatten for quick lookup
SUPPORTED_IMAGE_ENCODINGS = frozenset(
    enc for encs in IMAGE_ENCODINGS.values() for enc in encs
)


# =============================================================================
# Image Decoding Helpers
# =============================================================================


def _nearest_resize(img: np.ndarray, rh: int, rw: int) -> np.ndarray:
    """Pure-numpy nearest-neighbor resize for HxW or HxWxC arrays."""
    H, W = img.shape[:2]
    if H == rh and W == rw:
        return img
    y = np.linspace(0, H - 1, rh).astype(np.int64)
    x = np.linspace(0, W - 1, rw).astype(np.int64)
    # Works for both 2D (HxW) and 3D (HxWxC) arrays
    return img[y][:, x]


def _mono_to_rgb(arr: np.ndarray) -> np.ndarray:
    """Convert HxW grayscale to HxWx3 RGB by replication."""
    return np.repeat(arr[..., None], 3, axis=-1)


class DepthEncodingNotSupported(ValueError):
    """Raised when a depth image encoding is encountered.

    LeRobot does not currently have proper depth image support.
    See: https://github.com/huggingface/lerobot
    """

    pass


def _decode_image_by_encoding(
    enc: str,
    raw: np.ndarray,
    h: int,
    w: int,
    step: int,
) -> np.ndarray:
    """Decode raw image bytes to HWC uint8 RGB based on encoding.

    Args:
        enc: Image encoding string (lowercase)
        raw: Raw bytes as uint8 array
        h: Image height
        w: Image width
        step: Row stride in bytes (0 = compute from width)

    Returns:
        HWC uint8 RGB array (h, w, 3)

    Raises:
        DepthEncodingNotSupported: If encoding is a depth format
        ValueError: If encoding is not supported
    """
    # --- Depth encodings - not supported ---
    if enc in DEPTH_ENCODINGS:
        raise DepthEncodingNotSupported(
            f"Depth image encoding '{enc}' is not supported. "
            f"LeRobot does not currently have proper depth image handling - it forces all images "
            f"through RGB conversion which causes precision loss for depth data. "
            f"Remove this observation from your contract or wait for LeRobot depth support."
        )

    # --- 8-bit 3-channel color ---
    if enc in IMAGE_ENCODINGS["color_8bit_3ch"]:
        ch = 3
        if not step:
            step = w * ch
        row = raw.reshape(h, step)[:, :w * ch]
        arr = row.reshape(h, w, ch)
        if enc == "bgr8":
            arr = arr[..., ::-1].copy()  # BGR -> RGB
        return arr.astype(np.uint8)

    # --- 8-bit 4-channel color (drop alpha) ---
    if enc in IMAGE_ENCODINGS["color_8bit_4ch"]:
        ch = 4
        if not step:
            step = w * ch
        row = raw.reshape(h, step)[:, :w * ch]
        arr = row.reshape(h, w, ch)
        rgb = arr[..., :3]
        if enc == "bgra8":
            rgb = rgb[..., ::-1].copy()  # BGR -> RGB
        return rgb.astype(np.uint8)

    # --- 8-bit grayscale ---
    if enc in IMAGE_ENCODINGS["mono_8bit"]:
        if not step:
            step = w
        arr = raw.reshape(h, step)[:, :w]
        return _mono_to_rgb(arr).astype(np.uint8)

    # --- Unsupported encoding ---
    raise ValueError(
        f"Unsupported image encoding: '{enc}'. "
        f"Supported: {sorted(SUPPORTED_IMAGE_ENCODINGS)}"
    )


def decode_ros_image(
    msg,
    expected_encoding: str | None = None,
    resize_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    """Decode ROS Image message to HWC uint8 RGB array.

    Args:
        msg: ROS Image message
        expected_encoding: Fallback encoding if msg.encoding is missing
        resize_hw: Optional (height, width) to resize output

    Returns:
        HWC uint8 RGB array

    Raises:
        DepthEncodingNotSupported: If encoding is a depth format
        ValueError: If encoding is not supported or missing
    """
    h, w = int(msg.height), int(msg.width)
    enc = getattr(msg, "encoding", None) or expected_encoding
    if not enc:
        raise ValueError(
            "Image message has no encoding and no expected_encoding was provided. "
            "Specify encoding in contract image config."
        )
    enc = enc.lower()
    step = int(getattr(msg, "step", 0))
    raw = np.frombuffer(msg.data, dtype=np.uint8)

    # Decode based on encoding
    rgb = _decode_image_by_encoding(enc, raw, h, w, step)

    # Resize if requested
    if resize_hw:
        rgb = _nearest_resize(rgb, int(resize_hw[0]), int(resize_hw[1]))

    return rgb


# =============================================================================
# Image Decoders
# =============================================================================


@register_decoder("sensor_msgs/msg/Image", dtype="video")
def _dec_image(msg: Any, spec: ObservationStreamSpec) -> np.ndarray:
    """Decode sensor_msgs/Image to HWC uint8 RGB."""
    return decode_ros_image(msg, spec.image_encoding, spec.image_resize)


@register_decoder("sensor_msgs/msg/CompressedImage", dtype="video")
def _dec_compressed_image(msg: Any, spec: ObservationStreamSpec) -> np.ndarray:
    """Decode sensor_msgs/CompressedImage to HWC uint8 RGB.

    Supports jpeg, png, and other formats via cv2 or PIL fallback.
    """
    if _HAS_CV2:
        data = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cv2.imdecode failed for format: {msg.format}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.array(Image.open(io.BytesIO(msg.data)).convert("RGB"))

    if spec.image_resize:
        img = _nearest_resize(img, spec.image_resize[0], spec.image_resize[1])

    return img.astype(np.uint8)


# =============================================================================
# JointState Decoder
# =============================================================================


@register_decoder("sensor_msgs/msg/JointState", dtype="float64")
def _dec_joint_state(msg: Any, spec: ObservationStreamSpec) -> np.ndarray:
    """Decode sensor_msgs/JointState.

    With selector names like ["position.joint1", "velocity.joint2"]:
      - Extracts specified fields by joint name lookup
    With bare names like ["joint1", "joint2"]:
      - Defaults to position field
    Without names:
      - Returns all positions
    """
    if not spec.names:
        if hasattr(msg, "position") and msg.position:
            return np.asarray(msg.position, dtype=np.float64)
        return np.array([], dtype=np.float64)

    name_to_idx = {name: i for i, name in enumerate(msg.name)}
    out = []

    for selector in spec.names:
        # Support both "field.joint_name" and bare "joint_name" (defaults to position)
        if "." in selector:
            field, joint_name = selector.split(".", 1)
        else:
            field, joint_name = "position", selector

        if joint_name not in name_to_idx:
            raise ValueError(
                f"Joint '{joint_name}' not in message. Available: {list(msg.name)}"
            )
        idx = name_to_idx[joint_name]
        arr = getattr(msg, field)
        if idx >= len(arr):
            raise ValueError(f"Index {idx} out of range for {field} (len={len(arr)})")
        out.append(float(arr[idx]))

    return np.asarray(out, dtype=np.float64)


# =============================================================================
# IMU Decoder
# =============================================================================


@register_decoder("sensor_msgs/msg/Imu", dtype="float64")
def _dec_imu(msg: Any, spec: ObservationStreamSpec) -> np.ndarray:
    """Decode sensor_msgs/Imu.

    With selector names: extracts specified dotted paths
    Without names: returns [quat(4), angular_vel(3), linear_accel(3)]
    """
    if not spec.names:
        return np.concatenate([
            np.array([
                msg.orientation.x, msg.orientation.y,
                msg.orientation.z, msg.orientation.w
            ], dtype=np.float64),
            np.array([
                msg.angular_velocity.x, msg.angular_velocity.y,
                msg.angular_velocity.z
            ], dtype=np.float64),
            np.array([
                msg.linear_acceleration.x, msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ], dtype=np.float64),
        ])

    return np.asarray([float(dot_get(msg, name)) for name in spec.names], dtype=np.float64)


# =============================================================================
# Odometry Decoder
# =============================================================================


@register_decoder("nav_msgs/msg/Odometry", dtype="float64")
def _dec_odometry(msg: Any, spec: ObservationStreamSpec) -> np.ndarray:
    """Decode nav_msgs/Odometry.

    With selector names: extracts specified dotted paths
    Without names: returns [position(3), orientation_quat(4)]
    """
    if not spec.names:
        return np.concatenate([
            np.array([
                msg.pose.pose.position.x, msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ], dtype=np.float64),
            np.array([
                msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
            ], dtype=np.float64),
        ])

    return np.asarray([float(dot_get(msg, name)) for name in spec.names], dtype=np.float64)


# =============================================================================
# Twist Decoder
# =============================================================================


@register_decoder("geometry_msgs/msg/Twist", dtype="float64")
def _dec_twist(msg: Any, spec: ObservationStreamSpec) -> np.ndarray:
    """Decode geometry_msgs/Twist.

    With selector names: extracts specified dotted paths
    Without names: returns [linear(3), angular(3)]
    """
    if not spec.names:
        return np.concatenate([
            np.array([msg.linear.x, msg.linear.y, msg.linear.z], dtype=np.float64),
            np.array([msg.angular.x, msg.angular.y, msg.angular.z], dtype=np.float64),
        ])

    return np.asarray([float(dot_get(msg, name)) for name in spec.names], dtype=np.float64)


# =============================================================================
# TwistStamped Decoder
# =============================================================================


@register_decoder("geometry_msgs/msg/TwistStamped", dtype="float64")
def _dec_twist_stamped(msg: Any, spec: ObservationStreamSpec) -> np.ndarray:
    """Decode geometry_msgs/TwistStamped.

    Same selector syntax as Twist - the stamped wrapper is transparent.
    With selector names: extracts specified dotted paths from inner twist
    Without names: returns [linear(3), angular(3)]
    """
    if not spec.names:
        return np.concatenate([
            np.array([
                msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z
            ], dtype=np.float64),
            np.array([
                msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z
            ], dtype=np.float64),
        ])

    return np.asarray(
        [float(dot_get(msg.twist, name)) for name in spec.names], dtype=np.float64
    )


# =============================================================================
# MultiDOFCommand Decoder
# =============================================================================


@register_decoder("control_msgs/msg/MultiDOFCommand", dtype="float64")
def _dec_multidof_command(msg: Any, spec: ObservationStreamSpec) -> np.ndarray:
    """Decode control_msgs/MultiDOFCommand.

    With selector names like ["values.joint1", "values_dot.joint1"]:
      - Extracts specified DOF values by name
    Without names:
      - Returns [values, values_dot] concatenated
    """
    if not spec.names:
        values = np.asarray(msg.values, dtype=np.float64) if msg.values else np.array([], dtype=np.float64)
        values_dot = np.asarray(msg.values_dot, dtype=np.float64) if msg.values_dot else np.array([], dtype=np.float64)
        return np.concatenate([values, values_dot])

    dof_index = {name: i for i, name in enumerate(msg.dof_names)}
    out = []

    for selector in spec.names:
        if selector.startswith("values_dot."):
            dof_name = selector[11:]
            arr = msg.values_dot
        elif selector.startswith("values."):
            dof_name = selector[7:]
            arr = msg.values
        else:
            dof_name = selector
            arr = msg.values

        if dof_name not in dof_index:
            raise ValueError(
                f"DOF '{dof_name}' not in message. Available: {list(msg.dof_names)}"
            )
        idx = dof_index[dof_name]
        if idx >= len(arr):
            raise ValueError(f"Index {idx} out of range (len={len(arr)})")
        out.append(float(arr[idx]))

    return np.asarray(out, dtype=np.float64)


# =============================================================================
# Array Decoders
# =============================================================================


@register_decoder("std_msgs/msg/Float32MultiArray", dtype="float32")
def _dec_float32_array(msg: Any, spec: ObservationStreamSpec) -> np.ndarray:
    """Decode std_msgs/Float32MultiArray to float32 array."""
    _ = spec  # Unused - no selector needed for arrays
    return np.asarray(msg.data, dtype=np.float32)


@register_decoder("std_msgs/msg/Float64MultiArray", dtype="float64")
def _dec_float64_array(msg: Any, spec: ObservationStreamSpec) -> np.ndarray:
    """Decode std_msgs/Float64MultiArray to float64 array."""
    _ = spec  # Unused - no selector needed for arrays
    return np.asarray(msg.data, dtype=np.float64)


@register_decoder("std_msgs/msg/Int32MultiArray", dtype="int32")
def _dec_int32_array(msg: Any, spec: ObservationStreamSpec) -> np.ndarray:
    """Decode std_msgs/Int32MultiArray to int32 array."""
    _ = spec  # Unused - no selector needed for arrays
    return np.asarray(msg.data, dtype=np.int32)


# =============================================================================
# Scalar Decoders
# =============================================================================


@register_decoder("std_msgs/msg/Float32", dtype="float32")
def _dec_float32(msg: Any, spec: ObservationStreamSpec) -> np.ndarray:
    """Decode std_msgs/Float32 to float32 scalar."""
    _ = spec  # Unused
    return np.array([msg.data], dtype=np.float32)


@register_decoder("std_msgs/msg/Float64", dtype="float64")
def _dec_float64(msg: Any, spec: ObservationStreamSpec) -> np.ndarray:
    """Decode std_msgs/Float64 to float64 scalar."""
    _ = spec  # Unused
    return np.array([msg.data], dtype=np.float64)


@register_decoder("std_msgs/msg/Int32", dtype="int32")
def _dec_int32(msg: Any, spec: ObservationStreamSpec) -> np.ndarray:
    """Decode std_msgs/Int32 to int32 scalar."""
    _ = spec  # Unused
    return np.array([msg.data], dtype=np.int32)


@register_decoder("std_msgs/msg/Int64", dtype="int64")
def _dec_int64(msg: Any, spec: ObservationStreamSpec) -> np.ndarray:
    """Decode std_msgs/Int64 to int64 scalar."""
    _ = spec  # Unused
    return np.array([msg.data], dtype=np.int64)


@register_decoder("std_msgs/msg/String", dtype="string")
def _dec_string(msg: Any, spec: ObservationStreamSpec) -> str:
    """Decode std_msgs/String to Python string."""
    _ = spec  # Unused
    return str(msg.data)
