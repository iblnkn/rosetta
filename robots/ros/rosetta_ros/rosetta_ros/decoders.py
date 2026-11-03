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

Register a new decoder with @register_decoder to support a message type
that isn't handled here yet.

Decoders declare their output dtype at registration time. This is the
single source of truth for what LeRobot dtype the decoder produces.

Selector-driven decoders (JointState, Imu, Odometry, Twist, TwistStamped,
MultiDOFCommand, JointTrajectory, Joy) require ``select`` in the contract —
registered with ``requires_select=True``, so a select-less channel of these
types fails at contract load. Scalar std_msgs types need no select; a
MultiArray still needs one unless the stream is genuinely 1-wide, because
select declares the width and a select-less numeric stream is a scalar by
contract (see decode_value's width gate).

Image Encoding Support
----------------------
All images are normalized to HWC uint8 RGB format for LeRobot compatibility.

Supported raw encodings are the keys of IMAGE_ENCODINGS (from
sensor_msgs/image_encodings.h): rgb8/bgr8, rgba8/bgra8 (alpha dropped),
mono8/8uc1 (replicated to 3 channels). To add one, add a table entry.

A raw image whose data arrives in device memory (a CUDA-backed rosidl
Buffer) decodes there, to the same bytes, as a torch tensor (see
rosetta_ros.buffers).

Tensors
-------
tensor_msgs/ExperimentalTensor (from ros2/rosidl_buffer_backends) carries a
DLTensor: dtype, shape, element strides, byte offset and data. The stream
declares its shape (``shape:``) and dtype; the decoder checks the message's
shape against it and casts to the dtype. Data in device memory stays there.

Depth encodings (mono16, 16uc1, 32fc1, 32fc) are NOT supported and will raise
DepthEncodingNotSupported — on the raw path (message ``encoding``) and the
compressed path (``format``, including compressedDepth). LeRobot does not
currently have proper depth image handling - it forces all images through
RGB conversion which causes precision loss. See DEPTH_ENCODINGS for details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

# cv2 is REQUIRED for compressed image decoding — no fallback decoder.
# Different JPEG decoders (cv2 vs PIL) produce different pixels, so a silent
# fallback would quietly break bag/live parity across machines. The import
# stays soft so environments that never decode compressed images work.
try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

from rosetta.contract.specs import DECODED_IMAGE_CHANNELS
from rosetta.frames.arrays import take
from rosetta.frames.codecs import register_decoder
from rosetta_ros.buffers import DLPACK_DTYPES, message_bytes, message_tensor
from rosetta_ros.field_access import (
    JOINT_STATE_FIELDS,
    JOINT_TRAJECTORY_FIELDS,
    MULTIDOF_FIELDS,
    dot_get,
    gather_named_fields,
    parse_joy_selector,
)

DEPTH_ENCODINGS = frozenset({"mono16", "16uc1", "32fc1", "32fc"})
"""
Depth image encodings — rejected at decode due to LeRobot limitations.

LeRobot currently lacks proper depth image handling:
- Forces all images through PIL.convert("RGB")
- No depth-specific normalization or transforms
- Precision loss when converting uint16/float32 to uint8

Pixel encoding is a decode-time concern (read from the message's ``encoding``
field), not a contract declaration — which is why this lives here, not in
the contract schema.
"""

if TYPE_CHECKING:
    from rosetta.contract.specs import StreamSpec

# This module must stay importable without ROS: codec discovery
# (rosetta.frames.codecs.discover_codecs) imports it during ROS-less contract
# loading. Decoders are duck-typed on the message instance, so no rclpy,
# rosidl, or message-class import belongs at module level.

# =============================================================================
# Image Decoding
# =============================================================================

IMAGE_ENCODINGS: dict[str, tuple[int, bool]] = {
    # encoding -> (channels, stored channel order is BGR)
    "rgb8": (3, False),
    "bgr8": (3, True),
    "rgba8": (4, False),
    "bgra8": (4, True),
    "mono8": (1, False),
    "8uc1": (1, False),
}


_CHANNELS_TO_RGB: dict[tuple[int, bool], np.ndarray | None] = {
    # (channels, BGR) -> source channel for each of R, G, B; None keeps the
    # channels as they are.
    (3, False): None,
    (3, True): np.array([2, 1, 0]),
    (4, False): np.array([0, 1, 2]),
    (4, True): np.array([2, 1, 0]),
    (1, False): np.array([0, 0, 0]),
}


class ImageDataMissing(ValueError):
    """An Image message arrived with no pixel data.

    From a CUDA-backed publisher this means the message was taken after the
    backend recycled its memory, which a deep QoS history or slow callbacks
    allow. The stream drops it like any undecodable message.
    """


class DepthEncodingNotSupported(ValueError):
    """
    Raised when a depth image encoding is encountered.

    LeRobot does not currently have proper depth image support.
    See: https://github.com/huggingface/lerobot
    """


def decode_ros_image(msg) -> Any:
    """
    Decode a ROS Image message to an HWC uint8 RGB array.

    Host data decodes to numpy. Device data decodes in device memory, to the
    same bytes, as an array of the backend that imported it. Resizing is
    handled by the ``resize`` operator in the entry's ``apply`` pipeline, not
    here.

    Raises
    ------
        DepthEncodingNotSupported: If encoding is a depth format
        ImageDataMissing: If the message carries no pixel data
        ValueError: If encoding is not supported or missing

    """
    h, w = int(msg.height), int(msg.width)
    enc = msg.encoding
    if not enc:
        # Empty encoding is a real (lazy-publisher) case; attribute absence
        # is not — every sensor_msgs/Image has the field by construction.
        raise ValueError("Image message has no encoding.")
    enc = enc.lower()
    if enc in DEPTH_ENCODINGS:
        raise DepthEncodingNotSupported(
            f"Depth image encoding '{enc}' is not supported. "
            f"LeRobot does not currently have proper depth image handling - it forces all images "
            f"through RGB conversion which causes precision loss for depth data. "
            f"Remove this observation from your contract or wait for LeRobot depth support."
        )
    if enc not in IMAGE_ENCODINGS:
        raise ValueError(f"Unsupported image encoding: '{enc}'. Supported: {sorted(IMAGE_ENCODINGS)}")

    if len(msg.data) == 0:
        if getattr(msg.data, "backend_type", "cpu") == "cuda":
            raise ImageDataMissing(
                "Image message has no data: it was taken after the CUDA backend recycled its memory. "
                "Keep the subscription's QoS history shallow and its callbacks fast."
            )
        raise ImageDataMissing(f"Image message has no data for its {h}x{w} {enc} image")

    ch, is_bgr = IMAGE_ENCODINGS[enc]
    step = int(msg.step) or w * ch  # 0 = tightly packed rows
    pixels = message_bytes(msg.data).reshape(h, step)[:, : w * ch].reshape(h, w, ch)
    to_rgb = _CHANNELS_TO_RGB[(ch, is_bgr)]
    if to_rgb is not None:
        # One gather reorders, drops alpha or replicates mono, and detaches
        # from the message's memory.
        return take(pixels, to_rgb, axis=2)
    if isinstance(pixels, np.ndarray):
        # The one materializing copy: detaches from the read-only frombuffer
        # view (resize passes equal-dims arrays through unchanged, and the
        # serve path's torch.from_numpy needs writable memory).
        return pixels.copy()
    # Device memory is left in place: the view holds the message's read
    # handle, so the publisher can't reuse the block while a frame holds it.
    return pixels


@register_decoder("sensor_msgs/msg/Image", dtype="video", device_capable=True)
def _dec_image(msg: Any, spec: StreamSpec) -> Any:
    """Decode sensor_msgs/Image to full-resolution HWC uint8 RGB."""
    _ = spec  # Encoding comes from the message itself; nothing to select.
    return decode_ros_image(msg)


@register_decoder("sensor_msgs/msg/CompressedImage", dtype="video")
def _dec_compressed_image(msg: Any, spec: StreamSpec) -> np.ndarray:
    """
    Decode sensor_msgs/CompressedImage to full-resolution HWC uint8 RGB.

    Supports jpeg, png, and other formats via cv2 (required — see module
    header: a fallback decoder would break bag/live pixel parity). Resizing
    is handled by the ``resize`` operator in the entry's ``apply`` pipeline, not here.
    """
    _ = spec  # Selection/resize handled by the operator pipeline, not the codec.
    fmt = (getattr(msg, "format", "") or "").lower()
    # A depth stream (16UC1/32FC1, or compressed_depth_image_transport's PNGs)
    # would sail through IMREAD_COLOR as silently-downcast 8-bit garbage —
    # reject it loudly, exactly like the raw-Image path does.
    if "compresseddepth" in fmt or fmt.split(";", 1)[0].strip() in DEPTH_ENCODINGS:
        raise DepthEncodingNotSupported(
            f"CompressedImage format '{msg.format}' is a depth stream; depth images "
            f"are not supported (see DEPTH_ENCODINGS)."
        )
    if cv2 is None:
        raise ImportError(
            "CompressedImage decode requires opencv-python (cv2). There is "
            "deliberately no fallback: different JPEG decoders produce "
            "different pixels, which would silently break bag/live parity — "
            "install cv2 instead."
        )
    data = np.frombuffer(msg.data, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cv2.imdecode failed for format: {msg.format}")
    # compressed_image_transport formats read "<orig>; <codec> compressed
    # <order>", and cv2.imdecode round-trips the publisher's matrix byte
    # order — so <order> IS the decoded channel order. Bare "jpeg"/"png"/empty
    # formats mean bgr (cv_bridge's convention), hence the default swap.
    order = fmt.partition("compressed")[2].strip().split(" ")[0]
    if not order.startswith("rgb"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# =============================================================================
# Tensor Decoder
# =============================================================================


class TensorDataMissing(ValueError):
    """An ExperimentalTensor message arrived with no data.

    From a CUDA-backed publisher this means the message was taken after the
    backend recycled its memory, as with ImageDataMissing.
    """


def decode_tensor(msg: Any, shape: tuple[int, ...], dtype: str) -> Any:
    """
    Decode a tensor_msgs/ExperimentalTensor to an array of ``shape`` and ``dtype``.

    Reads the message as torch_conversions does: one-lane DLPack dtypes,
    element strides (empty means row-major contiguous), and a byte offset into
    ``data``, all bounds-checked (by :func:`message_tensor`). A host message is copied, cast to ``dtype``.
    A message in device memory stays there, cast on the device only when its
    dtype differs.
    """
    msg_shape = tuple(int(n) for n in msg.shape)
    if msg_shape != shape:
        raise ValueError(
            f"ExperimentalTensor has shape {list(msg_shape)}, but the stream declares {list(shape)}; "
            f"set the stream's 'shape:' to the tensor's shape"
        )
    lanes = int(msg.dtype_lanes)
    msg_dtype = DLPACK_DTYPES.get((int(msg.dtype_code), int(msg.dtype_bits))) if lanes == 1 else None
    if msg_dtype is None:
        raise ValueError(
            f"ExperimentalTensor dtype (code {msg.dtype_code}, {msg.dtype_bits} bits, {lanes} lanes) is not supported"
        )
    strides = tuple(int(s) for s in msg.strides)
    if not strides:
        strides = tuple(int(np.prod(shape[i + 1 :])) for i in range(len(shape)))
    if len(strides) != len(shape):
        raise ValueError(f"ExperimentalTensor has {len(strides)} strides for {len(shape)} dimensions")
    if any(s < 0 for s in strides):
        raise ValueError(f"ExperimentalTensor has negative strides {list(strides)}")
    if len(msg.data) == 0:
        raise TensorDataMissing(
            "ExperimentalTensor message has no data. From a CUDA-backed publisher, the message was taken after "
            "its memory was recycled: keep the subscription's QoS history shallow and its callbacks fast."
        )
    tensor = message_tensor(msg.data, msg_dtype, shape, strides, int(msg.byte_offset))
    if isinstance(tensor, np.ndarray):
        return tensor.astype(dtype)  # a copy, so the frame doesn't hold the message's buffer
    if msg_dtype == dtype:
        return tensor  # the view holds the message's read handle, as a device image does
    import torch

    return tensor.to(getattr(torch, dtype))


@register_decoder("tensor_msgs/msg/ExperimentalTensor", dtype="float32", device_capable=True)
def _dec_tensor(msg: Any, spec: StreamSpec) -> Any:
    """Decode tensor_msgs/ExperimentalTensor to the stream's declared shape and dtype.

    On an image key the spec's shape and dtype describe the frame after its
    operators (resized, ``'video'``), so the tensor decodes like any image:
    full-resolution HWC uint8 RGB, at its declared ``shape`` or else its own,
    and the resize runs after.
    """
    if spec.is_image:
        shape = spec.source.shape or tuple(int(n) for n in msg.shape)
        if len(shape) != 3 or shape[2] != DECODED_IMAGE_CHANNELS:
            raise ValueError(
                f"ExperimentalTensor on image key '{spec.key}' has shape {list(shape)}; "
                f"an image tensor is (height, width, {DECODED_IMAGE_CHANNELS}) RGB"
            )
        return decode_tensor(msg, shape, "uint8")
    return decode_tensor(msg, spec.shape, spec.dtype)


# =============================================================================
# JointState Decoder
# =============================================================================


@register_decoder("sensor_msgs/msg/JointState", dtype="float64", requires_select=True)
def _dec_joint_state(msg: Any, spec: StreamSpec) -> np.ndarray:
    """
    Decode sensor_msgs/JointState.

    Selector names like ["position.joint1", "velocity.joint2"] extract fields
    by joint-name lookup; a bare name means position.
    """
    if not spec.names:
        raise ValueError("JointState decoder requires select (e.g. [position.joint1, velocity.joint2])")
    return gather_named_fields(spec.names, msg.name, msg, JOINT_STATE_FIELDS, default="position", msg_type="JointState")


# =============================================================================
# Dotted-Path Decoders (Imu / Odometry / Twist / TwistStamped)
# =============================================================================


def _make_dotted_decoder(msg_name: str, example: str):
    """Build a decoder whose selector names are root-relative dotted paths.

    Root-relative for every type — a stamped wrapper (TwistStamped) is NOT
    transparent, so contracts name the wrapped field explicitly
    (``twist.linear.x``, like Odometry's ``twist.twist.angular.z``).
    """

    def decode(msg: Any, spec: StreamSpec) -> np.ndarray:
        if not spec.names:
            raise ValueError(f"{msg_name} decoder requires select (e.g. [{example}])")
        return np.asarray([float(dot_get(msg, name)) for name in spec.names], dtype=np.float64)

    decode.__doc__ = f"Decode {msg_name}: selector names are root-relative dotted paths (e.g. [{example}])."
    return decode


for _type, _name, _example in (
    ("sensor_msgs/msg/Imu", "Imu", "orientation.x, angular_velocity.z"),
    ("nav_msgs/msg/Odometry", "Odometry", "pose.pose.position.x, twist.twist.angular.z"),
    ("geometry_msgs/msg/Twist", "Twist", "linear.x, angular.z"),
    ("geometry_msgs/msg/TwistStamped", "TwistStamped", "twist.linear.x, twist.angular.z"),
):
    register_decoder(_type, dtype="float64", requires_select=True)(_make_dotted_decoder(_name, _example))


# =============================================================================
# MultiDOFCommand Decoder
# =============================================================================


@register_decoder("control_msgs/msg/MultiDOFCommand", dtype="float64", requires_select=True)
def _dec_multidof_command(msg: Any, spec: StreamSpec) -> np.ndarray:
    """
    Decode control_msgs/MultiDOFCommand.

    Selector names like ["values.joint1", "values_dot.joint1"] extract DOF
    values by name; a bare name means values.
    """
    if not spec.names:
        raise ValueError("MultiDOFCommand decoder requires select (e.g. [values.joint1, values_dot.joint1])")
    return gather_named_fields(
        spec.names, msg.dof_names, msg, MULTIDOF_FIELDS, default="values", msg_type="MultiDOFCommand"
    )


# =============================================================================
# JointTrajectory Decoder
# =============================================================================


@register_decoder("trajectory_msgs/msg/JointTrajectory", dtype="float64", requires_select=True)
def _dec_joint_trajectory(msg: Any, spec: StreamSpec) -> np.ndarray:
    """
    Decode trajectory_msgs/JointTrajectory (first point only).

    Selector names like ["position.joint1", "velocity.joint2"] extract fields
    by joint name from the first trajectory point; a bare name means position.

    Field name aliases (both singular and plural are accepted):
      position / positions, velocity / velocities,
      acceleration / accelerations, effort
    """
    if not spec.names:
        raise ValueError("JointTrajectory decoder requires select (e.g. [position.joint1, velocity.joint2])")
    if not msg.points:
        # An empty trajectory (a common "cancel" convention) carries none of
        # the selected fields — raise so ingest drops this message with
        # context instead of poisoning the stream with a wrong-width value.
        raise ValueError(f"JointTrajectory message has no points; cannot decode {list(spec.names)}")
    return gather_named_fields(
        spec.names,
        msg.joint_names,
        msg.points[0],
        JOINT_TRAJECTORY_FIELDS,
        default="position",
        msg_type="JointTrajectoryPoint",
    )


# =============================================================================
# Joy Decoder
# =============================================================================


@register_decoder("sensor_msgs/msg/Joy", dtype="float32", requires_select=True)
def _dec_joy(msg: Any, spec: StreamSpec) -> np.ndarray:
    """
    Decode sensor_msgs/Joy.

    Selector names like ["axes.0", "axes.1", "buttons.0"] extract specific
    axes/buttons by index; a bare index means axes. Buttons are cast to
    float32 (0.0 / 1.0).
    """
    if not spec.names:
        raise ValueError("Joy decoder requires select (e.g. [axes.0, buttons.0])")

    out = []
    for selector in spec.names:
        field, idx = parse_joy_selector(selector)
        values = msg.axes if field == "axes" else msg.buttons
        if idx >= len(values):
            raise ValueError(f"Joy {field} index {idx} out of range (len={len(values)})")
        out.append(float(values[idx]))

    return np.asarray(out, dtype=np.float32)


# =============================================================================
# Array Decoders
# =============================================================================


@register_decoder("std_msgs/msg/Float32MultiArray", dtype="float32")
def _dec_float32_array(msg: Any, spec: StreamSpec) -> np.ndarray:
    """Decode std_msgs/Float32MultiArray to float32 array."""
    _ = spec  # Unused - no selector needed for arrays
    return np.asarray(msg.data, dtype=np.float32)


@register_decoder("std_msgs/msg/Float64MultiArray", dtype="float64")
def _dec_float64_array(msg: Any, spec: StreamSpec) -> np.ndarray:
    """Decode std_msgs/Float64MultiArray to float64 array."""
    _ = spec  # Unused - no selector needed for arrays
    return np.asarray(msg.data, dtype=np.float64)


@register_decoder("std_msgs/msg/Int32MultiArray", dtype="int32")
def _dec_int32_array(msg: Any, spec: StreamSpec) -> np.ndarray:
    """Decode std_msgs/Int32MultiArray to int32 array."""
    _ = spec  # Unused - no selector needed for arrays
    return np.asarray(msg.data, dtype=np.int32)


# =============================================================================
# Scalar Decoders
# =============================================================================


@register_decoder("std_msgs/msg/Float32", dtype="float32")
def _dec_float32(msg: Any, spec: StreamSpec) -> np.ndarray:
    """Decode std_msgs/Float32 to float32 scalar."""
    _ = spec  # Unused
    return np.array([msg.data], dtype=np.float32)


@register_decoder("std_msgs/msg/Float64", dtype="float64")
def _dec_float64(msg: Any, spec: StreamSpec) -> np.ndarray:
    """Decode std_msgs/Float64 to float64 scalar."""
    _ = spec  # Unused
    return np.array([msg.data], dtype=np.float64)


@register_decoder("std_msgs/msg/Int32", dtype="int32")
def _dec_int32(msg: Any, spec: StreamSpec) -> np.ndarray:
    """Decode std_msgs/Int32 to int32 scalar."""
    _ = spec  # Unused
    return np.array([msg.data], dtype=np.int32)


@register_decoder("std_msgs/msg/Int64", dtype="int64")
def _dec_int64(msg: Any, spec: StreamSpec) -> np.ndarray:
    """Decode std_msgs/Int64 to int64 scalar."""
    _ = spec  # Unused
    return np.array([msg.data], dtype=np.int64)


@register_decoder("std_msgs/msg/Bool", dtype="bool")
def _dec_bool(msg: Any, spec: StreamSpec) -> np.ndarray:
    """Decode std_msgs/Bool to bool scalar."""
    _ = spec  # Unused
    return np.array([msg.data], dtype=bool)


@register_decoder("std_msgs/msg/String", dtype="string")
def _dec_string(msg: Any, spec: StreamSpec) -> str:
    """Decode std_msgs/String to Python string."""
    _ = spec  # Unused
    return str(msg.data)
