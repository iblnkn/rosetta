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
Converters: encoder/decoder registration for ROS messages.

This module provides:
- Registry for ROS message encoders/decoders
- encode_value/decode_value functions for converting messages <-> arrays
- Support for custom converters specified in contract YAML

Usage:
    from rosetta.common.converters import register_decoder, register_encoder

    @register_decoder("sensor_msgs/msg/JointState", dtype="float64")
    def decode_joint_state(msg, spec):
        return np.array(msg.position, dtype=np.float64)

    @register_encoder("geometry_msgs/msg/Twist")
    def encode_twist(action_vec, spec, stamp_ns=None):
        msg = Twist()
        msg.linear.x = action_vec[0]
        return msg

Custom converters can be specified per-stream in the contract:
    observations:
      - key: observation.state
        topic: /my_sensor
        type: my_msgs/msg/MyMessage
        decoder: my_package.converters:decode_my_message
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

if TYPE_CHECKING:
    from .contract import ActionStreamSpec, ObservationStreamSpec


# =============================================================================
# Type Aliases
# =============================================================================

DecoderFn = Callable[[Any, "ObservationStreamSpec | ActionStreamSpec"], "np.ndarray | str"]
EncoderFn = Callable[[np.ndarray, "ActionStreamSpec", int | None], Any]


# =============================================================================
# Registries
# =============================================================================

DECODERS: dict[str, DecoderFn] = {}
DTYPES: dict[str, str] = {}  # msg_type -> LeRobot dtype
ENCODERS: dict[str, EncoderFn] = {}

# Cache for dynamically loaded custom converters
_CONVERTER_CACHE: dict[str, Callable] = {}


# =============================================================================
# Registration Decorators
# =============================================================================


def register_decoder(type_str: str, dtype: str):
    """Register a decoder for a ROS message type.

    Args:
        type_str: ROS message type (e.g., "sensor_msgs/msg/JointState")
        dtype: LeRobot dtype (video, float64, float32, int32, int64, bool, string)

    Example:
        @register_decoder("sensor_msgs/msg/JointState", dtype="float64")
        def decode_joint_state(msg, spec):
            return np.array(msg.position, dtype=np.float64)
    """

    def _wrap(fn: DecoderFn):
        DECODERS[type_str] = fn
        DTYPES[type_str] = dtype
        return fn

    return _wrap


def register_encoder(type_str: str):
    """Register an encoder for a ROS message type.

    Args:
        type_str: ROS message type (e.g., "geometry_msgs/msg/Twist")

    Encoder signature: (action_vec, spec, stamp_ns=None) -> ROS message

    Example:
        @register_encoder("geometry_msgs/msg/Twist")
        def encode_twist(action_vec, spec, stamp_ns=None):
            msg = Twist()
            msg.linear.x = action_vec[0]
            return msg
    """

    def _wrap(fn: EncoderFn):
        ENCODERS[type_str] = fn
        return fn

    return _wrap


# =============================================================================
# Custom Converter Loading
# =============================================================================


def load_converter(path: str) -> Callable:
    """Load a custom converter from a 'module.path:function_name' string.

    Args:
        path: Converter path in format "module.path:function_name"

    Returns:
        The converter function

    Raises:
        ValueError: If path format is invalid
        ImportError: If module cannot be imported
        AttributeError: If function not found in module
    """
    if path in _CONVERTER_CACHE:
        return _CONVERTER_CACHE[path]

    if ":" not in path:
        raise ValueError(
            f"Invalid converter path '{path}'. Expected format: 'module.path:function_name'"
        )

    module_path, func_name = path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    fn = getattr(module, func_name)

    _CONVERTER_CACHE[path] = fn
    return fn


# =============================================================================
# Lookup Functions
# =============================================================================


def get_decoder_dtype(msg_type: str) -> str:
    """Get the LeRobot dtype for a message type."""
    if msg_type not in DTYPES:
        raise ValueError(f"No decoder registered for '{msg_type}'")
    return DTYPES[msg_type]


# =============================================================================
# Encode/Decode Functions
# =============================================================================


def decode_value(msg, spec: "ObservationStreamSpec | ActionStreamSpec") -> Any:
    """Decode a ROS message using a registered or custom decoder.

    Checks for a custom decoder on the spec first, then falls back to the
    global registry.

    When ``spec.unit_conversion == "rad2deg"`` the ROS message contains radians
    but the dataset (and therefore the policy) expects degrees, so the decoded
    numeric array is converted from radians to degrees.

    Args:
        msg: ROS message instance
        spec: Stream spec with msg_type and optional decoder path

    Returns:
        Decoded value (numpy array or string)

    Raises:
        ValueError: If no decoder found for message type
    """
    # Check for custom decoder (experimental)
    if hasattr(spec, "decoder") and spec.decoder:
        fn = load_converter(spec.decoder)
        val = fn(msg, spec)
    else:
        # Fall back to registry
        fn = DECODERS.get(spec.msg_type)
        if not fn:
            raise ValueError(f"No decoder registered for message type: {spec.msg_type}")
        val = fn(msg, spec)

    # Apply unit conversion: ROS (radians) -> dataset (degrees)
    if getattr(spec, "unit_conversion", None) == "rad2deg" and isinstance(val, np.ndarray):
        val = np.rad2deg(val)

    return val


def encode_value(
    spec: "ActionStreamSpec",
    action_vec: Sequence[float],
    stamp_ns: int | None = None,
):
    """Encode a flat action vector into a ROS message.

    Checks for a custom encoder on the spec first, then falls back to the
    global registry.

    When ``spec.unit_conversion == "rad2deg"`` the dataset stores degrees but
    ROS expects radians, so the **inverse** conversion (deg → rad) is applied
    before encoding.

    Args:
        spec: Action stream spec with msg_type, names, clamp, and optional encoder path
        action_vec: Flat array of action values
        stamp_ns: Optional timestamp in nanoseconds for message header

    Returns:
        ROS message instance

    Raises:
        ValueError: If no encoder found for message type
    """
    # Apply inverse unit conversion: dataset (degrees) -> ROS (radians)
    if getattr(spec, "unit_conversion", None) == "rad2deg":
        action_vec = np.deg2rad(action_vec)

    # Check for custom encoder (experimental)
    if hasattr(spec, "encoder") and spec.encoder:
        fn = load_converter(spec.encoder)
        return fn(action_vec, spec, stamp_ns)

    # Fall back to registry
    fn = ENCODERS.get(spec.msg_type)
    if not fn:
        raise ValueError(f"No encoder registered for message type: {spec.msg_type}")
    return fn(action_vec, spec, stamp_ns)
