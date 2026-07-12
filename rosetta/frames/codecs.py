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
    from rosetta.frames.codecs import register_decoder, register_encoder

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
import importlib.metadata as _ilm
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from ..contract.operators import forward_pipeline, inverse_pipeline

if TYPE_CHECKING:
    from ..contract.specs import ActionStreamSpec, ObservationStreamSpec


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
ENCODER_ROUNDTRIP: dict[str, bool] = {}  # msg_type -> decode(encode(v)) == v holds

# Cache for dynamically loaded custom converters
_CONVERTER_CACHE: dict[str, Callable] = {}


# =============================================================================
# Registration Decorators
# =============================================================================


def register_decoder(type_str: str, dtype: str, *, override: bool = False):
    """
    Register a decoder for a ROS message type.

    Args:
    ----
    type_str: ROS message type (e.g., "sensor_msgs/msg/JointState")
    dtype: LeRobot dtype (video, float64, float32, int32, int64, bool, string)
    override: Replace an existing decoder for ``type_str``. Without it, a second
        registration for an already-registered type is an error -- so a plugin
        that means to replace a built-in must say so, and two plugins fighting
        over a type fail loudly instead of silently depending on import order.

    Example:
    -------
    @register_decoder("sensor_msgs/msg/JointState", dtype="float64")
    def decode_joint_state(msg, spec):
        return np.array(msg.position, dtype=np.float64)

    """

    def _wrap(fn: DecoderFn):
        if type_str in DECODERS and not override:
            raise ValueError(
                f"Decoder already registered for '{type_str}'. Pass override=True to replace it intentionally."
            )
        DECODERS[type_str] = fn
        DTYPES[type_str] = dtype
        return fn

    return _wrap


def register_encoder(type_str: str, *, override: bool = False, roundtrip: bool = True):
    """
    Register an encoder for a ROS message type.

    Args:
    ----
    type_str: ROS message type (e.g., "geometry_msgs/msg/Twist")
    override: Replace an existing encoder for ``type_str``. Without it, a second
        registration for an already-registered type is an error (see
        :func:`register_decoder`).
    roundtrip: Whether the encode/decode pair is value-round-trippable --
        ``decode(encode(v)) == v`` for the selected fields. True is the norm
        and the default; the round-trip test suite verifies it for every pair.
        Set False only for a genuinely lossy encoder (e.g. one that quantizes
        or cannot reconstruct the selected values); the suite then checks the
        weaker stability invariant instead.

    Encoder signature: (action_vec, spec, stamp_ns=None) -> ROS message

    Example:
    -------
    @register_encoder("geometry_msgs/msg/Twist")
    def encode_twist(action_vec, spec, stamp_ns=None):
        msg = Twist()
        msg.linear.x = action_vec[0]
        return msg

    """

    def _wrap(fn: EncoderFn):
        if type_str in ENCODERS and not override:
            raise ValueError(
                f"Encoder already registered for '{type_str}'. Pass override=True to replace it intentionally."
            )
        ENCODERS[type_str] = fn
        ENCODER_ROUNDTRIP[type_str] = roundtrip
        return fn

    return _wrap


# =============================================================================
# Entry-point codec plugin discovery
# =============================================================================

CODEC_ENTRY_POINT_GROUP = "rosetta.codecs"
"""Entry-point group third-party codec plugins register under."""

_codecs_discovered = False


def discover_codecs() -> None:
    """
    Import codec-plugin packages advertised under the ``rosetta.codecs`` group.

    Each entry point's value is a module path; loading it runs that module's
    ``@register_decoder`` / ``@register_encoder`` decorators, so installed
    plugins populate the registries keyed by msg_type. The contract therefore
    names a ``type:`` only -- a custom codec for a brand-new type needs no path
    in the contract. To *replace* a built-in codec, the plugin registers with
    ``override=True`` (global), or a spec uses the inline ``decoder:``/
    ``encoder:`` path (per-spec, deterministic).

    Idempotent: the scan runs once per process. A plugin that fails to import
    is a hard error, not a silent skip.
    """
    global _codecs_discovered  # noqa: PLW0603 - module-level discovery latch
    if _codecs_discovered:
        return
    _codecs_discovered = True  # set first: a failure must not trigger a re-scan

    try:
        eps = _ilm.entry_points(group=CODEC_ENTRY_POINT_GROUP)
    except TypeError:
        # importlib.metadata < 3.10 dict API (ROS 2 Jazzy is 3.12; defensive).
        eps = _ilm.entry_points().get(CODEC_ENTRY_POINT_GROUP, [])

    for ep in eps:
        try:
            ep.load()  # imports module -> runs @register_decoder/@register_encoder
        except Exception as e:
            raise ValueError(
                f"Failed to load codec plugin '{ep.name}' ({ep.value}) from "
                f"entry-point group '{CODEC_ENTRY_POINT_GROUP}': {e}"
            ) from e


# =============================================================================
# Custom Converter Loading
# =============================================================================


def load_converter(path: str) -> Callable:
    """
    Load a custom converter from a 'module.path:function_name' string.

    Args:
    ----
        path: Converter path in format "module.path:function_name"

    Returns:
    -------
        The converter function

    Raises:
    ------
        ValueError: If path format is invalid
        ImportError: If module cannot be imported
        AttributeError: If function not found in module

    """
    if path in _CONVERTER_CACHE:
        return _CONVERTER_CACHE[path]

    if ":" not in path:
        raise ValueError(f"Invalid converter path '{path}'. Expected format: 'module.path:function_name'")

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
    discover_codecs()
    if msg_type not in DTYPES:
        raise ValueError(f"No decoder registered for '{msg_type}'")
    return DTYPES[msg_type]


# =============================================================================
# Encode/Decode Functions
# =============================================================================


def decode_value(msg, spec: "ObservationStreamSpec | ActionStreamSpec") -> Any:
    """
    Decode a ROS message: codec projection, then the forward operator pipeline.

    The codec (custom or registry) reduces the message to a numeric array via
    field selection. ``spec.operators`` is then applied front-to-back (the build /
    decode direction): e.g. ``rad2deg`` for joints, or ``resize`` for an
    image. String values (e.g. std_msgs/String) bypass the pipeline.

    Args:
    ----
        msg: ROS message instance
        spec: Stream spec with msg_type, optional decoder path, and operators

    Returns:
    -------
        Decoded value (numpy array or string)

    Raises:
    ------
        ValueError: If no decoder found for message type

    """
    # Inline path = deterministic per-spec override; it wins over the registry.
    if spec.source.channel.decoder:
        fn = load_converter(spec.source.channel.decoder)
        val = fn(msg, spec)
    else:
        # Fall back to the registry (built-ins + discovered plugin codecs).
        discover_codecs()
        fn = DECODERS.get(spec.source.channel.type)
        if not fn:
            raise ValueError(f"No decoder registered for message type: {spec.source.channel.type}")
        val = fn(msg, spec)

    # Run the forward op pipeline (rad2deg, resize, ...) on numeric values.
    operators = getattr(spec, "operators", ()) or ()
    if operators and isinstance(val, np.ndarray):
        val = forward_pipeline(val, operators)

    return val


def encode_value(
    spec: "ActionStreamSpec",
    action_vec: Sequence[float],
    stamp_ns: int | None = None,
):
    """
    Encode a flat action vector into a ROS message.

    Runs ``spec.operators`` in the serve direction first: ``inverse_pipeline``
    walks the operators back-to-front via their inverses (so ``rad2deg`` becomes deg→rad).
    The codec (custom or registry) then scatters the resulting values into the
    ROS message fields.

    Args:
    ----
        spec: Action stream spec with msg_type, names, operators, and optional encoder path
        action_vec: Flat array of action values
        stamp_ns: Optional timestamp in nanoseconds for message header

    Returns:
    -------
        ROS message instance

    Raises:
    ------
        ValueError: If no encoder found for message type

    """
    # Run the inverse op pipeline (deg2rad, ...): dataset space -> ROS space.
    action_vec = np.asarray(action_vec, dtype=np.float64)
    operators = getattr(spec, "operators", ()) or ()
    if operators:
        action_vec = inverse_pipeline(action_vec, operators)

    # Inline path = deterministic per-spec override; it wins over the registry.
    if spec.source.channel.encoder:
        fn = load_converter(spec.source.channel.encoder)
        return fn(action_vec, spec, stamp_ns)

    # Fall back to the registry (built-ins + discovered plugin codecs).
    discover_codecs()
    fn = ENCODERS.get(spec.source.channel.type)
    if not fn:
        raise ValueError(f"No encoder registered for message type: {spec.source.channel.type}")
    return fn(action_vec, spec, stamp_ns)
