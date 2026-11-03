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
The codec registry: which decoder and encoder serve each message type.

A *decoder* turns an incoming message into the numeric array (or string) a
frame carries; an *encoder* turns a flat action vector back into an outgoing
message. ``@register_decoder`` / ``@register_encoder`` key a function by
message type string. Robot interfaces register their built-in codecs here,
and installed ``rosetta.codecs`` plugins add theirs, so a contract naming only
``type:`` resolves with no path. A spec's inline ``decoder:`` / ``encoder:``
``"module:function"`` path wins over the registry for that channel
(:func:`load_codec`).

This module also owns the contract's dtype vocabulary, because a dtype is a
decoder's output declaration. Lookups here read the registry as it stands;
:func:`rosetta.contract.interfaces.discover_codecs` fills it. Running a codec
on a message is :mod:`rosetta.frames.codecs`.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Sequence

import numpy as np

from .plugins import load_entry_point_plugins

__all__ = [
    "CODEC_ENTRY_POINT_GROUP",
    "SUPPORTED_NUMERIC_DTYPES",
    "DecoderFn",
    "EncoderFn",
    "load_codec_plugins",
    "register_decoder",
    "register_encoder",
]


# =============================================================================
# Type Aliases
# =============================================================================

DecoderFn = Callable[[Any, Any], Any]
"""``(msg, spec) -> np.ndarray | str``, where ``spec`` is a
:class:`~rosetta.contract.specs.ObservationStreamSpec` or
:class:`~rosetta.contract.specs.ActionStreamSpec`. A ``device_capable`` decoder may also
return a device array."""
EncoderFn = Callable[[np.ndarray, Any, int | None], Any]
"""``(action_vec, spec, stamp_ns) -> msg``, where ``spec`` is a
:class:`~rosetta.contract.specs.ActionStreamSpec`."""


# =============================================================================
# Dtype Vocabulary
# =============================================================================

SUPPORTED_NUMERIC_DTYPES: dict[str, Any] = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "uint8": np.uint8,
    "int32": np.int32,
    "int64": np.int64,
    "bool": np.bool_,
}
"""The contract numeric-dtype vocabulary (lerobot-compatible by choice), mapped
to numpy dtypes for the frame-assembly paths. FrameLayout assembles only these.
Accepting anything else here would defer the failure to bridge configure or
writer open. It lives beside the decoder registry because a dtype is a
decoder's output declaration. Schema re-exports it for the contract layer."""

SPECIAL_DTYPES = frozenset(["video", "string"])
"""Non-numeric contract dtypes: 'video' (image streams), 'string' (task text)."""


def is_valid_dtype(dtype: str) -> bool:
    """True if ``dtype`` is in the contract vocabulary (numeric or special)."""
    return dtype in SUPPORTED_NUMERIC_DTYPES or dtype in SPECIAL_DTYPES


# =============================================================================
# Registries
# =============================================================================

DECODERS: dict[str, DecoderFn] = {}
DTYPES: dict[str, str] = {}  # msg_type -> LeRobot dtype
DECODER_REQUIRES_SELECT: set[str] = set()
DECODER_DEVICE_CAPABLE: set[str] = set()
ENCODERS: dict[str, EncoderFn] = {}
ENCODER_REQUIRES_SELECT: set[str] = set()


# =============================================================================
# Registration Decorators
# =============================================================================


def _check_registration(registry: dict, type_str: str, override: bool, kind: str) -> None:
    """Enforce the override discipline shared by both registration decorators."""
    if type_str in registry and not override:
        raise ValueError(f"{kind} already registered for '{type_str}'. Pass override=True to replace it intentionally.")
    if override and type_str not in registry:
        raise ValueError(f"override=True for '{type_str}' but no {kind.lower()} is registered; nothing to override.")


def _record_flag(flag_registry: set[str], type_str: str, flag: bool) -> None:
    # discard on re-registration: an override that no longer has the flag
    # (needs select, runs on a device) must not inherit it from the codec it replaced.
    if flag:
        flag_registry.add(type_str)
    else:
        flag_registry.discard(type_str)


def register_decoder(
    type_str: str,
    dtype: str,
    *,
    override: bool = False,
    requires_select: bool = False,
    device_capable: bool = False,
):
    """
    Register a decoder for a ROS message type.

    Args:
    ----
    type_str: Message type, as the robot interface names it (e.g., "sensor_msgs/msg/JointState")
    dtype: Contract dtype, one of SUPPORTED_NUMERIC_DTYPES or SPECIAL_DTYPES
        (float32, float64, int32, int64, bool, video, string). Validated at
        registration, so a typo fails the plugin at import instead of
        surfacing later as an obscure spec-resolution or numpy error.
    override: Replace an existing decoder for ``type_str``. Without it, a second
        registration for an already-registered type is an error, so a plugin
        that means to replace a built-in must say so, and two plugins fighting
        over a type fail loudly instead of silently depending on import order.
        ``override=True`` with nothing registered is also an error: a plugin
        that means to replace a built-in should fail loudly when that
        built-in is renamed or removed, not silently register fresh.
    requires_select: The decoder gathers values by the spec's selected field
        names, so a channel of this type without ``select:`` is a
        misconfiguration. Declaring it here lets contract load reject that
        contract outright instead of the decoder raising per-message at
        runtime (where ingest treats it like a corrupt message and the
        stream silently zero-fills).
    device_capable: The decoder handles a message whose data is in device
        memory, returning the value on that device as a device array of the
        stream's dtype (see :mod:`rosetta.arrays`). It must produce the
        same values it would on the host, and still return numpy for a
        message whose data is on the host.

    Example:
    -------
    @register_decoder("sensor_msgs/msg/JointState", dtype="float64")
    def decode_joint_state(msg, spec):
        return np.array(msg.position, dtype=np.float64)

    """

    def _wrap(fn: DecoderFn):
        if not is_valid_dtype(dtype):
            raise ValueError(
                f"Invalid dtype '{dtype}' for decoder '{type_str}'. Supported: "
                f"{', '.join(sorted(SUPPORTED_NUMERIC_DTYPES) + sorted(SPECIAL_DTYPES))}"
            )
        _check_registration(DECODERS, type_str, override, "Decoder")
        DECODERS[type_str] = fn
        DTYPES[type_str] = dtype
        _record_flag(DECODER_REQUIRES_SELECT, type_str, requires_select)
        _record_flag(DECODER_DEVICE_CAPABLE, type_str, device_capable)
        return fn

    return _wrap


def register_encoder(type_str: str, *, override: bool = False, requires_select: bool = False):
    """
    Register an encoder for a ROS message type.

    Args:
    ----
    type_str: Message type, as the robot interface names it (e.g., "geometry_msgs/msg/Twist")
    override: Replace an existing encoder for ``type_str``. Without it, a second
        registration for an already-registered type is an error (see
        :func:`register_decoder`).
    requires_select: The encoder scatters values by the spec's selected field
        names, the same load-time contract check as :func:`register_decoder`.

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
        _check_registration(ENCODERS, type_str, override, "Encoder")
        ENCODERS[type_str] = fn
        _record_flag(ENCODER_REQUIRES_SELECT, type_str, requires_select)
        return fn

    return _wrap


# =============================================================================
# Entry-point codec plugins
# =============================================================================

CODEC_ENTRY_POINT_GROUP = "rosetta.codecs"
"""Entry-point group third-party codec plugins register under."""


def load_codec_plugins() -> None:
    """Import every ``rosetta.codecs`` plugin module, once per process.

    Loading a module runs its ``@register_decoder`` / ``@register_encoder``
    decorators. A plugin that fails to import raises ContractValidationError,
    latched like every module plugin (see
    :func:`rosetta.plugins.load_entry_point_plugins`).
    """
    load_entry_point_plugins(CODEC_ENTRY_POINT_GROUP, "codec")


# =============================================================================
# Custom Codec Loading
# =============================================================================


def load_codec(path: str) -> Callable:
    """Import a ``"module.path:function_name"`` codec and return the function.

    Raises ValueError on a malformed path, ImportError if the module is
    missing, AttributeError if the function is not in it.
    """
    if ":" not in path:
        raise ValueError(f"Invalid codec path '{path}'. Expected format: 'module.path:function_name'")

    module_path, func_name = path.rsplit(":", 1)
    if not module_path or not func_name:
        raise ValueError(f"Invalid codec path '{path}'. Expected format: 'module.path:function_name'")
    # No cache: import_module is a sys.modules lookup on repeats.
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


# =============================================================================
# Lookup Functions
# =============================================================================
# These read the registry as it stands and run no discovery. Loading a
# contract, or rosetta.frames.codecs.discover_codecs(), registers the robot
# interfaces' and plugins' codecs first.


def get_decoder_dtype(msg_type: str) -> str:
    """Get the LeRobot dtype for a message type."""
    if msg_type not in DTYPES:
        raise ValueError(f"No decoder registered for '{msg_type}'")
    return DTYPES[msg_type]


def has_decoder(msg_type: str) -> bool:
    """True if a decoder is registered for ``msg_type`` (see the note above on discovery)."""
    return msg_type in DECODERS


def has_encoder(msg_type: str) -> bool:
    """True if an encoder is registered for ``msg_type`` (see the note above on discovery)."""
    return msg_type in ENCODERS


def decoder_requires_select(msg_type: str) -> bool:
    """True if the registered decoder for ``msg_type`` needs ``select:`` to declare its fields."""
    return msg_type in DECODER_REQUIRES_SELECT


def encoder_requires_select(msg_type: str) -> bool:
    """True if the registered encoder for ``msg_type`` needs ``select:`` to declare its fields."""
    return msg_type in ENCODER_REQUIRES_SELECT


def width_mismatch_message(what: str, key: str, topic: str, got: int, dim: int, names: Sequence[str]) -> str:
    """One wording for every leg of the select-declares-width gate (decode, encode, assemble)."""
    return f"{what} for '{key}' (topic {topic}) has {got} values for its {dim}-wide spec. " + (
        "A select-less stream is a scalar; add select: to declare its fields."
        if not names
        else "Width must match the spec's select count."
    )
