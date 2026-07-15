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
Codecs: the message <-> array boundary for one ROS channel.

A *decoder* turns an incoming ROS message into the numeric array (or string) a
frame carries. An *encoder* turns a flat action vector back into an outgoing
ROS message. :func:`decode_value` and :func:`encode_value` are the two entry
points. Everything else here is how a codec gets bound to a message type.

A codec binds to a message type in one of two ways:

- By type, via the global registry. ``@register_decoder`` / ``@register_encoder``
  key a function by ROS type string. The built-in ros2 codecs and installed
  ``rosetta.codecs`` plugins populate it, so a contract naming only ``type:``
  resolves with no path.
- Inline, per channel. A spec's ``decoder:`` / ``encoder:`` gives a
  ``"module:function"`` path that wins over the registry for that channel.

Custom codecs are wired per channel in the contract (see contracts/stone.yaml
for the annotated reference):

    observations:
      observation.state:
        channel: {topic: /my_sensor, type: my_msgs/msg/MyMessage,
                  decoder: "my_package.codecs:decode_my_message"}
        align: {strategy: hold, timeline: receive}
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from ..contract.operators import forward_pipeline, inverse_pipeline
from ..contract.plugins import load_entry_point_plugins

if TYPE_CHECKING:
    from ..contract.specs import ActionStreamSpec, ObservationStreamSpec


# =============================================================================
# Type Aliases
# =============================================================================

DecoderFn = Callable[[Any, "ObservationStreamSpec | ActionStreamSpec"], "np.ndarray | str"]
EncoderFn = Callable[[np.ndarray, "ActionStreamSpec", int | None], Any]


# =============================================================================
# Dtype Vocabulary
# =============================================================================

SUPPORTED_NUMERIC_DTYPES: dict[str, Any] = {
    "float32": np.float32,
    "float64": np.float64,
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


def _record_requires_select(select_registry: set[str], type_str: str, requires_select: bool) -> None:
    # discard on re-registration: an override that no longer needs select must
    # not inherit the flag from the codec it replaced.
    if requires_select:
        select_registry.add(type_str)
    else:
        select_registry.discard(type_str)


def register_decoder(type_str: str, dtype: str, *, override: bool = False, requires_select: bool = False):
    """
    Register a decoder for a ROS message type.

    Args:
    ----
    type_str: ROS message type (e.g., "sensor_msgs/msg/JointState")
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
        _record_requires_select(DECODER_REQUIRES_SELECT, type_str, requires_select)
        return fn

    return _wrap


def register_encoder(type_str: str, *, override: bool = False, requires_select: bool = False):
    """
    Register an encoder for a ROS message type.

    Args:
    ----
    type_str: ROS message type (e.g., "geometry_msgs/msg/Twist")
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
        _record_requires_select(ENCODER_REQUIRES_SELECT, type_str, requires_select)
        return fn

    return _wrap


# =============================================================================
# Entry-point codec plugin discovery
# =============================================================================

CODEC_ENTRY_POINT_GROUP = "rosetta.codecs"
"""Entry-point group third-party codec plugins register under."""


def discover_codecs() -> None:
    """
    Register the built-in ros2 codecs, then import ``rosetta.codecs`` plugins.

    Each entry point's value is a module path. Loading it runs that module's
    ``@register_decoder`` / ``@register_encoder`` decorators, so installed
    plugins populate the registries keyed by msg_type. The contract therefore
    names a ``type:`` only, so a custom codec for a brand-new type needs no path
    in the contract. To *replace* a built-in codec, the plugin registers with
    ``override=True`` (global), or a spec uses the inline ``decoder:``/
    ``encoder:`` path (per-spec, deterministic).

    Idempotent, hard-error semantics live in the shared loader (see
    :func:`rosetta.contract.plugins.load_entry_point_plugins`): a plugin that
    fails to import raises ContractValidationError, and that failure is
    latched, so every later call re-raises the same error until the process
    restarts with a fixed environment.
    """
    # Built-in ros2 codecs register exactly like a plugin would: importing the
    # module runs its @register_* decorators. Imported here, not at module
    # level, to keep the module graph acyclic (codecs -> decoders -> schema ->
    # codecs). Both modules import without ROS. rclpy/rosidl are call-time
    # dependencies of the codec bodies only. Loaded before the entry-point
    # scan so a plugin can replace a built-in with override=True. Needs no
    # latch: repeat imports are sys.modules lookups.
    from rosetta.robots.ros2 import decoders as _decoders  # noqa: F401
    from rosetta.robots.ros2 import encoders as _encoders  # noqa: F401

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


def get_decoder_dtype(msg_type: str) -> str:
    """Get the LeRobot dtype for a message type."""
    discover_codecs()
    if msg_type not in DTYPES:
        raise ValueError(f"No decoder registered for '{msg_type}'")
    return DTYPES[msg_type]


def has_decoder(msg_type: str) -> bool:
    """True if a decoder (built-in or discovered plugin) is registered for ``msg_type``."""
    discover_codecs()
    return msg_type in DECODERS


def has_encoder(msg_type: str) -> bool:
    """True if an encoder (built-in or discovered plugin) is registered for ``msg_type``."""
    discover_codecs()
    return msg_type in ENCODERS


def decoder_requires_select(msg_type: str) -> bool:
    """True if the registered decoder for ``msg_type`` needs ``select:`` to declare its fields."""
    discover_codecs()
    return msg_type in DECODER_REQUIRES_SELECT


def encoder_requires_select(msg_type: str) -> bool:
    """True if the registered encoder for ``msg_type`` needs ``select:`` to declare its fields."""
    discover_codecs()
    return msg_type in ENCODER_REQUIRES_SELECT


# =============================================================================
# Encode/Decode Functions
# =============================================================================


def _resolve_codec(inline_path: str | None, type_str: str, registry: dict[str, Callable], role: str) -> Callable:
    """Resolve the codec for a spec: inline ``module:fn`` path wins over the registry."""
    if inline_path:
        return load_codec(inline_path)
    discover_codecs()
    fn = registry.get(type_str)
    if fn is None:
        raise ValueError(f"No {role} registered for message type: {type_str}")
    return fn


def width_mismatch_message(what: str, key: str, topic: str, got: int, dim: int, names: Sequence[str]) -> str:
    """One wording for every leg of the select-declares-width gate (decode, encode, assemble)."""
    return f"{what} for '{key}' (topic {topic}) has {got} values for its {dim}-wide spec. " + (
        "A select-less stream is a scalar; add select: to declare its fields."
        if not names
        else "Width must match the spec's select count."
    )


def decode_value(msg, spec: "ObservationStreamSpec | ActionStreamSpec") -> "np.ndarray | str":
    """
    Decode a ROS message: codec projection, then the forward operator pipeline.

    The codec (custom or registry) reduces the message to a numeric array via
    field selection. ``spec.operators`` is then applied front-to-back (the build /
    decode direction): e.g. ``rad2deg`` for joints, or ``resize`` for an
    image.

    Args:
    ----
        msg: ROS message instance
        spec: Stream spec with msg_type, optional decoder path, and operators

    Returns:
    -------
        Decoded value (numpy array or string)

    Raises:
    ------
        ValueError: If no decoder is found for the message type, if the decoder
            returned None (a decoder-contract violation: a decoded value must
            exist, and pushing None poisons the stream buffer into a "missing"
            state while messages keep arriving), if the stream declares
            operators but the decoder returned a non-array value (operators
            transform numeric arrays only, and spec resolution rejects string
            streams with ``apply``, so this backstop guards custom decoders
            whose declared dtype is wrong), or if a numeric result's width does
            not match ``spec.dim`` (select declares the width, and select-less
            numeric streams are scalars).

    """
    fn = _resolve_codec(spec.source.channel.decoder, spec.source.channel.type, DECODERS, "decoder")
    val = fn(msg, spec)

    if val is None:
        raise ValueError(
            f"decoder for stream '{spec.key}' ({spec.source.channel.type}) returned None; "
            f"a decoder must return an array or string"
        )

    # Run the forward op pipeline (rad2deg, resize, ...) on numeric values.
    operators = spec.operators
    if operators:
        if not isinstance(val, np.ndarray):
            raise ValueError(
                f"stream '{spec.key}' declares apply operators but its decoder returned "
                f"{type(val).__name__}, not a numpy array; operators transform numeric arrays only"
            )
        val = forward_pipeline(val, operators)

    # Width gate for numeric vectors (images and strings excepted): select
    # declares the stream's width. FrameLayout re-checks on the bridge path,
    # but this is the only guard for direct decode->encode callers (the HIL
    # teleop passthrough) and for custom decoders. It fails at ingest, next to
    # the message that caused it, instead of at sample time.
    if isinstance(val, np.ndarray) and not spec.is_image and val.size != spec.dim:
        raise ValueError(
            width_mismatch_message("Decoded value", spec.key, spec.source.channel.topic, val.size, spec.dim, spec.names)
        )

    return val


class NonFiniteActionError(ValueError):
    """A policy command contained NaN/Inf and must never be encoded onto a channel.

    Subclasses ValueError to keep the runtime-backstop convention (see
    rosetta.contract.errors) while letting the serve path catch exactly this
    condition (drop the frame, let the watchdog's ``channel.safety`` take
    over) without masking other encoder errors.
    """


def encode_value(
    action_vec: "np.ndarray | Sequence[float]",
    spec: "ActionStreamSpec",
    stamp_ns: int | None = None,
) -> Any:
    """
    Encode a flat action vector into a ROS message.

    Payload-first, like :func:`decode_value` and the registered
    :data:`EncoderFn` callables this dispatches to.

    Runs ``spec.operators`` in the serve direction first: ``inverse_pipeline``
    walks the operators back-to-front via their inverses (so ``rad2deg`` becomes
    deg->rad). The codec (custom or registry) then scatters the resulting values
    into the ROS message fields.

    This is the single choke point before the wire, so it owns the finiteness
    invariant: a NaN/Inf command (e.g. from a diverged policy) is refused here,
    after the pipeline, so exactly the values headed for message fields are
    checked. Note ``clamp`` does not scrub NaN. ``np.clip`` propagates it, so
    this gate is the guarantee.

    Args:
    ----
        action_vec: Flat array of action values
        spec: Action stream spec with msg_type, names, operators, and optional encoder path
        stamp_ns: Optional timestamp in nanoseconds for message header

    Returns:
    -------
        ROS message instance

    Raises:
    ------
        NonFiniteActionError: If the vector contains NaN/Inf after the
            inverse pipeline.
        ValueError: If the vector width does not match ``spec.dim``, or if
            no encoder is found for the message type

    """
    # Run the inverse op pipeline (deg2rad, ...): dataset space -> ROS space.
    action_vec = np.asarray(action_vec, dtype=np.float64)
    operators = spec.operators
    if operators:
        action_vec = inverse_pipeline(action_vec, operators)

    # Width gate: select declares the vector width, and every encoder scatters
    # by name position, so a mismatched vector is a structural error. Checked
    # before the finiteness gate (whose per-name labels only make sense at the
    # right width) and before dispatch, so custom encoders are covered too.
    # FrameLayout.split guarantees this on the bridge path. The HIL teleop
    # passthrough and direct callers have no other guard.
    if action_vec.size != spec.dim:
        raise ValueError(
            width_mismatch_message(
                "Action vector", spec.key, spec.source.channel.topic, action_vec.size, spec.dim, spec.names
            )
        )

    if not np.all(np.isfinite(action_vec)):
        bad = np.flatnonzero(~np.isfinite(np.atleast_1d(action_vec)))
        labels = [spec.names[i] if i < len(spec.names) else str(i) for i in bad]
        raise NonFiniteActionError(
            f"Non-finite action value(s) for '{spec.key}' (topic "
            f"{spec.source.channel.topic}) at {labels}: refusing to encode"
        )

    fn = _resolve_codec(spec.source.channel.encoder, spec.source.channel.type, ENCODERS, "encoder")
    return fn(action_vec, spec, stamp_ns)
