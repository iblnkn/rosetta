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
points.

A codec binds to a message type in one of two ways:

- By type, via the registry in :mod:`rosetta.codecs`. ``@register_decoder`` /
  ``@register_encoder`` key a function by ROS type string. The built-in ros2
  codecs and installed ``rosetta.codecs`` plugins populate it
  (:func:`discover_codecs`), so a contract naming only ``type:`` resolves with
  no path.
- Inline, per channel. A spec's ``decoder:`` / ``encoder:`` gives a
  ``"module:function"`` path that wins over the registry for that channel.

Custom codecs are wired per channel in the contract (see contracts/stone.yaml
for the annotated reference):

    observations:
      observation.state:
        channel: {topic: /my_sensor, type: my_msgs/msg/MyMessage,
                  decoder: "my_package.codecs:decode_my_message"}
        align: {strategy: hold, timeline: receive}

A decoder registered ``device_capable=True`` handles messages whose data is in
device memory, returning the value as a device array (see
:mod:`rosetta.arrays`) instead of copying it to the host. A robot
interface delivers device memory only to streams :func:`host_only_reason`
clears, so every other stream decodes as numpy, as always.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from ..arrays import is_device_array
from ..codecs import (
    DECODER_DEVICE_CAPABLE,
    DECODERS,
    ENCODERS,
    load_codec,
    load_codec_plugins,
    width_mismatch_message,
)
from ..operators import forward_pipeline, inverse_pipeline

__all__ = [
    "NonFiniteActionError",
    "decode_value",
    "discover_codecs",
    "encode_value",
    "host_only_reason",
]

if TYPE_CHECKING:
    from ..contract.specs import ActionStreamSpec, ObservationStreamSpec


# =============================================================================
# Codec discovery
# =============================================================================


def discover_codecs(robot_interface: str | None = None) -> None:
    """
    Register every robot interface's built-in codecs, then import ``rosetta.codecs`` plugins.

    Raises if ``robot_interface`` (the one a contract names) failed to register.

    Each entry point's value is a module path. Loading it runs that module's
    ``@register_decoder`` / ``@register_encoder`` decorators, so installed
    plugins populate the registries keyed by msg_type. The contract therefore
    names a ``type:`` only, so a custom codec for a brand-new type needs no path
    in the contract. To *replace* a built-in codec, the plugin registers with
    ``override=True`` (global), or a spec uses the inline ``decoder:``/
    ``encoder:`` path (per-spec, deterministic).

    Idempotent, hard-error semantics live in the shared loader (see
    :func:`rosetta.plugins.load_entry_point_plugins`): a plugin that
    fails to import raises ContractValidationError, and that failure is
    latched, so every later call re-raises the same error until the process
    restarts with a fixed environment.
    """
    # Each installed robot interface registers its built-in codecs first, so a
    # plugin can replace a built-in with override=True. Imported here, not at
    # module level: an interface's codec modules import this one.
    from rosetta.robots.registry import register_interface_codecs

    register_interface_codecs(robot_interface)
    load_codec_plugins()


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


def host_only_reason(spec: "ObservationStreamSpec | ActionStreamSpec") -> str | None:
    """
    Why ``spec`` can't take device memory from its transport, or None if it can.

    A stream can when its decoder is a registered ``device_capable`` one (an
    inline ``decoder:`` path never is) and every operator in its ``apply`` is
    ``device_capable``. A robot interface asks its transport for device
    memory only on such streams.
    """
    if spec.source.channel.decoder:
        return f"inline decoder '{spec.source.channel.decoder}' runs on the host"
    discover_codecs()
    if spec.source.channel.type not in DECODER_DEVICE_CAPABLE:
        return f"the {spec.source.channel.type} decoder runs on the host"
    host_ops = [op.name for op in spec.operators if not op.device_capable]
    if host_ops:
        return f"operator(s) {', '.join(host_ops)} run on the host"
    return None


def decode_value(msg, spec: "ObservationStreamSpec | ActionStreamSpec") -> Any:
    """
    Decode a ROS message: codec projection, then the forward operator pipeline.

    An image whose message data was in device memory stays there, as a device
    array, with its work queued on the device's legacy default stream (see
    :mod:`rosetta.arrays`).

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
        Decoded value: a numpy array or string, or a device image as a device
        array

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
        on_device = is_device_array(val)
        if not on_device and not isinstance(val, np.ndarray):
            raise ValueError(
                f"stream '{spec.key}' declares apply operators but its decoder returned "
                f"{type(val).__name__}, not an array; operators transform numeric arrays only"
            )
        if on_device:
            reason = host_only_reason(spec)
            if reason is not None:
                raise ValueError(
                    f"stream '{spec.key}' decoded to device memory, but {reason}; the robot "
                    f"interface must not request device memory for this stream (see host_only_reason)"
                )
        val = forward_pipeline(val, operators)

    # Width gate for numeric values (images and strings excepted): select or
    # shape declares the stream's size. FrameLayout re-checks on the bridge
    # path, but this is the only guard for direct decode->encode callers (a
    # teleop passthrough) and for custom decoders. It fails at ingest, next to
    # the message that caused it, instead of at sample time. Size comes from
    # shape, which every array library spells the same.
    if hasattr(val, "shape") and not spec.is_image:
        size = math.prod(val.shape)
        if size != spec.dim:
            raise ValueError(
                width_mismatch_message("Decoded value", spec.key, spec.source.channel.topic, size, spec.dim, spec.names)
            )

    return val


class NonFiniteActionError(ValueError):
    """A policy command contained NaN/Inf and must never be encoded onto a channel.

    Subclasses ValueError to keep the runtime-backstop convention (see
    rosetta.errors) while letting the serve path catch exactly this
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
    # FrameLayout.split guarantees this on the bridge path. A teleop
    # passthrough and other direct callers have no other guard.
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
