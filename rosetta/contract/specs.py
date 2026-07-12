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
What the runtime *consumes*: the ``StreamSpec`` family and the ``iter_*_specs`` pass.

Spec resolution turns a loaded :class:`~rosetta.contract.schema.Contract`
(what the YAML *says*) into the runtime stream specs defined here. It does
three distinct jobs:

1. **Reference** — each spec carries its declaration: ``source`` is the
   :class:`~rosetta.contract.schema.Source` it was resolved from, and
   declaration facts are read through it (``spec.source.channel.topic``,
   ``spec.source.align``, ``spec.source.kind``). Nothing is copied, so a
   declaration field can never silently go missing from a spec.
2. **Compute** — everything derivable but written nowhere in the YAML: the
   dtype precedence rule (explicit > video > custom-decoder float64 > native
   codec dtype, consulting the codec registry), ``apply`` name/arg pairs
   built into ``Operator`` objects, per-topic ``namespace`` prefixes so
   shared-key feature names stay unique, ``names`` from ``select``, and
   ``image_resize`` from the resize operator. Computed fields live flat on
   the spec.
3. **Project** — the same document read differently per consumer:
   ``iter_reward_as_action_specs`` re-casts the rewards section as the action
   output (classifier flow, forcing ``key="action"``),
   ``iter_teleop_*_specs`` re-cast teleop entries as observation/action-shaped
   streams, ``iter_policy_specs`` vs the recorder's view select different
   subsets.

One runtime spec per source; a multi-source entry yields its specs in
source order (which is concatenation/splitting order in FrameLayout).

Layout machinery (FrameLayout, feature building, zero-fill) lives in
rosetta.frames.layout; resampling (StreamBuffer) in rosetta.frames.resample;
frame-key naming helpers in rosetta.frames.naming — import them from there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..frames.codecs import DTYPES, ENCODERS, get_decoder_dtype
from .errors import ContractValidationError
from .operators import Operator, OperatorContext, build_operator
from .schema import (
    Contract,
    FrameEntry,
    Source,
)

# =============================================================================
# Runtime Stream Specs
# =============================================================================
#
# Design note — composition, not copies: a spec holds `source: Source` (its
# declaration) plus only COMPUTED fields. Declaration facts are read through
# the reference (spec.source.channel.topic, spec.source.align,
# spec.source.kind), so there is no per-field copy step that could be
# forgotten — the silent-default hazard the old flat design guarded with a
# parity test is structurally impossible. The computed fields (key, names,
# fps, dtype, namespace, operators, image geometry) stay flat because they
# exist nowhere in the declaration; their derivations are pinned by
# test_spec_resolution_parity.py.


@dataclass(frozen=True, slots=True)
class StreamSpec:
    """Base configuration for observation and action streams.

    ``source`` is the declaration this spec was resolved from; read
    declaration facts through it (``source.channel.topic``,
    ``source.channel.qos``, ``source.align``, ``source.kind``, ...).
    """

    key: str
    names: list[str]
    fps: int
    source: Source


@dataclass(frozen=True, slots=True)
class ObservationStreamSpec(StreamSpec):
    """Resolved observation stream configuration for runtime use."""

    is_image: bool
    image_resize: tuple[int, int] | None  # output (h, w); derived from the resize operator
    dtype: str = "float32"
    namespace: str | None = None
    operators: tuple[Operator, ...] = ()  # Resolved forward operator pipeline (rad2deg, resize, ...)


@dataclass(frozen=True, slots=True)
class ActionStreamSpec(StreamSpec):
    """Resolved action stream configuration for runtime use."""

    # Resolved frame dtype (native decoder dtype, or float64 for custom decoders)
    dtype: str = "float64"
    namespace: str | None = None
    operators: tuple[Operator, ...] = ()  # Resolved operator pipeline; run in reverse (inverse) to serve


# =============================================================================
# Namespace Derivation
# =============================================================================


def _derive_namespaces(topics: list[str]) -> dict[str, str]:
    """
    Derive unique namespace for each topic from path segments.

    Finds the first segment that uniquely identifies each topic.

    Examples
    --------
        ['/arm/state', '/base/state'] -> {'...': 'arm', '...': 'base'}
        ['/arm/pos', '/arm/vel'] -> {'...': 'pos', '...': 'vel'}

    """
    if len(topics) <= 1:
        return dict.fromkeys(topics, "")

    parts_list = [[p for p in t.split("/") if p] for t in topics]
    max_depth = max(len(p) for p in parts_list)

    # Find first segment where all topics differ
    for i in range(max_depth):
        segments = [p[i] if i < len(p) else "" for p in parts_list]
        if len(set(segments)) == len(topics):
            return dict(zip(topics, segments, strict=False))

    # No single segment unique - build compound namespace
    for depth in range(2, max_depth + 1):
        namespaces = [".".join(p[:depth]) for p in parts_list]
        if len(set(namespaces)) == len(namespaces):
            return dict(zip(topics, namespaces, strict=False))

    # Fallback: full path
    return {t: ".".join([p for p in t.split("/") if p]) for t in topics}


def _sources_with_namespaces(entry: FrameEntry) -> Iterable[tuple[Source, str | None]]:
    """Yield ``(source, namespace)`` for an entry, in source order.

    Single-source entries need no namespace. Multi-source entries (shared-key
    concatenation / action splitting) get a per-topic distinguishing segment
    so downstream feature names stay unique.
    """
    # Namespaces only help when they distinguish: several sources on ONE
    # topic (different selectors) are already told apart by their names.
    unique_topics = list(dict.fromkeys(s.channel.topic for s in entry.sources))
    ns_map: dict[str, str] = {}
    if len(unique_topics) > 1:
        ns_map = _derive_namespaces(unique_topics)
    for src in entry.sources:
        ns = ns_map.get(src.channel.topic, "")
        yield src, (ns or None)


# =============================================================================
# Operator Pipeline Resolution
# =============================================================================


def _resolve_spec_dtype(
    explicit: str | None,
    decoder: str | None,
    msg_type: str,
    *,
    is_image: bool = False,
    fallback: str | None = None,
    context: str = "",
) -> str:
    """Declared stream dtype — THE precedence rule, used by every spec section:

        explicit > video (image streams) > custom-decoder float64 > native codec

    ``fallback`` is the dtype for a message type with no registered decoder.
    Streams that must be decodable (observations, teleop inputs) leave it
    None, which raises; encode-capable sections (actions, whose types may
    only have a custom encoder) pass 'float64'.
    """
    if explicit:
        return explicit
    if is_image:
        return "video"
    if decoder:
        return "float64"  # custom decoders default to float64
    if msg_type in DTYPES:
        return get_decoder_dtype(msg_type)
    if fallback is not None:
        return fallback
    where = f" in {context}" if context else ""
    raise ContractValidationError(
        f"No decoder registered for '{msg_type}'{where}. "
        f"Add a decoder in decoders.py, specify dtype explicitly, or "
        f"provide a custom decoder."
    )


def _build_operators(
    apply: tuple[tuple[str, Any], ...] | list[tuple[str, Any]] | None,
    *,
    is_image: bool = False,
) -> tuple:
    """Resolve a parsed ``apply`` list into a tuple of runtime operator instances."""
    ctx = OperatorContext(is_image=is_image)
    return tuple(build_operator(name, args, ctx) for name, args in (apply or ()))


def _resize_from_operators(operators: tuple) -> tuple[int, int] | None:
    """Output (h, w) from a ``resize`` operator in the pipeline, if present."""
    for operator in operators:
        if operator.name == "resize":
            return (operator.height, operator.width)
    return None


# =============================================================================
# Observation Spec Iteration
# =============================================================================


def iter_observation_specs(contract: Contract) -> Iterable[ObservationStreamSpec]:
    """
    Yield observation stream specs from a contract.

    Resolves dtypes and derives namespaces for multi-source entries
    (shared-key concatenation).
    """
    for entry in contract.observations:
        is_image = entry.key.startswith("observation.images.")

        if is_image and len(entry.sources) > 1:
            raise ContractValidationError(
                f"Cannot aggregate multiple channels under image key '{entry.key}'. Each image must have a unique key."
            )

        for src, namespace in _sources_with_namespaces(entry):
            ch = src.channel

            # Reject depth images
            if is_image and ("depth" in ch.topic.lower() or "depth" in entry.key.lower()):
                raise ContractValidationError(
                    f"Depth image observation '{entry.key}' (topic: {ch.topic}) is not supported. "
                    f"LeRobot does not currently have proper depth image handling."
                )

            dtype = _resolve_spec_dtype(
                ch.dtype, ch.decoder, ch.type, is_image=is_image, context=f"observation '{entry.key}'"
            )

            operators = _build_operators(src.apply, is_image=is_image)
            resize = _resize_from_operators(operators)
            if is_image and resize is None:
                raise ContractValidationError(
                    f"Image observation '{entry.key}' must specify a resize operator (apply: [resize: [h, w]])"
                )

            yield ObservationStreamSpec(
                key=entry.key,
                names=list(src.select or []),
                fps=contract.fps,
                source=src,
                is_image=is_image,
                image_resize=resize,
                dtype=dtype,
                namespace=namespace,
                operators=operators,
            )


def iter_action_specs(contract: Contract) -> Iterable[ActionStreamSpec]:
    """Yield action stream specs from a contract."""
    for entry in contract.actions:
        for src, namespace in _sources_with_namespaces(entry):
            ch = src.channel

            # Only require registered encoder if no custom encoder provided
            if not ch.encoder and ch.type not in ENCODERS:
                raise ContractValidationError(
                    f"No encoder registered for '{ch.type}' in action '{entry.key}'. "
                    f"Add an encoder in encoders.py or provide a custom encoder."
                )

            yield ActionStreamSpec(
                key=entry.key,
                names=list(src.select or []),
                fps=contract.fps,
                source=src,
                dtype=_resolve_spec_dtype(ch.dtype, ch.decoder, ch.type, fallback="float64"),
                namespace=namespace,
                operators=_build_operators(src.apply),
            )


def iter_reward_as_action_specs(contract: Contract) -> Iterable[ActionStreamSpec]:
    """
    Yield action stream specs derived from the contract's reward section.

    Used when is_classifier=True so that a reward classifier's policy output
    publishes to reward topics instead of action topics.
    """
    # All reward sources publish under the single forced key "action", so
    # namespaces are derived across every source of every reward entry.
    pairs = [(entry, src) for entry in contract.rewards for src in entry.sources]
    ns_map: dict[str, str] = {}
    if len(pairs) > 1:
        ns_map = _derive_namespaces([src.channel.topic for _, src in pairs])

    for entry, src in pairs:
        ch = src.channel

        # A reward-as-action always publishes via the built-in encoder
        # registry (custom encoders are not supported on rewards), so an
        # unregistered type must fail here, not at first publish — even
        # when a custom decoder is set (decoders only cover recording).
        if ch.type not in ENCODERS:
            raise ContractValidationError(
                f"No encoder registered for '{ch.type}' in reward '{entry.key}'. "
                f"Reward-as-action publishes via the built-in encoder "
                f"registry; add an encoder in encoders.py."
            )

        # A reward used as an action must be serveable (it publishes).
        operators = _build_operators(src.apply)
        for operator in operators:
            if not operator.kind.serveable:
                raise ContractValidationError(
                    f"Reward '{entry.key}' used as an action has {operator.kind.name} "
                    f"operator '{operator.name}' in apply (no serve direction); remove it."
                )

        # Reward channels cannot declare safety/encoder (rejected at load), so
        # reading them through source agrees with the old forced values
        # ("none"/None). Note: a declared `kind` on a reward channel is now
        # honored (the flat design silently dropped it).
        ns = ns_map.get(ch.topic, "")
        yield ActionStreamSpec(
            key="action",
            names=list(src.select or []) or ["data"],
            fps=contract.fps,
            source=src,
            dtype=_resolve_spec_dtype(ch.dtype, ch.decoder, ch.type, fallback="float64"),
            namespace=ns or None,
            operators=operators,
        )


def iter_extended_specs(contract: Contract) -> Iterable[ObservationStreamSpec]:
    """Yield specs from extended categories (rewards, signals, info, complementary_data)."""
    extended = [
        contract.rewards,
        contract.signals,
        contract.info,
        contract.complementary_data,
    ]

    for entries in extended:
        for entry in entries:
            for src, namespace in _sources_with_namespaces(entry):
                ch = src.channel
                yield ObservationStreamSpec(
                    key=entry.key,
                    names=list(src.select or []),
                    fps=contract.fps,
                    source=src,
                    is_image=False,
                    image_resize=None,
                    dtype=_resolve_spec_dtype(ch.dtype, ch.decoder, ch.type, fallback="float64"),
                    namespace=namespace,
                    operators=_build_operators(src.apply),
                )


def iter_specs(contract: Contract) -> Iterable[ObservationStreamSpec | ActionStreamSpec]:
    """Yield all stream specs (observations, actions, extended)."""
    yield from iter_observation_specs(contract)
    yield from iter_action_specs(contract)
    yield from iter_extended_specs(contract)


def iter_policy_specs(contract: Contract) -> Iterable[ObservationStreamSpec | ActionStreamSpec]:
    """Yield the streams that feed a policy: observations, then actions.

    Unlike :func:`iter_specs` (what the porter records), this excludes the
    extended sections (rewards/signals/info/complementary_data) — record-only
    and RL columns that must not leak into a framework's concatenated
    state/action vectors. Framework writers and runners derive their layouts
    from this so they agree with each other by construction.
    """
    yield from iter_observation_specs(contract)
    yield from iter_action_specs(contract)


# =============================================================================
# Teleop Spec Iteration
# =============================================================================


def iter_teleop_input_specs(contract: Contract) -> Iterable[ObservationStreamSpec]:
    """Yield teleop input stream specs."""
    if not contract.teleop or not contract.teleop.input:
        return

    entry = contract.teleop.input
    for src, namespace in _sources_with_namespaces(entry):
        ch = src.channel
        yield ObservationStreamSpec(
            key=entry.key,
            names=list(src.select or []),
            fps=contract.fps,
            source=src,
            is_image=False,
            image_resize=None,
            dtype=_resolve_spec_dtype(ch.dtype, ch.decoder, ch.type, context=f"teleop input '{entry.key}'"),
            namespace=namespace,
            operators=_build_operators(src.apply),
        )


def iter_teleop_feedback_specs(contract: Contract) -> Iterable[ActionStreamSpec]:
    """Yield teleop feedback stream specs.

    Feedback channels cannot declare ``safety`` (rejected at load), so
    ``safety_behavior`` is always ``'none'`` by construction.
    """
    if not contract.teleop or not contract.teleop.feedback:
        return

    entry = contract.teleop.feedback
    for src, namespace in _sources_with_namespaces(entry):
        ch = src.channel
        yield ActionStreamSpec(
            key=entry.key,
            names=list(src.select or []),
            fps=contract.fps,
            source=src,
            dtype=_resolve_spec_dtype(ch.dtype, ch.decoder, ch.type, fallback="float64"),
            namespace=namespace,
            operators=_build_operators(src.apply),
        )
