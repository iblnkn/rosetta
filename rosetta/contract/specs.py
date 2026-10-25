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

"""Resolve a parsed contract into the runtime ``StreamSpec`` objects the pipeline consumes.

Each spec holds the :class:`~rosetta.contract.schema.Source` it came from
(read declaration facts through ``spec.source``) plus fields computed here:
dtype, operators, namespace, image geometry.

The ``iter_*_specs`` functions are different views of one contract:
``iter_specs`` is everything the recorder records, ``iter_policy_specs`` is
observations and actions only, ``iter_reward_as_action_specs`` recasts
rewards as the action output for a reward classifier, and
``iter_teleop_*_specs`` recast teleop sources as diagnostic columns.

One spec per source. A multi-source entry yields its specs in source order,
which is the concatenation/splitting order in FrameLayout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..frames.codecs import (
    decoder_requires_select,
    encoder_requires_select,
    get_decoder_dtype,
    has_decoder,
    has_encoder,
)
from ..frames.naming import IMAGE_KEY_PREFIX
from .errors import ContractValidationError
from .operators import Operator, OperatorContext, build_operator
from .schema import (
    EXTENDED_SECTIONS,
    SUPPORTED_NUMERIC_DTYPES,
    TELEOP_FEEDBACK_KEY,
    TELEOP_INPUT_KEY,
    Channel,
    Contract,
    FrameEntry,
    Source,
    owning_key,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class StreamSpec:
    """Base configuration for observation and action streams.

    ``source`` is the declaration this spec was resolved from; read
    declaration facts through it (``source.channel.topic``, ``source.align``).

    ``kw_only`` lets subclasses add required fields after this class's
    defaulted ones.
    """

    key: str
    names: tuple[str, ...]
    fps: int
    source: Source
    dtype: str  # from _resolve_spec_dtype
    namespace: str | None = None
    operators: tuple[Operator, ...] = ()  # forward to decode, inverse to serve
    is_image: bool = False  # True only for observation.images.* keys

    @property
    def dim(self) -> int:
        """Vector width: len(names), or 1 for a select-less scalar.

        Layout slices, zero-fill and safety actions all use it. A select-less
        numeric stream is a scalar (enforced at frame assembly).
        """
        return max(len(self.names), 1)

    @property
    def namespaced_names(self) -> list[str]:
        """Selector names with the namespace prefix applied (feature names)."""
        if self.namespace:
            return [f"{self.namespace}.{n}" for n in self.names]
        return list(self.names)


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationStreamSpec(StreamSpec):
    """Resolved observation stream configuration for runtime use."""

    image_resize: tuple[int, int] | None  # output (h, w): the pipeline's last declared output_hw


@dataclass(frozen=True, slots=True, kw_only=True)
class ActionStreamSpec(StreamSpec):
    """Resolved action stream configuration for runtime use."""


def _derive_namespaces(topics: list[str]) -> dict[str, str]:
    """
    Map each topic to a namespace that tells it apart from the others.

    Uses the first path segment that differs across all topics; failing that,
    dotted segment prefixes of increasing depth.

        ['/arm/state', '/base/state'] -> {'/arm/state': 'arm', '/base/state': 'base'}
        ['/arm/pos', '/arm/vel'] -> {'/arm/pos': 'pos', '/arm/vel': 'vel'}
    """
    if len(topics) <= 1:
        return dict.fromkeys(topics, "")

    parts_list = [[p for p in t.split("/") if p] for t in topics]
    max_depth = max(len(p) for p in parts_list)

    for i in range(max_depth):
        segments = [p[i] if i < len(p) else "" for p in parts_list]
        if len(set(segments)) == len(topics):
            return dict(zip(topics, segments, strict=True))

    for depth in range(2, max_depth + 1):
        namespaces = [".".join(p[:depth]) for p in parts_list]
        if len(set(namespaces)) == len(namespaces):
            return dict(zip(topics, namespaces, strict=True))

    # The last depth was the full path, so the topics normalize to identical
    # segments (e.g. '/a/b' vs 'a/b').
    raise ContractValidationError(
        f"Cannot derive unique namespaces for topics {topics}: they normalize "
        f"to identical paths; rename or disambiguate the topics."
    )


def _namespaces_by_topic(sources: Iterable[Source]) -> dict[str, str]:
    """Topic -> distinguishing namespace for sources feeding one frame key.

    Sources sharing one topic are told apart by their select names, so topics
    are deduped first and a single topic gets no namespace.
    """
    unique_topics = list(dict.fromkeys(s.channel.topic for s in sources))
    if len(unique_topics) <= 1:
        return {}
    return _derive_namespaces(unique_topics)


def _sources_with_namespaces(entry: FrameEntry) -> Iterable[tuple[Source, str | None]]:
    """Yield ``(source, namespace)`` for an entry, in source order."""
    ns_map = _namespaces_by_topic(entry.sources)
    for src in entry.sources:
        yield src, (ns_map.get(src.channel.topic) or None)


def _resolve_spec_dtype(
    explicit: str | None,
    decoder: str | None,
    msg_type: str,
    *,
    is_image: bool = False,
    decode_required: bool = True,
    context: str = "",
) -> str:
    """Resolve a stream's dtype: explicit > video (images) > float64 (custom decoder) > native codec.

    With ``decode_required`` (streams decoded at ingest), an undecodable type
    is an error even when ``dtype`` is explicit. Without it (publishing
    streams), a type with only a custom encoder resolves to float64.
    """
    decodable = bool(decoder) or has_decoder(msg_type)
    where = f" in {context}" if context else ""
    if decode_required and not decodable:
        raise ContractValidationError(
            f"No decoder registered for '{msg_type}'{where}. "
            f"Add a decoder in decoders.py, install a 'rosetta.codecs' "
            f"entry-point plugin that registers one, or set a custom "
            f"'decoder:' on the channel."
        )
    if explicit:
        if is_image and explicit != "video":
            raise ContractValidationError(
                f"Explicit dtype '{explicit}'{where} conflicts with the image key: "
                f"image streams are always dtype 'video' (omit dtype, or declare 'video')."
            )
        if not is_image and explicit == "video":
            raise ContractValidationError(
                f"Explicit dtype 'video'{where} on a non-image key: 'video' is "
                f"reserved for observation.images.* streams. Use a numeric dtype, "
                f"or move the entry under an image key."
            )
        return explicit
    if is_image:
        return "video"
    if decoder:
        return "float64"
    if decodable:
        return get_decoder_dtype(msg_type)
    return "float64"  # encode-only type; reachable only when decode_required=False


def _build_operators(
    apply: tuple[tuple[str, Any], ...] | list[tuple[str, Any]],
    *,
    is_image: bool = False,
) -> tuple[Operator, ...]:
    """Resolve a parsed ``apply`` list into a tuple of runtime operator instances."""
    ctx = OperatorContext(is_image=is_image)
    return tuple(build_operator(name, args, ctx) for name, args in apply)


def _image_hw_from_operators(operators: tuple[Operator, ...]) -> tuple[int, int] | None:
    """The last ``Operator.output_hw`` in the pipeline, if any."""
    for operator in reversed(operators):
        if operator.output_hw is not None:
            return operator.output_hw
    return None


def _require_numeric_apply(dtype: str | None, operators: tuple[Operator, ...], key: str) -> None:
    """Operators transform numeric arrays; a string stream cannot carry an apply pipeline."""
    if operators and dtype == "string":
        names = ", ".join(operator.name for operator in operators)
        raise ContractValidationError(
            f"'{key}' resolves to dtype 'string' but declares apply operators [{names}]; "
            f"operators transform numeric arrays only. Remove 'apply' or fix the dtype."
        )


def _require_select_for_decoder(src: Source, context: str) -> None:
    """A registry decoder that gathers values by field names needs ``select:``.

    Checked at load because at runtime ingest treats the decode failure as a
    corrupt message and the stream silently zero-fills. Custom decoders are
    exempt.
    """
    if src.select or src.channel.decoder:
        return
    if has_decoder(src.channel.type) and decoder_requires_select(src.channel.type):
        raise ContractValidationError(
            f"'{src.channel.type}' in {context}: its decoder gathers values by "
            f"selected field names; add 'select:' to declare them."
        )


def _build_observation_spec(
    key: str,
    names: tuple[str, ...],
    src: Source,
    namespace: str | None,
    contract: Contract,
    *,
    context: str,
    is_image: bool = False,
) -> ObservationStreamSpec:
    """Build an observation-shaped spec (observations, extended sections, teleop input).

    All are decoded at ingest, so the type must be decodable. An image must
    declare its output geometry through its operators.
    """
    operators = _build_operators(src.apply, is_image=is_image)
    image_resize = _image_hw_from_operators(operators) if is_image else None
    if is_image and image_resize is None:
        raise ContractValidationError(
            f"Image observation '{key}' must declare its output geometry via "
            f"an operator with a fixed output size (apply: [resize: [h, w]])"
        )
    ch = src.channel
    resolved_dtype = _resolve_spec_dtype(ch.dtype, ch.decoder, ch.type, is_image=is_image, context=context)
    if not is_image:
        _require_select_for_decoder(src, context)
    _require_numeric_apply(resolved_dtype, operators, key)
    return ObservationStreamSpec(
        key=key,
        names=names,
        fps=contract.fps,
        source=src,
        is_image=is_image,
        image_resize=image_resize,
        dtype=resolved_dtype,
        namespace=namespace,
        operators=operators,
    )


def _build_action_spec(
    key: str,
    names: tuple[str, ...],
    src: Source,
    namespace: str | None,
    contract: Contract,
    *,
    context: str,
) -> ActionStreamSpec:
    """Build an action-shaped spec (actions, reward-as-action, teleop feedback).

    All publish, so a type with only a custom encoder is allowed (float64).
    """
    ch = src.channel
    resolved_dtype = _resolve_spec_dtype(ch.dtype, ch.decoder, ch.type, decode_required=False, context=context)
    if resolved_dtype not in SUPPORTED_NUMERIC_DTYPES:
        raise ContractValidationError(
            f"Action-shaped entry '{key}' (topic {ch.topic}) resolves to dtype "
            f"'{resolved_dtype}', but published streams are numeric vectors. "
            f"Supported: {', '.join(sorted(SUPPORTED_NUMERIC_DTYPES))}."
        )
    # Actions are decoded too (the porter records them), so a registry
    # decoder's select requirement applies here as well.
    _require_select_for_decoder(src, context)
    operators = _build_operators(src.apply)
    return ActionStreamSpec(
        key=key,
        names=names,
        fps=contract.fps,
        source=src,
        dtype=resolved_dtype,
        namespace=namespace,
        operators=operators,
    )


def _require_publishable(ch: Channel, select: tuple[str, ...] | None, context: str) -> None:
    """A publishing channel needs an encoder, plus ``select:`` when it scatters by field names.

    Custom encoders are exempt from the select requirement.
    """
    if not ch.encoder and not has_encoder(ch.type):
        raise ContractValidationError(
            f"No encoder registered for '{ch.type}' in {context}. "
            f"Add an encoder in encoders.py or provide a custom encoder."
        )
    if not select and not ch.encoder and encoder_requires_select(ch.type):
        raise ContractValidationError(
            f"'{ch.type}' in {context}: its encoder scatters values by "
            f"selected field names; add 'select:' to declare them."
        )


def iter_observation_specs(contract: Contract) -> Iterable[ObservationStreamSpec]:
    """Yield observation stream specs from a contract."""
    for entry in contract.observations:
        is_image = entry.key.startswith(IMAGE_KEY_PREFIX)

        if is_image and len(entry.sources) > 1:
            raise ContractValidationError(
                f"Cannot aggregate multiple channels under image key '{entry.key}'. Each image must have a unique key."
            )

        for src, namespace in _sources_with_namespaces(entry):
            yield _build_observation_spec(
                entry.key,
                tuple(src.select or ()),
                src,
                namespace,
                contract,
                context=f"observation '{entry.key}'",
                is_image=is_image,
            )


def iter_action_specs(contract: Contract) -> Iterable[ActionStreamSpec]:
    """Yield action stream specs from a contract."""
    for entry in contract.actions:
        for src, namespace in _sources_with_namespaces(entry):
            _require_publishable(src.channel, src.select, f"action '{entry.key}'")
            yield _build_action_spec(
                entry.key,
                tuple(src.select or ()),
                src,
                namespace,
                contract,
                context=f"action '{entry.key}'",
            )


def iter_reward_as_action_specs(contract: Contract) -> Iterable[ActionStreamSpec]:
    """
    Yield the rewards section as action specs, all under the key "action".

    Used when is_classifier=True, so a reward classifier's output publishes to
    reward topics. Raises if ``rewards`` is empty; that is only an error here,
    not at load.
    """
    if not contract.rewards:
        raise ContractValidationError(
            "is_classifier requires the contract to declare at least one 'rewards' "
            "entry (its policy output publishes there instead of 'actions'), but "
            "this contract's 'rewards' section is empty."
        )

    # Every reward source shares the key "action", so namespaces span all of them.
    pairs = [(entry, src) for entry in contract.rewards for src in entry.sources]
    ns_map = _namespaces_by_topic(src for _, src in pairs)

    seen_names: set[tuple[str | None, str]] = set()
    for entry, src in pairs:
        ch = src.channel
        context = f"reward '{entry.key}' (as action)"
        # Reward channels cannot declare a custom encoder, so this requires a registered one.
        _require_publishable(ch, src.select, context)

        # A select-less reward gets the name "data", so every source under
        # the shared "action" key has a feature name.
        ns = ns_map.get(ch.topic) or None
        names = tuple(src.select or ()) or ("data",)

        # "data" bypasses FrameLayout's shared-key select check, so two
        # select-less rewards on one topic must be caught here.
        for name in names:
            namespaced = (ns, name)
            if namespaced in seen_names:
                raise ContractValidationError(
                    f"Reward-as-action feature name '{name}' from reward '{entry.key}' "
                    f"(topic {ch.topic}) collides with another reward source; add "
                    f"'select' entries or split the rewards across distinct topics."
                )
            seen_names.add(namespaced)

        spec = _build_action_spec("action", names, src, ns, contract, context=context)
        # It publishes, so every operator needs a serve direction.
        for operator in spec.operators:
            if not operator.kind.serveable:
                raise ContractValidationError(
                    f"Reward '{entry.key}' used as an action has {operator.kind.name} "
                    f"operator '{operator.name}' in apply (no serve direction); remove it."
                )
        yield spec


def _iter_extended_specs(contract: Contract) -> Iterable[ObservationStreamSpec]:
    """Yield specs from the extended sections, decoded like observations."""
    for section in EXTENDED_SECTIONS:
        for entry in getattr(contract, section):
            for src, namespace in _sources_with_namespaces(entry):
                yield _build_observation_spec(
                    entry.key,
                    tuple(src.select or ()),
                    src,
                    namespace,
                    contract,
                    context=f"{section} '{entry.key}'",
                )


def iter_specs(contract: Contract) -> Iterable[ObservationStreamSpec | ActionStreamSpec]:
    """Yield all stream specs (observations, actions, extended, teleop): what the recorder records.

    Teleop specs are diagnostic columns. During teleop the executed action is
    already recorded through the action's own spec, because hil_manager_node
    republishes teleop input onto the action topic.
    """
    yield from iter_observation_specs(contract)
    yield from iter_action_specs(contract)
    yield from _iter_extended_specs(contract)
    yield from iter_teleop_input_specs(contract)
    yield from iter_teleop_feedback_specs(contract)


def iter_policy_specs(contract: Contract) -> Iterable[ObservationStreamSpec | ActionStreamSpec]:
    """Yield the streams that feed a policy: observations, then actions.

    Excludes the extended and teleop sections, which must not leak into a
    framework's state/action vectors. Framework writers and runners both
    derive their layouts from this, so they agree.
    """
    yield from iter_observation_specs(contract)
    yield from iter_action_specs(contract)


def _teleop_namespaces(resolved: list[tuple[Source, str]]) -> dict[int, str | None]:
    """Index -> namespace for teleop sources.

    Several teleop sources can share one synthesized key (two targets in one
    multi-source action entry); those get per-topic namespaces.
    """
    by_key: dict[str, list[int]] = {}
    for i, (_, key) in enumerate(resolved):
        by_key.setdefault(key, []).append(i)

    namespaces: dict[int, str | None] = dict.fromkeys(range(len(resolved)))
    for indices in by_key.values():
        if len(indices) <= 1:
            continue
        ns_map = _namespaces_by_topic(resolved[i][0] for i in indices)
        for i in indices:
            namespaces[i] = ns_map.get(resolved[i][0].channel.topic) or None
    return namespaces


def iter_teleop_input_specs(contract: Contract) -> Iterable[ObservationStreamSpec]:
    """Yield teleop inputs as diagnostic observation columns, keyed by the action each drives.

    ``hil_manager_node`` also uses these specs to decode teleop input before
    re-encoding it onto the target action topic.
    """
    if not contract.teleop or not contract.teleop.input:
        return

    # Yield order must match contract.teleop.input: hil_manager_node zips
    # these specs strict=True against the declaration list.
    resolved = [
        (tis.source, owning_key(contract.actions, tis.target, "teleop.input: target")) for tis in contract.teleop.input
    ]
    namespaces = _teleop_namespaces(resolved)
    for i, tis in enumerate(contract.teleop.input):
        src, key = resolved[i]
        spec = _build_observation_spec(
            f"{TELEOP_INPUT_KEY}.{key}",
            tuple(src.select or ()),
            src,
            namespaces[i],
            contract,
            context=f"teleop input (target '{tis.target}')",
        )
        # hil_manager_node re-encodes the decoded value onto the numeric action
        # topic; a non-numeric input would fail on every message.
        if spec.dtype not in SUPPORTED_NUMERIC_DTYPES:
            raise ContractValidationError(
                f"Teleop input (target '{tis.target}') resolves to dtype '{spec.dtype}', "
                f"but its decoded value is encoded onto the numeric action topic. "
                f"Supported: {', '.join(sorted(SUPPORTED_NUMERIC_DTYPES))}."
            )
        yield spec


def iter_teleop_feedback_specs(contract: Contract) -> Iterable[ActionStreamSpec]:
    """Yield teleop feedback (messages to the human device) as action specs, keyed by the observation forwarded."""
    if not contract.teleop or not contract.teleop.feedback:
        return

    # Yield order must match contract.teleop.feedback (see input counterpart).
    resolved = [
        (tfs.source, owning_key(contract.observations, tfs.origin, "teleop.feedback: origin"))
        for tfs in contract.teleop.feedback
    ]
    namespaces = _teleop_namespaces(resolved)
    for i, tfs in enumerate(contract.teleop.feedback):
        src, key = resolved[i]
        _require_publishable(src.channel, src.select, f"teleop feedback (origin '{tfs.origin}')")
        yield _build_action_spec(
            f"{TELEOP_FEEDBACK_KEY}.{key}",
            tuple(src.select or ()),
            src,
            namespaces[i],
            contract,
            context=f"teleop feedback (origin '{tfs.origin}')",
        )
