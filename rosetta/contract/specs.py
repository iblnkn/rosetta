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
2. **Compute** — what the runtime needs flat on the spec: derived facts (the
   dtype precedence rule — explicit > video > custom-decoder float64 > native
   codec dtype, consulting the codec registry; ``apply`` name/arg pairs built
   into ``Operator`` objects; per-topic ``namespace`` prefixes so shared-key
   feature names stay unique; ``image_resize`` from the pipeline's declared
   output geometry, ``Operator.output_hw``) plus two verbatim carries for
   convenience (``fps`` from the contract root, ``names`` from ``select``).
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

from ..frames.codecs import (
    decoder_requires_select,
    encoder_requires_select,
    get_decoder_dtype,
    has_decoder,
    has_encoder,
)
from ..frames.naming import IMAGE_KEY_PREFIX, camera_short_name
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
    topic_owners,
)

# =============================================================================
# Runtime Stream Specs
# =============================================================================
#
# Design note — composition, not copies: a spec holds `source: Source` (its
# declaration) plus flat fields that are either derived (dtype, namespace,
# operators, image geometry) or carried verbatim for the runtime's
# convenience (fps from the contract root, names from select). Declaration
# facts are read through the reference (spec.source.channel.topic,
# spec.source.align, spec.source.kind), so there is no per-field copy step
# that could be forgotten — the silent-default hazard the old flat design
# guarded with a parity test is structurally impossible. The derivations are
# pinned by test_spec_resolution_parity.py.


@dataclass(frozen=True, slots=True, kw_only=True)
class StreamSpec:
    """Base configuration for observation and action streams.

    ``source`` is the declaration this spec was resolved from; read
    declaration facts through it (``source.channel.topic``,
    ``source.channel.qos``, ``source.align``, ``source.kind``, ...).

    ``dtype`` has no default: it is always the output of
    ``_resolve_spec_dtype`` (the builders pass it), never an implicit guess.

    ``kw_only=True``: every construction site already uses keyword arguments
    (see ``_build_observation_spec``/``_build_action_spec``), and it removes
    a foot-gun for subclasses — without it, adding any defaulted field to this
    base class would push a subclass's non-default fields after a default
    field and raise ``TypeError`` at class-definition time.
    """

    key: str
    names: tuple[str, ...]
    fps: int
    source: Source
    dtype: str  # resolved via _resolve_spec_dtype; observations decode-required, actions float64-capable
    namespace: str | None = None
    operators: tuple[Operator, ...] = ()  # Resolved pipeline; forward to decode, inverse to serve
    is_image: bool = False  # only observation.images.* keys; the honest answer everywhere else

    @property
    def dim(self) -> int:
        """Static vector width: len(names), or 1 for a select-less scalar.

        THE dim formula — layout slices, zero-fill, and safety actions all
        derive from it. A select-less numeric stream is a scalar by contract
        (enforced at frame assembly), never a dynamic-width vector.
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

    image_resize: tuple[int, int] | None  # output (h, w); the pipeline's last declared output_hw


@dataclass(frozen=True, slots=True, kw_only=True)
class ActionStreamSpec(StreamSpec):
    """Resolved action stream configuration (dtype: native decoder dtype, or float64 for custom/encode-only)."""


# =============================================================================
# Namespace Derivation
# =============================================================================


def _derive_namespaces(topics: list[str]) -> dict[str, str]:
    """
    Derive unique namespace for each topic from path segments.

    Finds the first segment that uniquely identifies each topic.

    Keys are the topics passed in. The first path segment that tells every
    topic apart wins; when no single segment does, dotted segment prefixes are
    tried in increasing depth.

    Examples
    --------
        ['/arm/state', '/base/state'] -> {'/arm/state': 'arm', '/base/state': 'base'}
        ['/arm/pos', '/arm/vel'] -> {'/arm/pos': 'pos', '/arm/vel': 'vel'}

    """
    if len(topics) <= 1:
        return dict.fromkeys(topics, "")

    parts_list = [[p for p in t.split("/") if p] for t in topics]
    max_depth = max(len(p) for p in parts_list)

    # Find first segment where all topics differ
    for i in range(max_depth):
        segments = [p[i] if i < len(p) else "" for p in parts_list]
        if len(set(segments)) == len(topics):
            return dict(zip(topics, segments, strict=True))

    # No single segment unique - build compound namespace
    for depth in range(2, max_depth + 1):
        namespaces = [".".join(p[:depth]) for p in parts_list]
        if len(set(namespaces)) == len(namespaces):
            return dict(zip(topics, namespaces, strict=True))

    # The compound loop's last iteration was the full path, so reaching here
    # proves the topics normalize to identical segment paths (e.g. '/a/b' vs
    # 'a/b'): no unique namespaces exist.
    raise ContractValidationError(
        f"Cannot derive unique namespaces for topics {topics}: they normalize "
        f"to identical paths; rename or disambiguate the topics."
    )


def _namespaces_by_topic(sources: Iterable[Source]) -> dict[str, str]:
    """Topic -> distinguishing namespace for sources feeding one frame key.

    Namespaces only help when they distinguish: several sources on ONE topic
    (different selectors) are already told apart by their names, so a single
    unique topic derives no namespace. Deduping first also keeps a duplicate
    topic from making every uniqueness tier in ``_derive_namespaces``
    unsatisfiable.
    """
    unique_topics = list(dict.fromkeys(s.channel.topic for s in sources))
    if len(unique_topics) <= 1:
        return {}
    return _derive_namespaces(unique_topics)


def _sources_with_namespaces(entry: FrameEntry) -> Iterable[tuple[Source, str | None]]:
    """Yield ``(source, namespace)`` for an entry, in source order.

    Single-source entries need no namespace. Multi-source entries (shared-key
    concatenation / action splitting) get a per-topic distinguishing segment
    so downstream feature names stay unique.
    """
    ns_map = _namespaces_by_topic(entry.sources)
    for src in entry.sources:
        yield src, (ns_map.get(src.channel.topic) or None)


# =============================================================================
# Operator Pipeline Resolution
# =============================================================================


def _resolve_spec_dtype(
    explicit: str | None,
    decoder: str | None,
    msg_type: str,
    *,
    is_image: bool = False,
    decode_required: bool = True,
    context: str = "",
) -> str:
    """Declared stream dtype — THE precedence rule, used by every spec section:

        explicit > video (image streams) > custom-decoder float64 > native codec

    ``decode_required=True`` (observations, extended sections, teleop inputs —
    everything decoded at ingest): a channel with no custom decoder and no
    registered codec raises here, at load, instead of silently dropping every
    message at ingest. An explicit ``dtype:`` never exempts the check — an
    undecodable channel is undecodable regardless of what dtype it claims.

    ``decode_required=False`` (action-shaped sections, which publish): a type
    with only a custom encoder is legitimate; its recorded column defaults to
    float64.
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
        return "float64"  # custom decoders default to float64
    if decodable:
        return get_decoder_dtype(msg_type)
    return "float64"  # encode-only type (custom encoder); reachable only when decode_required=False


def _build_operators(
    apply: tuple[tuple[str, Any], ...] | list[tuple[str, Any]],
    *,
    is_image: bool = False,
) -> tuple[Operator, ...]:
    """Resolve a parsed ``apply`` list into a tuple of runtime operator instances."""
    ctx = OperatorContext(is_image=is_image)
    return tuple(build_operator(name, args, ctx) for name, args in apply)


def _image_hw_from_operators(operators: tuple[Operator, ...]) -> tuple[int, int] | None:
    """Final output (h, w) declared by the pipeline, if any.

    Reads the ``Operator.output_hw`` declaration -- no operator is matched by
    name, so any plugin can fulfill the image-geometry role. Pipelines run
    front-to-back, so the last declaration is the stream's final geometry.
    """
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

    Enforced at load: without it the decoder raises per-message at runtime,
    where ingest treats the failure like a corrupt message and the stream
    silently zero-fills. A custom ``decoder:`` defines its own contract and
    is exempt.
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
    is_image: bool = False,
    image_resize: tuple[int, int] | None = None,
    dtype_context: str = "",
    operators: tuple[Operator, ...] | None = None,
) -> ObservationStreamSpec:
    """Shared construction for every observation-shaped spec: channel -> dtype/operators -> spec.

    Used by observations, extended sections, and teleop input — all decoded
    at ingest, so dtype resolution always requires decodability. Callers keep
    whatever validation genuinely differs (image/depth checks on
    observations; none on the others). ``operators`` lets a caller pass an
    already-built pipeline (observations need it pre-built to check for a
    resize operator before yielding); omit it to build fresh from ``src.apply``.
    """
    ch = src.channel
    resolved_dtype = _resolve_spec_dtype(ch.dtype, ch.decoder, ch.type, is_image=is_image, context=dtype_context)
    if not is_image:
        _require_select_for_decoder(src, dtype_context or f"'{key}'")
    if operators is None:
        operators = _build_operators(src.apply, is_image=is_image)
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
    operators: tuple[Operator, ...] | None = None,
    dtype_context: str = "",
) -> ActionStreamSpec:
    """Shared construction for every action-shaped spec: channel -> dtype/operators -> spec.

    Used by actions, reward-as-action, and teleop feedback — all encode-capable
    (they publish), so a type with only a custom encoder is acceptable and its
    recorded column defaults to float64. Callers keep whatever validation
    genuinely differs (encoder-registration checks, serveable-operator
    checks). ``operators`` lets a caller pass an already-built pipeline
    (reward-as-action needs it pre-built to validate serveability before
    yielding); omit it to build fresh from ``src.apply``.
    """
    ch = src.channel
    resolved_dtype = _resolve_spec_dtype(ch.dtype, ch.decoder, ch.type, decode_required=False, context=dtype_context)
    if resolved_dtype not in SUPPORTED_NUMERIC_DTYPES:
        raise ContractValidationError(
            f"Action-shaped entry '{key}' (topic {ch.topic}) resolves to dtype "
            f"'{resolved_dtype}', but published streams are numeric vectors. "
            f"Supported: {', '.join(sorted(SUPPORTED_NUMERIC_DTYPES))}."
        )
    # Actions are decoded too (the porter records them), so a registry
    # decoder's select requirement applies here as well.
    _require_select_for_decoder(src, dtype_context or f"'{key}'")
    if operators is None:
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
    """A publishing channel needs an encoder — and ``select:`` when it scatters by field names.

    Both are load-time facts: a missing encoder or a select-less
    scatter-by-name encoder would otherwise fail on every publish at runtime.
    A custom ``encoder:`` defines its own contract and is exempt from the
    select requirement.
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


# =============================================================================
# Observation Spec Iteration
# =============================================================================


def iter_observation_specs(contract: Contract) -> Iterable[ObservationStreamSpec]:
    """
    Yield observation stream specs from a contract.

    Resolves dtypes and derives namespaces for multi-source entries
    (shared-key concatenation).
    """
    # Adapters key camera dicts/filenames by the SANITIZED short name, so two
    # image keys that sanitize identically ('cam.left' / 'cam_left') would
    # silently overwrite each other downstream — reject at load instead.
    camera_names: dict[str, str] = {}

    for entry in contract.observations:
        is_image = entry.key.startswith(IMAGE_KEY_PREFIX)

        if is_image and len(entry.sources) > 1:
            raise ContractValidationError(
                f"Cannot aggregate multiple channels under image key '{entry.key}'. Each image must have a unique key."
            )

        if is_image:
            short = camera_short_name(entry.key)
            if short in camera_names:
                raise ContractValidationError(
                    f"Image keys '{camera_names[short]}' and '{entry.key}' both "
                    f"sanitize to camera name '{short}' ([^A-Za-z0-9_] -> '_'); "
                    f"downstream camera dicts are keyed by that name, so one "
                    f"camera would silently overwrite the other. Rename one key."
                )
            camera_names[short] = entry.key

        for src, namespace in _sources_with_namespaces(entry):
            # Geometry must be checked before yielding: an image without a
            # declared output size is a load-time error, not a spec with
            # image_resize=None. (Depth images are rejected at decode time,
            # where the message's encoding field is authoritative — a name
            # heuristic here would both false-positive and false-negative.)
            operators = _build_operators(src.apply, is_image=is_image)
            resize = _image_hw_from_operators(operators)
            if is_image and resize is None:
                raise ContractValidationError(
                    f"Image observation '{entry.key}' must declare its output geometry via "
                    f"an operator with a fixed output size (apply: [resize: [h, w]])"
                )

            yield _build_observation_spec(
                entry.key,
                tuple(src.select or ()),
                src,
                namespace,
                contract,
                is_image=is_image,
                image_resize=resize,
                dtype_context=f"observation '{entry.key}'",
                operators=operators,
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
                dtype_context=f"action '{entry.key}'",
            )


def iter_reward_as_action_specs(contract: Contract) -> Iterable[ActionStreamSpec]:
    """
    Yield action stream specs derived from the contract's reward section.

    Used when is_classifier=True so that a reward classifier's policy output
    publishes to reward topics instead of action topics.

    Raises
    ------
    ContractValidationError
        If the contract's ``rewards`` section is empty. A normal (non-classifier)
        robot may legitimately have no ``rewards`` entries, so this can only be
        checked here, where ``is_classifier=True`` makes ``rewards`` the sole
        source of action output — an empty section would otherwise silently
        resolve to zero action specs instead of failing.

    """
    if not contract.rewards:
        raise ContractValidationError(
            "is_classifier requires the contract to declare at least one 'rewards' "
            "entry (its policy output publishes there instead of 'actions'), but "
            "this contract's 'rewards' section is empty."
        )

    # All reward sources publish under the single forced key "action", so
    # namespaces are derived across every source of every reward entry.
    pairs = [(entry, src) for entry in contract.rewards for src in entry.sources]
    ns_map = _namespaces_by_topic(src for _, src in pairs)

    seen_names: set[tuple[str | None, str]] = set()
    for entry, src in pairs:
        ch = src.channel

        # A reward-as-action always publishes via the built-in encoder
        # registry (custom encoders are not supported on rewards), so an
        # unregistered type must fail here, not at first publish — even
        # when a custom decoder is set (decoders only cover recording).
        if not has_encoder(ch.type):
            raise ContractValidationError(
                f"No encoder registered for '{ch.type}' in reward '{entry.key}'. "
                f"Reward-as-action publishes via the built-in encoder "
                f"registry; add an encoder in encoders.py."
            )
        if not src.select and encoder_requires_select(ch.type):
            raise ContractValidationError(
                f"'{ch.type}' in reward '{entry.key}' (as action): its encoder "
                f"scatters values by selected field names; add 'select:' to declare them."
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
        # A select-less reward synthesizes a "data" name so the forced shared
        # "action" key always has static dims (unlike iter_action_specs,
        # where a select-less single-source key may stay dynamic).
        ns = ns_map.get(ch.topic) or None
        names = tuple(src.select or ()) or ("data",)

        # The synthesized name bypasses FrameLayout's shared-key "every spec
        # needs a select" check, so duplicates must be rejected here: two
        # select-less reward entries on one topic would otherwise emit
        # identical feature names and collide silently downstream.
        for name in names:
            namespaced = (ns, name)
            if namespaced in seen_names:
                raise ContractValidationError(
                    f"Reward-as-action feature name '{name}' from reward '{entry.key}' "
                    f"(topic {ch.topic}) collides with another reward source; add "
                    f"'select' entries or split the rewards across distinct topics."
                )
            seen_names.add(namespaced)

        yield _build_action_spec(
            "action", names, src, ns, contract, operators=operators, dtype_context=f"reward '{entry.key}' (as action)"
        )


def iter_extended_specs(contract: Contract) -> Iterable[ObservationStreamSpec]:
    """Yield specs from extended categories (rewards, signals, info, complementary_data).

    Extended streams are decoded at ingest like observations, so their
    channels face the same decodability requirement — an undecodable type is
    a load-time error, not a stream that silently drops every message.
    """
    for section in EXTENDED_SECTIONS:
        for entry in getattr(contract, section):
            for src, namespace in _sources_with_namespaces(entry):
                yield _build_observation_spec(
                    entry.key,
                    tuple(src.select or ()),
                    src,
                    namespace,
                    contract,
                    dtype_context=f"{section} '{entry.key}'",
                )


def iter_specs(contract: Contract) -> Iterable[ObservationStreamSpec | ActionStreamSpec]:
    """Yield all stream specs (observations, actions, extended, teleop).

    Teleop input/feedback are record-only diagnostic columns, like the
    extended sections: the live decode(teleop) -> encode(action) step in
    hil_manager_node already makes the real action topic carry a
    human-driven command during teleop, so the action's own spec (from
    iter_action_specs) already records the executed action correctly. These
    teleop specs additionally record the leader-side signal itself (e.g. the
    leader arm's raw reading) under its own key, for diagnostics -- not to
    derive the action.
    """
    yield from iter_observation_specs(contract)
    yield from iter_action_specs(contract)
    yield from iter_extended_specs(contract)
    yield from iter_teleop_input_specs(contract)
    yield from iter_teleop_feedback_specs(contract)


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


def _owning_key(entries: Iterable[FrameEntry], topic: str) -> str:
    """Frame key of the single entry whose sources include ``topic``.

    ``topic`` is a teleop input's ``target`` or a teleop feedback's
    ``origin``. schema.py already validates both conditions at YAML load
    (the topic exists, and belongs to exactly one entry), so the raises here
    guard only directly-constructed Contracts — with the same error type, so
    a programmatic caller sees a contract problem, not an internal one.
    """
    owners = topic_owners(entries).get(topic, [])
    if not owners:
        raise ContractValidationError(f"Teleop references topic '{topic}', which no entry's sources declare.")
    if len(owners) > 1:
        raise ContractValidationError(
            f"Teleop references topic '{topic}', which belongs to multiple entries "
            f"({owners}); a teleop target/origin must belong to exactly one entry."
        )
    return owners[0]


def _teleop_namespaces(resolved: list[tuple[Source, str]]) -> dict[int, str | None]:
    """Index -> namespace for teleop sources, namespacing only shared owning keys.

    Teleop entries are single-source, so per-entry namespacing never applies;
    but several entries may legally share one synthesized key (two targets
    inside one multi-source action entry). Those groups get the same
    per-topic distinguishing namespace the shared-key sections get, so
    flattened feature names stay unique downstream.
    """
    by_key: dict[str, list[int]] = {}
    for i, (_, owning_key) in enumerate(resolved):
        by_key.setdefault(owning_key, []).append(i)

    namespaces: dict[int, str | None] = dict.fromkeys(range(len(resolved)))
    for indices in by_key.values():
        if len(indices) <= 1:
            continue
        ns_map = _namespaces_by_topic(resolved[i][0] for i in indices)
        for i in indices:
            namespaces[i] = ns_map.get(resolved[i][0].channel.topic) or None
    return namespaces


def iter_teleop_input_specs(contract: Contract) -> Iterable[ObservationStreamSpec]:
    """Yield teleop input streams as diagnostic observation columns, decoded exactly like an observation.

    This is independent of live control: ``hil_manager_node`` drives
    the actual action topic named by each entry's ``target`` via its own
    decode(teleop) -> encode(action) step (so the action topic itself already
    carries a human-driven command during teleop). These specs exist so the
    porter can additionally record the leader-side signal for diagnostics,
    keyed by the action it drives so the column name is self-describing.
    """
    if not contract.teleop or not contract.teleop.input:
        return

    # Yield order must match contract.teleop.input: hil_manager_node zips
    # these specs strict=True against the declaration list.
    resolved = [(tis.source, _owning_key(contract.actions, tis.target)) for tis in contract.teleop.input]
    namespaces = _teleop_namespaces(resolved)
    for i, tis in enumerate(contract.teleop.input):
        src, owning_key = resolved[i]
        spec = _build_observation_spec(
            f"{TELEOP_INPUT_KEY}.{owning_key}",
            tuple(src.select or ()),
            src,
            namespaces[i],
            contract,
            dtype_context=f"teleop input (target '{tis.target}')",
        )
        # The decoded teleop value is re-encoded onto its (numeric) target
        # action topic by hil_manager_node; a non-numeric input would load
        # fine and then fail on every message at teleop rate.
        if spec.dtype not in SUPPORTED_NUMERIC_DTYPES:
            raise ContractValidationError(
                f"Teleop input (target '{tis.target}') resolves to dtype '{spec.dtype}', "
                f"but its decoded value is encoded onto the numeric action topic. "
                f"Supported: {', '.join(sorted(SUPPORTED_NUMERIC_DTYPES))}."
            )
        yield spec


def iter_teleop_feedback_specs(contract: Contract) -> Iterable[ActionStreamSpec]:
    """Yield teleop feedback streams (the outgoing message to the human device), encoded like an action.

    Feedback channels cannot declare ``safety`` (rejected at load), so
    ``safety_behavior`` is always ``'none'`` by construction. Keyed by the
    observation it forwards, so the column name is self-describing.
    """
    if not contract.teleop or not contract.teleop.feedback:
        return

    # Yield order must match contract.teleop.feedback (see input counterpart).
    resolved = [(tfs.source, _owning_key(contract.observations, tfs.origin)) for tfs in contract.teleop.feedback]
    namespaces = _teleop_namespaces(resolved)
    for i, tfs in enumerate(contract.teleop.feedback):
        src, owning_key = resolved[i]
        # Feedback publishes (to the human device), so it faces the same
        # load-time encoder requirement as actions — not a first-publish error.
        _require_publishable(src.channel, src.select, f"teleop feedback (origin '{tfs.origin}')")
        yield _build_action_spec(
            f"{TELEOP_FEEDBACK_KEY}.{owning_key}",
            tuple(src.select or ()),
            src,
            namespaces[i],
            contract,
            dtype_context=f"teleop feedback (origin '{tfs.origin}')",
        )
