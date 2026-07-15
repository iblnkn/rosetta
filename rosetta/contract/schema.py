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
What the contract *says*: YAML parsing, strict validation, and loading.

The document types live in :mod:`.model` (re-exported here — this module is
the contract layer's public import surface); the runtime consumes the
resolved specs in :mod:`rosetta.contract.specs`, not these types.

A contract binds channels to frame keys. Every frame-clock entry reads::

    channel (provides) -> align (chooses a timeline; mandatory) -> select
    -> apply -> the mapping key

Loading is strict, and complete: ``load_contract`` returning means the
contract is fully valid in this environment, never "valid until some
consumer iterates it". That is enforced in three layers, all at load —

1. *Parse*: unknown keys, missing align, malformed scalars, duplicate YAML
   keys, empty-but-present sections, codec paths that don't import —
   every lie the YAML shape could tell is an error with a dotted-path
   context. Top-level ``x-*`` keys are ignored (YAML anchor holders).
2. *Interface attestation*: timelines are open names, not an enum —
   the robot interface produces a message's timelines under names
   (``receive``, ``header``, ...); align only selects one. Loading asks the
   declared interface what each channel provides (and that its type exists
   at all — bare tasks/adjunct/teleop-events channels included), when the
   interface is installed; in a non-ROS tooling env these checks defer to
   runtime enforcement (see :func:`_load_interface_capability`).
3. *Resolution*: the registry-backed document rules (decodability, encoder
   registration, image geometry — see :mod:`.specs`) run eagerly by
   draining the spec projections once.

This module stays importable without ROS.
"""

from __future__ import annotations

import importlib.util
import warnings
from collections.abc import Hashable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, TypeVar

import yaml

from ..frames.codecs import (
    SPECIAL_DTYPES,
    SUPPORTED_NUMERIC_DTYPES,
    discover_codecs,
    is_valid_dtype,
    load_codec,
)
from ..frames.naming import IMAGE_KEY_PREFIX
from . import builtin_operators  # noqa: F401 - registers built-in operators
from .errors import ContractValidationError

# Re-exported document-model names: schema.py is the public import surface for
# the contract layer (the types themselves live in .model, a dependency-free
# leaf any layer can import without cycles).
from .model import (  # noqa: F401 - re-exports
    BARE_CHANNEL_RULES,
    EXTENDED_SECTIONS,
    FRAME_SECTION_TABLE,
    FRAME_SECTIONS,
    TELEOP_EVENT_NAMES,
    TELEOP_FEEDBACK_KEY,
    TELEOP_FEEDBACK_RULES,
    TELEOP_INPUT_KEY,
    TELEOP_INPUT_RULES,
    Align,
    Channel,
    ChannelRules,
    Contract,
    FieldKind,
    FrameEntry,
    ResamplePolicy,
    SafetyBehavior,
    SectionSpec,
    Source,
    Task,
    Teleop,
    TeleopEventMap,
    TeleopFeedbackSource,
    TeleopInputSource,
    topic_owners,
)
from .operators import lookup_operator

# =============================================================================
# Constants
# =============================================================================

SUPPORTED_ROBOT_INTERFACES = frozenset({"ros2"})
"""Robot interfaces a contract may declare (``robot_interface``)."""


def _load_interface_capability(robot_interface: str) -> Any | None:
    """The declared interface's introspection surface, or None when it isn't installed.

    Returns an object exposing ``provided_timelines(msg_type) -> frozenset[str]``
    (raising ValueError for an unknown type) and ``qos_profile_from_dict(mapping)``
    (raising ValueError for an invalid qos mapping). ros2 is the only interface
    today; this function is the single seam to generalize when a second one exists.

    The deferral gate is the marker package's *availability*, not a blanket
    ImportError catch: absent rclpy = a non-ROS tooling env, interface-backed
    checks defer to runtime enforcement; present-but-broken raises loudly
    rather than silently disabling validation.
    """
    if importlib.util.find_spec("rclpy") is None:
        return None
    from ..robots.ros2 import ros2_utils, timelines

    return SimpleNamespace(
        provided_timelines=timelines.provided_timelines,
        qos_profile_from_dict=ros2_utils.qos_profile_from_dict,
    )


# =============================================================================
# YAML Loading - Validation Helpers
# =============================================================================


EnumT = TypeVar("EnumT", bound=StrEnum)


def _validate_enum(value: Any, enum_cls: type[EnumT], field_name: str, context: str) -> EnumT:
    """Validate a string against an enum and return the member (a str subclass)."""
    normalized = str(value).lower().strip()
    try:
        return enum_cls(normalized)
    except ValueError:
        valid = sorted(e.value for e in enum_cls)
        raise ContractValidationError(
            f"Invalid {field_name} '{normalized}' in {context}. Must be one of: {valid}"
        ) from None


def _check_keys(data: dict[str, Any], allowed: frozenset[str] | set[str], ctx: str) -> None:
    """The one strictness primitive: reject unknown keys with precise context."""
    unknown = sorted(k for k in data if k not in allowed)
    if unknown:
        raise ContractValidationError(f"Unknown key(s) {unknown} in {ctx}. Allowed: {sorted(allowed)}")


def _require_mapping(data: Any, ctx: str) -> dict[str, Any]:
    """Require a YAML mapping, with a precise error otherwise."""
    if not isinstance(data, dict):
        raise ContractValidationError(f"{ctx} must be a mapping, got {type(data).__name__}")
    return data


def _require_fields(data: dict, fields: list[str], context: str) -> None:
    """Validate required fields are present."""
    for field in fields:
        if field not in data:
            raise ContractValidationError(f"Missing required field '{field}' in {context}")


def _parse_strict_int(value: Any, desc: str) -> int:
    """Parse an integer strictly: bools, strings, and non-integral floats are errors.

    An integral float (``30.0``) is accepted — YAML writers produce those — but
    ``3.9`` would silently change a clock or window if truncated, so it errors.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ContractValidationError(f"{desc} must be an integer, got {value!r}")


def _validate_dtype(dtype: str | None, context: str, required: bool = False) -> str | None:
    """Validate dtype if provided."""
    if dtype is None:
        if required:
            raise ContractValidationError(f"Missing required field 'dtype' in {context}")
        return None

    dtype = str(dtype).lower()
    if not is_valid_dtype(dtype):
        raise ContractValidationError(
            f"Invalid dtype '{dtype}' in {context}. Supported: "
            f"{', '.join(sorted(SUPPORTED_NUMERIC_DTYPES) + sorted(SPECIAL_DTYPES))}"
        )
    return dtype


def _validate_kind(kind: str | None, names: Iterable[str] | None, context: str) -> FieldKind:
    """Validate a field kind. Returns the member, default CONTINUOUS.

    Errors on an unknown kind or a dim count that doesn't match. Warns when an
    untagged field looks like a quaternion.
    """
    kind = _validate_enum(kind, FieldKind, "kind", context) if kind is not None else FieldKind.CONTINUOUS

    names = list(names) if names else []
    n = len(names)
    expected = kind.dims
    if expected is not None and n != expected:
        raise ContractValidationError(
            f"kind '{kind}' in {context} requires {expected} dim(s) but select has {n} ({names})."
        )

    # Warn if an untagged field has an x/y/z/w run (likely a quaternion getting
    # min_max'd). Catches it inside a larger select too, e.g. an IMU whose first
    # 4 dims are orientation.{x,y,z,w}. Split that into its own quaternion spec.
    if kind is FieldKind.CONTINUOUS and names:
        leaves = [str(s).split(".")[-1].lower() for s in names]
        has_quat_run = any(set(leaves[i : i + 4]) == {"x", "y", "z", "w"} for i in range(len(leaves) - 3))
        looks_quat = has_quat_run or any("quat" in str(s).lower() for s in names)
        if looks_quat:
            warnings.warn(
                f"{context}: select {names} contains an x/y/z/w run that looks like a "
                f"quaternion but kind is 'continuous'. Split it into its own spec with "
                f"`kind: quaternion` for correct rotation handling.",
                stacklevel=2,
            )
    return kind


def _validate_codec_path(path: str | None, context: str) -> str | None:
    """
    Resolve a codec path at contract load time.

    Delegates the parse/import/lookup to :func:`rosetta.frames.codecs.load_codec`
    -- the one resolver, so the format rules cannot drift between load-time
    validation and runtime dispatch -- and warms its cache, so the first
    message never pays the import. What stays here is the YAML-side
    normalization the resolver deliberately doesn't share: an absent or
    whitespace-only path means "no codec".

    Args:
    ----
        path: Codec path ("module.path:function_name") or None
        context: Error context string

    Returns:
    -------
        Validated path or None

    Raises:
    ------
        ContractValidationError: If path format is invalid or module/function not found

    """
    if path is None:
        return None

    path = str(path).strip()
    if not path:
        return None

    try:
        load_codec(path)
    except ValueError as e:
        raise ContractValidationError(f"{e} (in {context})") from e
    except ImportError as e:
        raise ContractValidationError(f"Cannot import codec module for '{path}' in {context}: {e}") from e
    except AttributeError as e:
        raise ContractValidationError(f"Codec function not found for '{path}' in {context}: {e}") from e

    return path


def _validate_interface(contract: Contract, capability: Any) -> None:
    """Interface-attested document checks, for every channel the contract declares.

    Three facts only the robot interface can attest: a channel's ``type``
    names a real message type (checked on *all* channels, including the bare
    tasks/adjunct/teleop-events ones that have no align), an aligned source's
    ``timeline`` is one the channel provides, and a channel's ``qos`` mapping
    speaks the interface's qos vocabulary. Runs only when the interface is
    installed — :func:`_load_interface_capability` returns None in a non-ROS
    tooling env, deferring these checks to runtime enforcement (extraction
    raises on an unknown timeline; qos construction and subscription raise on
    bad qos and unknown types).
    """

    def check(channel: Channel, align: Align | None, ctx: str) -> None:
        try:
            provided = capability.provided_timelines(channel.type)
        except ValueError as e:
            raise ContractValidationError(f"{ctx}: {e}") from e
        if align is not None and align.timeline not in provided:
            raise ContractValidationError(
                f"{ctx}.align: timeline '{align.timeline}' is not provided by '{channel.type}'. "
                f"This channel provides: {sorted(provided)}"
            )
        if channel.qos is not None:
            try:
                capability.qos_profile_from_dict(channel.qos)
            except ValueError as e:
                raise ContractValidationError(f"{ctx}.qos: {e}") from e

    for section, entry in contract.frame_entries():
        for i, src in enumerate(entry.sources):
            suffix = f"[{i}]" if len(entry.sources) > 1 else ""
            check(src.channel, src.align, f"{section}.{entry.key}{suffix}")
    for task in contract.tasks:
        check(task.channel, None, f"tasks.{task.key}.channel")
    for i, channel in enumerate(contract.adjunct):
        check(channel, None, f"adjunct[{i}].channel")
    if contract.teleop is not None:
        for i, tis in enumerate(contract.teleop.input):
            check(tis.source.channel, tis.source.align, f"teleop.input[{i}]")
        if contract.teleop.events is not None:
            check(contract.teleop.events.channel, None, "teleop.events.channel")
        for i, tfs in enumerate(contract.teleop.feedback):
            check(tfs.source.channel, tfs.source.align, f"teleop.feedback[{i}]")


def _validate_by_resolution(contract: Contract) -> None:
    """Eagerly run spec resolution's and FrameLayout's validation rules.

    The registry-backed document rules (decodability, encoder registration,
    select requirements, image single-source/geometry — see specs.py) live in
    the spec projections; building a FrameLayout over the drained specs
    additionally runs the layout rules (shared-key dtype/select coherence,
    rendered feature-name uniqueness). Doing both here makes "load_contract
    returned" mean "fully valid in this environment", instead of the same
    errors firing lazily in whichever consumer iterates or assembles first.
    ``iter_reward_as_action_specs`` is deliberately excluded: its rules only
    apply when a classifier caller re-casts rewards as the action output.
    """
    # Function-local: specs and layout consume this module's types.
    from ..frames.layout import FrameLayout
    from .specs import iter_specs

    FrameLayout(list(iter_specs(contract)))


# =============================================================================
# YAML Loading - select / apply helpers
# =============================================================================


def _parse_select_paths(raw: Any, ctx: str) -> tuple[str, ...] | None:
    """Parse the list form of ``select`` (observations/actions): field paths to project."""
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ContractValidationError(f"'select' must be a list in {ctx}")
    if not raw:
        raise ContractValidationError(f"'select' in {ctx} is present but empty — omit it to take the whole message")
    seen: set[str] = set()
    for i, item in enumerate(raw):
        if not isinstance(item, str):
            raise ContractValidationError(f"select[{i}] in {ctx} must be a string field path, got {item!r}")
        if item in seen:
            raise ContractValidationError(f"select[{i}] in {ctx} duplicates '{item}'; each field may be selected once")
        seen.add(item)
    return tuple(raw)


def _parse_select_map(raw: Any, ctx: str) -> dict[str, str]:
    """Parse the dict form of ``select`` (teleop events): event name -> field path."""
    if not isinstance(raw, dict):
        raise ContractValidationError(f"'select' must be a mapping in {ctx}")
    if not raw:
        raise ContractValidationError(f"'select' in {ctx} is present but empty — an events block must map at least one")
    for k, v in raw.items():
        if not isinstance(v, str):
            raise ContractValidationError(f"select.{k} in {ctx} must be a string field path, got {v!r}")
    return {str(k): v for k, v in raw.items()}


def _parse_apply(raw: Any, ctx: str, *, require_serveable: bool = False) -> list[tuple[str, Any]]:
    """
    Parse an ``apply`` operator list into ``[(name, args), ...]``.

    Items are bare strings (``rad2deg`` -> ``('rad2deg', None)``) or single-key
    mappings (``{resize: [h, w]}`` -> ``('resize', [h, w])``). Operator names are
    validated against the registry. When ``require_serveable`` (action
    entries), a FORWARD_ONLY operator (e.g. ``resize``) raises here, at load.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ContractValidationError(f"'apply' must be a list in {ctx}")

    operators: list[tuple[str, Any]] = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            name, args = item, None
        elif isinstance(item, dict):
            if len(item) != 1:
                raise ContractValidationError(
                    f"apply[{i}] in {ctx} must be a single-key mapping, got keys {sorted(item)}"
                )
            ((name, args),) = item.items()
            name = str(name)
        else:
            raise ContractValidationError(
                f"apply[{i}] in {ctx} must be a string or single-key mapping, got {type(item).__name__}"
            )

        cls = lookup_operator(name, ctx=f"apply[{i}] of {ctx}")
        if require_serveable and not cls.kind.serveable:
            raise ContractValidationError(
                f"Operator '{name}' in apply[{i}] of {ctx} is {cls.kind.name} (no "
                "serve direction), so it cannot be used on an action"
            )
        operators.append((name, args))
    return operators


# =============================================================================
# YAML Loading - Section Parsers
# =============================================================================

_SOURCE_KEYS = frozenset({"channel", "align", "select", "apply", "kind"})

_ALIGN_KEYS = frozenset({"strategy", "timeline", "tolerance_ms"})


def _parse_channel(data: Any, ctx: str, rules: ChannelRules) -> Channel:
    """Parse one channel block under a section's rules."""
    data = _require_mapping(data, ctx)
    _check_keys(data, rules.allowed_keys, ctx)
    _require_fields(data, ["topic", "type"], ctx)
    # Strings only: a YAML scalar of another type (topic: 123) is a typo'd
    # contract, not something to coerce.
    topic = data["topic"]
    if not isinstance(topic, str) or not topic:
        raise ContractValidationError(f"'topic' in {ctx} must be a non-empty string, got {topic!r}")
    type_ = data["type"]
    if not isinstance(type_, str) or not type_:
        raise ContractValidationError(f"'type' in {ctx} must be a non-empty string, got {type_!r}")

    dtype = _validate_dtype(data.get("dtype"), ctx, required=rules.dtype_required)
    if rules.dtype_required and dtype == "video":
        raise ContractValidationError(f"Invalid dtype '{dtype}' in {ctx}: extended sections are never images.")

    safety = SafetyBehavior.NONE
    if "safety" in data:
        safety = _validate_enum(data["safety"], SafetyBehavior, "safety", ctx)

    # Copy qos: YAML anchors alias one dict object across channels, and a
    # frozen dataclass shouldn't share mutable state with its siblings.
    qos = data.get("qos")
    if qos is not None:
        qos = dict(_require_mapping(qos, f"{ctx}.qos"))
    return Channel(
        topic=topic,
        type=type_,
        qos=qos,
        dtype=dtype,
        decoder=_validate_codec_path(data.get("decoder"), f"{ctx}.decoder"),
        encoder=_validate_codec_path(data.get("encoder"), f"{ctx}.encoder"),
        safety=safety,
    )


def _parse_align(data: Any, ctx: str) -> Align:
    """Parse a mandatory ``align`` block. No defaults: strategy and timeline are required."""
    if data is None:
        raise ContractValidationError(
            f"Missing required 'align' block in {ctx} — every frame-clock entry must choose a strategy and a timeline."
        )
    data = _require_mapping(data, ctx)
    _check_keys(data, _ALIGN_KEYS, ctx)
    _require_fields(data, ["strategy", "timeline"], ctx)

    strategy = _validate_enum(data["strategy"], ResamplePolicy, "strategy", ctx)
    # Timelines are open names produced by the robot interface — selected
    # verbatim (no case folding), matching provided_timelines().
    timeline = str(data["timeline"]).strip()
    if not timeline:
        raise ContractValidationError(f"Empty timeline in {ctx}")

    tol_raw = data.get("tolerance_ms")
    if strategy is ResamplePolicy.ASOF:
        if tol_raw is None:
            raise ContractValidationError(f"'asof' alignment in {ctx} requires 'tolerance_ms'")
        tolerance_ms = _parse_strict_int(tol_raw, f"'tolerance_ms' in {ctx}")
        if tolerance_ms <= 0:
            raise ContractValidationError(f"'tolerance_ms' must be positive in {ctx}, got {tolerance_ms}")
    else:
        if tol_raw is not None:
            raise ContractValidationError(
                f"'tolerance_ms' in {ctx} is only valid with strategy 'asof' (got '{strategy}')"
            )
        tolerance_ms = 0

    return Align(strategy=strategy, timeline=timeline, tolerance_ms=tolerance_ms)


def _parse_source(data: Any, ctx: str, rules: ChannelRules) -> Source:
    """Parse one channel -> align -> select -> apply source."""
    data = _require_mapping(data, ctx)
    _check_keys(data, _SOURCE_KEYS, ctx)
    _require_fields(data, ["channel", "align"], ctx)

    channel = _parse_channel(data["channel"], f"{ctx}.channel", rules)
    align = _parse_align(data["align"], f"{ctx}.align")

    select = _parse_select_paths(data.get("select"), ctx)
    apply = _parse_apply(data.get("apply"), ctx, require_serveable=rules.serveable)
    kind = _validate_kind(data.get("kind"), select, ctx)

    return Source(
        channel=channel,
        align=align,
        select=select,
        apply=tuple(apply),
        kind=kind,
    )


def _parse_entry(key: str, value: Any, section: SectionSpec) -> FrameEntry:
    """Parse one frame-key entry: a single source, or an ordered list of sources."""
    ctx = f"{section.name}.{key}"
    if isinstance(value, list):
        if not value:
            raise ContractValidationError(f"{ctx} has an empty source list")
        sources = tuple(_parse_source(item, f"{ctx}[{i}]", section.rules) for i, item in enumerate(value))
    else:
        sources = (_parse_source(value, ctx, section.rules),)
    return FrameEntry(key=key, sources=sources)


def _parse_frame_section(data: Any, section: SectionSpec) -> tuple[FrameEntry, ...]:
    """Parse a frame-clock section: a mapping keyed by frame key."""
    if data is None:
        return ()
    if isinstance(data, list):
        raise ContractValidationError(
            f"'{section.name}' must be a mapping keyed by frame key, got a list "
            f"(the v1 list-of-entries contract format is no longer supported)"
        )
    data = _require_mapping(data, f"'{section.name}'")
    if not data:
        raise ContractValidationError(f"'{section.name}' is present but empty — omit the section instead")

    entries: list[FrameEntry] = []
    for raw_key, value in data.items():
        key = str(raw_key)
        if section.name != "observations" and key.startswith(IMAGE_KEY_PREFIX):
            raise ContractValidationError(
                f"{section.name}.{key}: keys under '{IMAGE_KEY_PREFIX}' are only "
                f"allowed in 'observations' — image streams are observations."
            )
        entries.append(_parse_entry(key, value, section))
    return tuple(entries)


def _parse_tasks(data: Any) -> tuple[Task, ...]:
    """Parse the tasks section: a mapping of task key -> {channel}."""
    if data is None:
        return ()
    if isinstance(data, list):
        raise ContractValidationError(
            "'tasks' must be a mapping keyed by task key, got a list "
            "(the v1 list-of-entries contract format is no longer supported)"
        )
    data = _require_mapping(data, "'tasks'")
    if not data:
        raise ContractValidationError("'tasks' is present but empty — omit the section instead")

    tasks: list[Task] = []
    for key, value in data.items():
        ctx = f"tasks.{key}"
        entry = _require_mapping(value, ctx)
        _check_keys(entry, {"channel"}, ctx)
        _require_fields(entry, ["channel"], ctx)
        channel = _parse_channel(entry["channel"], f"{ctx}.channel", BARE_CHANNEL_RULES)
        tasks.append(Task(key=str(key), channel=channel))
    return tuple(tasks)


def _parse_adjunct(data: Any) -> tuple[Channel, ...]:
    """Parse the adjunct section: a list of bare channels (record-only, no key, no align)."""
    if data is None:
        return ()
    if not isinstance(data, list):
        raise ContractValidationError(f"'adjunct' must be a list of channel entries, got {type(data).__name__}")
    if not data:
        raise ContractValidationError("'adjunct' is present but empty — omit the section instead")

    channels: list[Channel] = []
    for i, item in enumerate(data):
        ctx = f"adjunct[{i}]"
        entry = _require_mapping(item, ctx)
        _check_keys(entry, {"channel"}, ctx)
        _require_fields(entry, ["channel"], ctx)
        channels.append(_parse_channel(entry["channel"], f"{ctx}.channel", BARE_CHANNEL_RULES))
    return tuple(channels)


def _parse_teleop_events(data: Any) -> TeleopEventMap:
    """Parse teleop events: a bare channel plus a dict-form select. No align."""
    ctx = "teleop.events"
    data = _require_mapping(data, ctx)
    _check_keys(data, {"channel", "select"}, ctx)
    _require_fields(data, ["channel", "select"], ctx)

    channel = _parse_channel(data["channel"], f"{ctx}.channel", BARE_CHANNEL_RULES)
    select = _parse_select_map(data["select"], ctx)
    unknown = sorted(set(select) - TELEOP_EVENT_NAMES)
    if unknown:
        raise ContractValidationError(
            f"{ctx}.select: unknown event name(s) {unknown}. Valid events: {sorted(TELEOP_EVENT_NAMES)}"
        )
    return TeleopEventMap(channel=channel, select=select)


@dataclass(frozen=True, slots=True)
class _TeleopRole:
    """How one teleop role section parses.

    ``input`` and ``feedback`` entries are the same shape — a Source plus one
    reference field binding it to an action/observation topic — so both parse
    through one code path, parameterized by this record. ``wrap`` is the
    role's dataclass; both take ``(source, ref)`` positionally.
    """

    section: str  # "teleop.input" | "teleop.feedback"
    ref_field: str  # "target" | "origin"
    ref_pool: str  # which section's topics the reference must name (error wording)
    dup_rule: str  # error tail for a repeated reference
    rules: ChannelRules
    wrap: type[TeleopInputSource] | type[TeleopFeedbackSource]


def _resolve_teleop_topic(topic: str, owners: dict[str, list[str]], field: str, section: str, ctx: str) -> None:
    """Validate a teleop ``target``/``origin`` names exactly one owning entry."""
    if topic not in owners:
        raise ContractValidationError(
            f"{ctx}: {field} '{topic}' does not match any {section} channel topic. "
            f"Known {section} topics: {sorted(owners)}"
        )
    if len(owners[topic]) > 1:
        raise ContractValidationError(
            f"{ctx}: {field} '{topic}' belongs to multiple {section} entries "
            f"({owners[topic]}); a teleop {field} must belong to exactly one entry."
        )


def _parse_teleop_role_source(data: Any, ctx: str, role: _TeleopRole, owners: dict[str, list[str]]) -> Any:
    """Parse one teleop entry: a Source plus its ``target``/``origin`` reference.

    The reference is validated against every channel topic the referenced
    section declares and must belong to exactly one of its entries.
    """
    data = _require_mapping(data, ctx)
    _check_keys(data, _SOURCE_KEYS | {role.ref_field}, ctx)
    _require_fields(data, [role.ref_field, "channel", "align"], ctx)

    ref = str(data[role.ref_field]).strip()
    _resolve_teleop_topic(ref, owners, role.ref_field, role.ref_pool, ctx)

    source = _parse_source({k: v for k, v in data.items() if k != role.ref_field}, ctx, role.rules)
    return role.wrap(source, ref)


def _parse_teleop_role_list(data: Any, role: _TeleopRole, owners: dict[str, list[str]]) -> tuple[Any, ...]:
    """Parse one teleop role section: independently-referenced sources, one per referenced topic."""
    if data is None:
        return ()
    name = role.section.partition(".")[2]
    if not isinstance(data, list):
        raise ContractValidationError(
            f"'{role.section}' must be a list of teleop {name} entries, got {type(data).__name__}"
        )
    if not data:
        raise ContractValidationError(f"'{role.section}' has an empty source list")
    entries = tuple(
        _parse_teleop_role_source(item, f"{role.section}[{i}]", role, owners) for i, item in enumerate(data)
    )

    seen: set[str] = set()
    for i, item in enumerate(entries):
        ref = getattr(item, role.ref_field)
        if ref in seen:
            raise ContractValidationError(f"{role.section}[{i}]: duplicate {role.ref_field} '{ref}' — {role.dup_rule}")
        seen.add(ref)
    return entries


_TELEOP_ROLES = {
    "input": _TeleopRole(
        section="teleop.input",
        ref_field="target",
        ref_pool="action",
        dup_rule="each action topic may be driven by at most one teleop input.",
        rules=TELEOP_INPUT_RULES,
        wrap=TeleopInputSource,
    ),
    "feedback": _TeleopRole(
        section="teleop.feedback",
        ref_field="origin",
        ref_pool="observation",
        dup_rule="each observation topic may feed at most one teleop feedback.",
        rules=TELEOP_FEEDBACK_RULES,
        wrap=TeleopFeedbackSource,
    ),
}


def _parse_teleop(data: Any, actions: Iterable[FrameEntry], observations: Iterable[FrameEntry]) -> Teleop | None:
    """Parse teleop role sections: input / events / feedback.

    ``actions``/``observations`` are the already-parsed frame entries, so a
    teleop input's ``target`` (or a feedback's ``origin``) that doesn't name
    a real topic — or names one owned by several entries, or repeats a
    target/origin — is a load-time error, not a message silently going
    nowhere or a recording column silently collapsing.
    """
    if data is None:
        return None
    data = _require_mapping(data, "'teleop'")
    if not data:
        raise ContractValidationError("'teleop' is present but empty — omit the section instead")
    _check_keys(data, {"input", "events", "feedback"}, "'teleop'")
    for name in ("input", "events", "feedback"):
        if name in data and data[name] is None:
            raise ContractValidationError(f"'teleop.{name}' is present but null — omit it instead")

    events = None
    if data.get("events") is not None:
        events = _parse_teleop_events(data["events"])

    return Teleop(
        input=_parse_teleop_role_list(data.get("input"), _TELEOP_ROLES["input"], topic_owners(actions)),
        events=events,
        feedback=_parse_teleop_role_list(data.get("feedback"), _TELEOP_ROLES["feedback"], topic_owners(observations)),
    )


# =============================================================================
# Main Loader
# =============================================================================


class _StrictYamlLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys.

    PyYAML silently keeps the last value of a duplicated key, which would let a
    copy-paste edit vanish an entire frame entry with no error anywhere — the
    one lie a contract could otherwise still tell undetected. Anchors/aliases
    and ``<<:`` merge keys still work: ``flatten_mapping`` resolves merges
    before the duplicate scan, and aliases live at value position.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if isinstance(key, Hashable):
                if key in seen:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        f"found duplicate key {key!r}",
                        key_node.start_mark,
                    )
                seen.add(key)
        return super().construct_mapping(node, deep)


def _parse_header(data: dict[str, Any]) -> tuple[str, str, int]:
    """Parse and validate the contract header scalars: robot_type, robot_interface, fps."""
    if "robot_type" not in data:
        raise ContractValidationError("robot_type is required")
    raw_type = data["robot_type"]
    robot_type = raw_type.strip() if isinstance(raw_type, str) else ""
    if not robot_type:
        raise ContractValidationError(f"robot_type must be a non-empty string, got {raw_type!r}")

    robot_interface = data.get("robot_interface")
    if not robot_interface:
        raise ContractValidationError(f"robot_interface is required. Supported: {sorted(SUPPORTED_ROBOT_INTERFACES)}")
    robot_interface = str(robot_interface).lower().strip()
    if robot_interface not in SUPPORTED_ROBOT_INTERFACES:
        raise ContractValidationError(
            f"Unsupported robot_interface '{robot_interface}'. Supported: {sorted(SUPPORTED_ROBOT_INTERFACES)}"
        )

    if "fps" not in data:
        raise ContractValidationError("fps is required")
    fps = _parse_strict_int(data["fps"], "fps")
    if fps <= 0:
        raise ContractValidationError(f"fps must be positive, got {fps}")

    return robot_type, robot_interface, fps


_TOP_LEVEL_KEYS = frozenset({"robot_type", "robot_interface", "fps", "tasks", "adjunct", "teleop", *FRAME_SECTIONS})


def load_contract(path: Path | str) -> Contract:
    """
    Load and validate a contract YAML file.

    Args:
    ----
        path: Path to the contract YAML file.

    Returns:
    -------
        Validated Contract dataclass.

    Raises:
    ------
        FileNotFoundError: If the file doesn't exist.
        ContractValidationError: If the contract is invalid.

    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Contract file not found: {path}")

    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictYamlLoader) or {}
    except yaml.YAMLError as e:
        raise ContractValidationError(f"Invalid YAML in {path}: {e}") from e

    if not isinstance(data, dict):
        raise ContractValidationError(f"Contract must be a YAML mapping, got {type(data).__name__}")

    # Top-level x-* keys hold YAML anchors (e.g. shared QoS blocks) — ignored.
    data = {k: v for k, v in data.items() if not str(k).startswith("x-")}
    _check_keys(data, _TOP_LEVEL_KEYS, "contract")

    # A present-but-null section is the same mistake as present-but-empty:
    # the author wrote the key and YAML resolved its value to null.
    for name in ("tasks", "adjunct", "teleop", *FRAME_SECTIONS):
        if name in data and data[name] is None:
            raise ContractValidationError(f"'{name}' is present but null — omit the section instead")

    robot_type, robot_interface, fps = _parse_header(data)

    # Register the built-in codecs and entry-point plugins so registry-keyed
    # dtype/encoder lookups during spec building see every codec.
    discover_codecs()

    sections = {s.name: _parse_frame_section(data.get(s.name), s) for s in FRAME_SECTION_TABLE}

    # A frame key must belong to exactly one section: the same key in two
    # sections would silently merge into one recording column (FrameLayout
    # groups by key) while narrower views (e.g. iter_policy_specs) disagree
    # about its contents. Within-section duplicates are already impossible
    # (YAML mappings; _StrictYamlLoader rejects duplicate mapping keys).
    key_owners: dict[str, str] = {}
    for section_name, entries in sections.items():
        for entry in entries:
            owner = key_owners.setdefault(entry.key, section_name)
            if owner != section_name:
                raise ContractValidationError(
                    f"Frame key '{entry.key}' is declared in both '{owner}' and "
                    f"'{section_name}'; a key must belong to exactly one section."
                )

    # teleop.input/feedback validate their target/origin against the parsed
    # actions/observations entries, so a typo'd or ambiguous reference is a
    # load-time error rather than a message silently going nowhere.
    teleop = _parse_teleop(data.get("teleop"), sections["actions"], sections["observations"])

    # The synthesized teleop recording keys (specs.py derives
    # teleop.input.<owner> / teleop.feedback.<owner>) claim key names too: a
    # user-declared key of the same name would silently merge with the
    # diagnostic column (FrameLayout groups by key).
    if teleop is not None:
        action_owners = topic_owners(sections["actions"])
        obs_owners = topic_owners(sections["observations"])
        synthesized = [f"{TELEOP_INPUT_KEY}.{action_owners[tis.target][0]}" for tis in teleop.input]
        synthesized += [f"{TELEOP_FEEDBACK_KEY}.{obs_owners[tfs.origin][0]}" for tfs in teleop.feedback]
        for skey in synthesized:
            if skey in key_owners:
                raise ContractValidationError(
                    f"Frame key '{skey}' declared in '{key_owners[skey]}' collides with "
                    f"the synthesized teleop recording key of the same name; rename it."
                )

    contract = Contract(
        robot_type=robot_type,
        robot_interface=robot_interface,
        fps=fps,
        tasks=_parse_tasks(data.get("tasks")),
        adjunct=_parse_adjunct(data.get("adjunct")),
        teleop=teleop,
        **sections,
    )

    # Full eager validation: "load succeeded" means "valid in this environment".
    capability = _load_interface_capability(robot_interface)
    if capability is not None:
        _validate_interface(contract, capability)
    _validate_by_resolution(contract)
    return contract
