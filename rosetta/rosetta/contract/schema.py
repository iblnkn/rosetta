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
Contract YAML parsing and loading.

The document types live in :mod:`.model` and are re-exported here, so callers
import them from this module.

If ``load_contract`` returns, the contract is valid in this environment. Load
runs three checks:

1. Parse: unknown keys, missing align, malformed scalars, duplicate YAML keys,
   empty-but-present sections and codec paths that don't import are errors.
   Top-level ``x-*`` keys are ignored (they hold YAML anchors).
2. Interface: each channel's type exists, its chosen timeline is one the
   type provides, and its qos parses. Skipped when ROS is not installed
   (see ``_ROS_INSTALLED``).
3. Resolution: every spec is resolved once (see :mod:`.specs`), which runs
   the registry-backed rules (decodability, encoders, image geometry).

This module stays importable without ROS.
"""

from __future__ import annotations

import importlib.util
import warnings
from collections.abc import Hashable
from enum import StrEnum
from pathlib import Path
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

# Re-exported: callers import the document model from here.
from .model import (
    EXTENDED_SECTIONS,
    FRAME_SECTIONS,
    TELEOP_EVENT_NAMES,
    TELEOP_FEEDBACK_KEY,
    TELEOP_INPUT_KEY,
    Align,
    Channel,
    Contract,
    FieldKind,
    FrameEntry,
    ResamplePolicy,
    SafetyBehavior,
    Source,
    Task,
    Teleop,
    TeleopEventMap,
    TeleopFeedbackSource,
    TeleopInputSource,
    owning_key,
)
from .operators import lookup_operator

__all__ = [
    "FRAME_SECTIONS",
    "TELEOP_EVENT_NAMES",
    "TELEOP_FEEDBACK_KEY",
    "TELEOP_INPUT_KEY",
    "Align",
    "Channel",
    "Contract",
    "FieldKind",
    "FrameEntry",
    "ResamplePolicy",
    "SafetyBehavior",
    "Source",
    "Task",
    "Teleop",
    "TeleopEventMap",
    "TeleopFeedbackSource",
    "TeleopInputSource",
    "load_contract",
    "owning_key",
    "parse_contract",
]

SUPPORTED_ROBOT_INTERFACES = frozenset({"ros2"})
"""Robot interfaces a contract may declare (``robot_interface``)."""


#: Interface checks (message types, timelines, qos) need ROS. Without it they
#: are left to runtime; a present-but-broken install still fails loudly.
_ROS_INSTALLED = importlib.util.find_spec("rclpy") is not None

# Channel keys each section admits. `safety` is a command concern (actions
# only); `encoder` exists only where Rosetta publishes. "bare" covers tasks,
# adjunct and teleop events.
_DATA_CHANNEL_KEYS = frozenset({"topic", "type", "qos", "dtype", "decoder"})
_CHANNEL_KEYS = {
    "observations": _DATA_CHANNEL_KEYS,
    "actions": _DATA_CHANNEL_KEYS | {"encoder", "safety"},
    **dict.fromkeys(EXTENDED_SECTIONS, _DATA_CHANNEL_KEYS),
    "teleop.input": _DATA_CHANNEL_KEYS,
    "teleop.feedback": _DATA_CHANNEL_KEYS | {"encoder"},
    "bare": frozenset({"topic", "type", "qos"}),
}
#: Sections whose ``apply`` runs in the serve direction (Rosetta publishes them).
_SERVEABLE = frozenset({"actions", "teleop.feedback"})


EnumT = TypeVar("EnumT", bound=StrEnum)


def _require_str(value: Any, desc: str) -> str:
    """A non-empty string, used exactly as written: no case folding, stripping or coercion."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ContractValidationError(f"{desc} must be a non-empty string without surrounding spaces, got {value!r}")
    return value


def _validate_enum(value: Any, enum_cls: type[EnumT], field_name: str, context: str) -> EnumT:
    """Validate a string against an enum and return the member (a str subclass)."""
    value = _require_str(value, f"'{field_name}' in {context}")
    try:
        return enum_cls(value)
    except ValueError:
        valid = sorted(e.value for e in enum_cls)
        raise ContractValidationError(f"Invalid {field_name} '{value}' in {context}. Must be one of: {valid}") from None


def _check_keys(data: dict[str, Any], allowed: frozenset[str] | set[str], ctx: str) -> None:
    """Reject keys not in ``allowed``."""
    unknown = sorted(k for k in data if k not in allowed)
    if unknown:
        raise ContractValidationError(f"Unknown key(s) {unknown} in {ctx}. Allowed: {sorted(allowed)}")


def _require_mapping(data: Any, ctx: str) -> dict[str, Any]:
    """Require a YAML mapping."""
    if not isinstance(data, dict):
        raise ContractValidationError(f"{ctx} must be a mapping, got {type(data).__name__}")
    return data


def _mapping(data: Any, ctx: str, *, allowed: Iterable[str], required: Iterable[str] = ()) -> dict[str, Any]:
    """A YAML mapping with only ``allowed`` keys and every ``required`` one."""
    data = _require_mapping(data, ctx)
    _check_keys(data, frozenset(allowed), ctx)
    for field in required:
        if field not in data:
            raise ContractValidationError(f"Missing required field '{field}' in {ctx}")
    return data


def _parse_strict_int(value: Any, desc: str) -> int:
    """Parse an integer strictly: bools, strings, and non-integral floats are errors.

    ``30.0`` is accepted (YAML writers produce it); ``3.9`` is not, since
    truncating it would silently change a clock or window.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ContractValidationError(f"{desc} must be an integer, got {value!r}")


def _validate_dtype(dtype: str | None, context: str, required: bool = False) -> str | None:
    """Validate ``dtype`` if given; error if absent and ``required``."""
    if dtype is None:
        if required:
            raise ContractValidationError(f"Missing required field 'dtype' in {context}")
        return None

    dtype = _require_str(dtype, f"'dtype' in {context}")
    if not is_valid_dtype(dtype):
        raise ContractValidationError(
            f"Invalid dtype '{dtype}' in {context}. Supported: "
            f"{', '.join(sorted(SUPPORTED_NUMERIC_DTYPES) + sorted(SPECIAL_DTYPES))}"
        )
    return dtype


def _validate_kind(kind: str | None, names: Iterable[str] | None, context: str) -> FieldKind:
    """Validate a field kind (default CONTINUOUS) against the select length.

    Warns when a continuous field looks like a quaternion.
    """
    kind = _validate_enum(kind, FieldKind, "kind", context) if kind is not None else FieldKind.CONTINUOUS

    names = list(names) if names else []
    n = len(names)
    expected = kind.dims
    if expected is not None and n != expected:
        raise ContractValidationError(
            f"kind '{kind}' in {context} requires {expected} dim(s) but select has {n} ({names})."
        )

    # An x/y/z/w run is likely a quaternion that would get min_max normalized,
    # including one inside a larger select (an IMU's orientation.{x,y,z,w}).
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
    """Check that a ``"module:function"`` codec path imports, at load time.

    Uses the same resolver as runtime dispatch (``load_codec``), which also
    warms its cache so the first message doesn't pay the import.
    """
    if path is None:
        return None
    path = _require_str(path, f"codec path in {context}")

    try:
        load_codec(path)
    except ValueError as e:
        raise ContractValidationError(f"{e} (in {context})") from e
    except ImportError as e:
        raise ContractValidationError(f"Cannot import codec module for '{path}' in {context}: {e}") from e
    except AttributeError as e:
        raise ContractValidationError(f"Codec function not found for '{path}' in {context}: {e}") from e

    return path


def _validate_interface(contract: Contract) -> None:
    """Check every channel against the robot interface.

    Each ``type`` must be a real message type (bare channels included), each
    aligned source's ``timeline`` must be one the type provides, and each
    ``qos`` must parse. Requires ROS (see ``_ROS_INSTALLED``).
    """
    from ..robots.ros2.rclpy_utils import qos_profile_from_dict
    from ..robots.ros2.timelines import provided_timelines

    def check(channel: Channel, align: Align | None, ctx: str) -> None:
        try:
            provided = provided_timelines(channel.type)
        except ValueError as e:
            raise ContractValidationError(f"{ctx}: {e}") from e
        if align is not None and align.timeline not in provided:
            raise ContractValidationError(
                f"{ctx}.align: timeline '{align.timeline}' is not provided by '{channel.type}'. "
                f"This channel provides: {sorted(provided)}"
            )
        if channel.qos is not None:
            try:
                qos_profile_from_dict(channel.qos)
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
        _require_str(k, f"event name in {ctx}")
        _require_str(v, f"select.{k} in {ctx}")
    return dict(raw)


def _parse_apply(raw: Any, ctx: str, *, require_serveable: bool = False) -> list[tuple[str, Any]]:
    """
    Parse an ``apply`` operator list into ``[(name, args), ...]``.

    Items are bare strings (``rad2deg`` -> ``('rad2deg', None)``) or single-key
    mappings (``{resize: [h, w]}`` -> ``('resize', [h, w])``). With
    ``require_serveable``, a FORWARD_ONLY operator is an error.
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
            name = _require_str(name, f"apply[{i}] operator name in {ctx}")
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


_SOURCE_KEYS = frozenset({"channel", "align", "select", "apply", "kind"})

_ALIGN_KEYS = frozenset({"strategy", "timeline", "tolerance_ms"})


def _parse_channel(data: Any, ctx: str, section: str) -> Channel:
    """Parse one channel block, admitting only the keys its section allows."""
    data = _mapping(data, ctx, allowed=_CHANNEL_KEYS[section], required=("topic", "type"))
    topic = data["topic"]
    if not isinstance(topic, str) or not topic:
        raise ContractValidationError(f"'topic' in {ctx} must be a non-empty string, got {topic!r}")
    type_ = data["type"]
    if not isinstance(type_, str) or not type_:
        raise ContractValidationError(f"'type' in {ctx} must be a non-empty string, got {type_!r}")

    dtype = _validate_dtype(data.get("dtype"), ctx, required=section in EXTENDED_SECTIONS)

    safety = SafetyBehavior.NONE
    if "safety" in data:
        safety = _validate_enum(data["safety"], SafetyBehavior, "safety", ctx)

    # Copy qos: a YAML anchor aliases one dict across channels.
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
    """Parse a mandatory ``align`` block. Strategy and timeline have no defaults."""
    if data is None:
        raise ContractValidationError(
            f"Missing required 'align' block in {ctx} — every frame-clock entry must choose a strategy and a timeline."
        )
    data = _mapping(data, ctx, allowed=_ALIGN_KEYS, required=("strategy", "timeline"))

    strategy = _validate_enum(data["strategy"], ResamplePolicy, "strategy", ctx)
    # An open name; _validate_interface checks the channel provides it.
    timeline = _require_str(data["timeline"], f"'timeline' in {ctx}")

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


def _parse_source(data: Any, ctx: str, section: str) -> Source:
    """Parse one channel -> align -> select -> apply source."""
    data = _mapping(data, ctx, allowed=_SOURCE_KEYS, required=("channel", "align"))

    channel = _parse_channel(data["channel"], f"{ctx}.channel", section)
    align = _parse_align(data["align"], f"{ctx}.align")

    select = _parse_select_paths(data.get("select"), ctx)
    apply = _parse_apply(data.get("apply"), ctx, require_serveable=section in _SERVEABLE)
    kind = _validate_kind(data.get("kind"), select, ctx)

    return Source(
        channel=channel,
        align=align,
        select=select,
        apply=tuple(apply),
        kind=kind,
    )


def _parse_entry(key: str, value: Any, section: str) -> FrameEntry:
    """Parse one frame-key entry: a single source, or an ordered list of sources."""
    ctx = f"{section}.{key}"
    if isinstance(value, list):
        if not value:
            raise ContractValidationError(f"{ctx} has an empty source list")
        sources = tuple(_parse_source(item, f"{ctx}[{i}]", section) for i, item in enumerate(value))
    else:
        sources = (_parse_source(value, ctx, section),)
    return FrameEntry(key=key, sources=sources)


def _parse_frame_section(data: Any, section: str) -> tuple[FrameEntry, ...]:
    """Parse a frame-clock section: a mapping keyed by frame key."""
    if data is None:
        return ()
    data = _require_mapping(data, f"'{section}'")
    if not data:
        raise ContractValidationError(f"'{section}' is present but empty — omit the section instead")

    entries: list[FrameEntry] = []
    for raw_key, value in data.items():
        key = _require_str(raw_key, f"frame key in '{section}'")
        if section != "observations" and key.startswith(IMAGE_KEY_PREFIX):
            raise ContractValidationError(
                f"{section}.{key}: keys under '{IMAGE_KEY_PREFIX}' are only "
                f"allowed in 'observations' — image streams are observations."
            )
        entries.append(_parse_entry(key, value, section))
    return tuple(entries)


def _parse_bare_entry(data: Any, ctx: str) -> Channel:
    """Parse a ``{channel: ...}`` entry with no align (tasks, adjunct)."""
    data = _mapping(data, ctx, allowed=("channel",), required=("channel",))
    return _parse_channel(data["channel"], f"{ctx}.channel", "bare")


def _parse_tasks(data: Any) -> tuple[Task, ...]:
    """Parse the tasks section: a mapping of task key -> {channel}."""
    if data is None:
        return ()
    data = _require_mapping(data, "'tasks'")
    if not data:
        raise ContractValidationError("'tasks' is present but empty — omit the section instead")
    return tuple(
        Task(key=_require_str(key, "task key"), channel=_parse_bare_entry(value, f"tasks.{key}"))
        for key, value in data.items()
    )


def _parse_adjunct(data: Any) -> tuple[Channel, ...]:
    """Parse the adjunct section: a list of bare channels (record-only, no key, no align)."""
    if data is None:
        return ()
    if not isinstance(data, list):
        raise ContractValidationError(f"'adjunct' must be a list of channel entries, got {type(data).__name__}")
    if not data:
        raise ContractValidationError("'adjunct' is present but empty — omit the section instead")

    return tuple(_parse_bare_entry(item, f"adjunct[{i}]") for i, item in enumerate(data))


def _parse_teleop_events(data: Any) -> TeleopEventMap:
    """Parse teleop events: a bare channel plus a dict-form select. No align."""
    ctx = "teleop.events"
    data = _mapping(data, ctx, allowed=("channel", "select"), required=("channel", "select"))
    channel = _parse_channel(data["channel"], f"{ctx}.channel", "bare")
    select = _parse_select_map(data["select"], ctx)
    unknown = sorted(set(select) - TELEOP_EVENT_NAMES)
    if unknown:
        raise ContractValidationError(
            f"{ctx}.select: unknown event name(s) {unknown}. Valid events: {sorted(TELEOP_EVENT_NAMES)}"
        )
    return TeleopEventMap(channel=channel, select=select)


def _parse_teleop_refs(data: Any, section: str, ref_field: str, entries: tuple[FrameEntry, ...], wrap) -> tuple:
    """Parse ``teleop.input`` or ``teleop.feedback``: sources each bound to one topic.

    ``ref_field`` (``target``/``origin``) must name a topic owned by exactly
    one of ``entries``, and no two sources may name the same topic.
    """
    if data is None:
        return ()
    if not isinstance(data, list):
        raise ContractValidationError(f"'{section}' must be a list, got {type(data).__name__}")
    if not data:
        raise ContractValidationError(f"'{section}' has an empty source list")
    parsed = []
    seen: set[str] = set()
    for i, raw in enumerate(data):
        ctx = f"{section}[{i}]"
        item = _mapping(raw, ctx, allowed=_SOURCE_KEYS | {ref_field}, required=(ref_field, "channel", "align"))
        ref = _require_str(item[ref_field], f"'{ref_field}' in {ctx}")
        owning_key(entries, ref, f"{ctx}: {ref_field}")
        if ref in seen:
            raise ContractValidationError(
                f"{ctx}: duplicate {ref_field} '{ref}'; each topic may appear once in {section}."
            )
        seen.add(ref)
        source = _parse_source({k: v for k, v in item.items() if k != ref_field}, ctx, section)
        parsed.append(wrap(source, ref))
    return tuple(parsed)


def _parse_teleop(data: Any, actions: tuple[FrameEntry, ...], observations: tuple[FrameEntry, ...]) -> Teleop | None:
    """Parse teleop input / events / feedback.

    ``target``/``origin`` references are checked against the parsed
    ``actions``/``observations``.
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
        input=_parse_teleop_refs(data.get("input"), "teleop.input", "target", actions, TeleopInputSource),
        events=events,
        feedback=_parse_teleop_refs(
            data.get("feedback"), "teleop.feedback", "origin", observations, TeleopFeedbackSource
        ),
    )


class _StrictYamlLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys.

    PyYAML silently keeps the last value, so a copy-paste slip could drop a
    whole frame entry. ``<<:`` merge keys still work because
    ``flatten_mapping`` resolves them before the duplicate scan.
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
    """Parse robot_type, robot_interface and fps."""
    if "robot_type" not in data:
        raise ContractValidationError("robot_type is required")
    robot_type = _require_str(data["robot_type"], "robot_type")

    robot_interface = data.get("robot_interface")
    if not robot_interface:
        raise ContractValidationError(f"robot_interface is required. Supported: {sorted(SUPPORTED_ROBOT_INTERFACES)}")
    robot_interface = _require_str(robot_interface, "robot_interface")
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
    """Load and validate a contract YAML file.

    Raises FileNotFoundError or ContractValidationError.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Contract file not found: {path}")
    return parse_contract(path.read_text(encoding="utf-8"), source=str(path))


def parse_contract(text: str, *, source: str = "<contract>") -> Contract:
    """Validate contract YAML from a string; ``source`` labels error messages.

    For callers that also keep the text (e.g. the recorder embeds it in bag
    metadata), so the text they hold is exactly the text validated.
    """
    try:
        data = yaml.load(text, Loader=_StrictYamlLoader) or {}
    except yaml.YAMLError as e:
        raise ContractValidationError(f"Invalid YAML in {source}: {e}") from e

    if not isinstance(data, dict):
        raise ContractValidationError(f"Contract must be a YAML mapping, got {type(data).__name__}")

    # Top-level x-* keys hold YAML anchors (e.g. shared QoS blocks).
    data = {k: v for k, v in data.items() if not str(k).startswith("x-")}
    _check_keys(data, _TOP_LEVEL_KEYS, "contract")

    for name in ("tasks", "adjunct", "teleop", *FRAME_SECTIONS):
        if name in data and data[name] is None:
            raise ContractValidationError(f"'{name}' is present but null — omit the section instead")

    robot_type, robot_interface, fps = _parse_header(data)

    # Spec resolution below looks up dtypes and encoders in the codec registry.
    discover_codecs()

    sections = {name: _parse_frame_section(data.get(name), name) for name in FRAME_SECTIONS}

    # A key in two sections would merge into one recording column (FrameLayout
    # groups by key) while narrower views like iter_policy_specs disagree
    # about its contents.
    key_owners: dict[str, str] = {}
    for section_name, entries in sections.items():
        for entry in entries:
            owner = key_owners.setdefault(entry.key, section_name)
            if owner != section_name:
                raise ContractValidationError(
                    f"Frame key '{entry.key}' is declared in both '{owner}' and "
                    f"'{section_name}'; a key must belong to exactly one section."
                )

    teleop = _parse_teleop(data.get("teleop"), sections["actions"], sections["observations"])

    # specs.py records teleop under teleop.input.<owner> / teleop.feedback.<owner>;
    # a declared key of the same name would merge with that column.
    if teleop is not None:
        synthesized = [
            f"{TELEOP_INPUT_KEY}.{owning_key(sections['actions'], tis.target, 'teleop.input: target')}"
            for tis in teleop.input
        ]
        synthesized += [
            f"{TELEOP_FEEDBACK_KEY}.{owning_key(sections['observations'], tfs.origin, 'teleop.feedback: origin')}"
            for tfs in teleop.feedback
        ]
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

    if _ROS_INSTALLED:
        _validate_interface(contract)
    # Resolve every spec and lay out the frame now, so the registry-backed
    # rules (decodability, encoders, image geometry, shared keys) fail at load.
    # Imported here because specs and layout import this module.
    from ..frames.layout import FrameLayout
    from .specs import iter_specs

    FrameLayout(list(iter_specs(contract)))
    return contract
