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
What the contract *says*: the typed YAML document model, validation, and loading.

The runtime consumes the resolved specs in :mod:`rosetta.contract.specs`,
not these types.

A contract binds channels to frame keys. Every frame-clock entry reads::

    channel (provides) -> align (chooses a timeline; mandatory) -> select
    -> apply -> the mapping key

The dataclasses here mirror that shape exactly: a :class:`Channel` is the
robot-interface half of an entry (topic/type/qos speak the interface's
dialect), an :class:`Align` chooses a timeline by name, a
:class:`Source` is one channel+align+select+apply pipeline, and a
:class:`FrameEntry` is a frame key with one or more ordered sources (several
sources = concatenation for observations, splitting for actions).

Timelines are open names, not an enum: everything before align (the robot
interface) is responsible for producing a message's timelines under names
(``receive``, ``header``, ...); align only selects one by name. Validation
asks the interface what a channel provides and rejects a timeline it doesn't.

The types have no ROS dependencies, making them easy to use in offline
tooling, tests, and type checking. Loading a contract is strict: unknown
keys, missing align, or a timeline the channel cannot provide are load-time
errors, never silent fallbacks. Top-level ``x-*`` keys are ignored (YAML
anchor holders).
"""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from ..frames.codecs import discover_codecs
from . import builtin_operators  # noqa: F401 - registers built-in operators
from .errors import ContractValidationError
from .operators import OPERATOR_REGISTRY, discover_operators

# =============================================================================
# Constants
# =============================================================================

LEROBOT_SPECIAL_DTYPES = frozenset(["video", "image", "string"])
"""Special LeRobot dtypes that aren't numpy dtypes."""

DEPTH_ENCODINGS = frozenset({"mono16", "16uc1", "32fc1", "32fc"})
"""
Depth image encodings - not supported due to LeRobot limitations.

LeRobot currently lacks proper depth image handling:
- Forces all images through PIL.convert("RGB")
- No depth-specific normalization or transforms
- Precision loss when converting uint16/float32 to uint8
"""

SUPPORTED_ROBOT_INTERFACES = frozenset({"ros2"})
"""Robot interfaces a contract may declare (``robot_interface``)."""


def _is_valid_numpy_dtype_string(dtype: str) -> bool:
    """Return True if ``dtype`` names a constructible numpy dtype."""
    try:
        np.dtype(dtype)
        return True
    except TypeError:
        return False


def is_valid_lerobot_dtype(dtype: str) -> bool:
    """
    Check if dtype is valid for LeRobot datasets.

    Valid dtypes are:
    - Any valid numpy dtype string (float32, float64, int32, int64, bool, etc.)
    - Special LeRobot types: video, image, string
    """
    return dtype in LEROBOT_SPECIAL_DTYPES or _is_valid_numpy_dtype_string(dtype)


# =============================================================================
# Enums
# =============================================================================


class ResamplePolicy(str, Enum):
    """Alignment strategy: how a stream's samples map onto frame ticks."""

    HOLD = "hold"  # Carry forward last value
    ASOF = "asof"  # Last value within tolerance window
    DROP = "drop"  # Only value if arrived within step window


class SafetyBehavior(str, Enum):
    """Safety behavior when policy fails to produce actions."""

    NONE = "none"  # No safety action (default)
    ZEROS = "zeros"  # Send zero commands
    HOLD = "hold"  # Hold last action


class FieldKind(str, Enum):
    """Value type of a state/action field.

    Tells the VLA frameworks how to normalize or rotate the values. LeRobot
    ignores it. Default is continuous (plain min_max), which leaves existing
    contracts unchanged.
    """

    CONTINUOUS = "continuous"  # plain scalar/vector (joints, positions, velocities)
    QUATERNION = "quaternion"  # 4D rotation [x, y, z, w]
    EULER_RPY = "euler_rpy"  # 3D roll/pitch/yaw
    AXIS_ANGLE = "axis_angle"  # 3D rotation vector
    ROTATION_6D = "rotation_6d"  # 6D continuous rotation representation
    BINARY = "binary"  # discrete on/off (e.g. gripper open/close)


# Required dim count per kind (None = any), checked at contract load.
FIELD_KIND_DIMS: dict[str, int | None] = {
    "continuous": None,
    "quaternion": 4,
    "euler_rpy": 3,
    "axis_angle": 3,
    "rotation_6d": 6,
    "binary": None,
}


# =============================================================================
# Contract Dataclasses (mirror the YAML shape)
# =============================================================================


@dataclass(frozen=True, slots=True)
class Channel:
    """One robot-interface channel: what a different pub/sub ecosystem would replace.

    Fields speak the declared interface's dialect (ros2: topic/type/qos).
    Codec concerns live here too: dtype, custom decoder/encoder, and the
    safety behavior for command channels. Image pixel encoding is not one of
    them — it's read from the message itself at decode time (``sensor_msgs/Image``
    carries a required ``encoding`` field; ``CompressedImage`` decodes to RGB
    unconditionally via cv2), so there is nothing to declare here.
    """

    topic: str
    type: str
    qos: dict[str, Any] | None = None
    dtype: str | None = None
    decoder: str | None = None  # Custom decoder path: "module.path:function_name"
    encoder: str | None = None  # Custom encoder path (actions/feedback only)
    safety: str = "none"  # SafetyBehavior (action channels only)


@dataclass(frozen=True, slots=True)
class Align:
    """How a source's samples land on the clock defined by the contract's ``fps``.

    ``timeline`` is an open name selecting one of the timelines the channel
    provides (the robot interface produces them; align only chooses).
    Mandatory on every observation/action entry — there are no defaults.
    """

    strategy: str  # ResamplePolicy value: "hold" | "asof" | "drop"
    timeline: str  # open timeline name, e.g. "receive" | "header"
    tolerance_ms: int = 0  # required iff strategy == "asof"


@dataclass(frozen=True, slots=True)
class Source:
    """One channel -> align -> select -> apply pipeline feeding a frame key."""

    channel: Channel
    align: Align
    select: list[str] | None = None  # field paths to project (list form)
    apply: tuple[tuple[str, Any], ...] = ()  # ordered operators: ((name, args), ...)
    kind: str = "continuous"  # FieldKind representation for VLA frameworks


@dataclass(frozen=True, slots=True)
class FrameEntry:
    """A frame key and its ordered sources.

    Several sources mean concatenation (observations: values joined in
    order) or splitting (actions: one vector sliced across channels in
    order).
    """

    key: str
    sources: tuple[Source, ...]


@dataclass(frozen=True, slots=True)
class Task:
    """Task channel (e.g. prompts). Not sampled on the frame's tick clock — no align."""

    key: str
    channel: Channel


@dataclass(frozen=True, slots=True)
class TeleopEventMap:
    """Teleoperator event button mappings for HIL-SERL. No align — events are edges."""

    channel: Channel
    select: dict[str, str]  # event_name -> field path (dict form of select)


@dataclass(frozen=True, slots=True)
class Teleop:
    """Teleoperator role sections: input / events / feedback."""

    input: FrameEntry | None
    events: TeleopEventMap | None
    feedback: FrameEntry | None


#: Synthesized frame keys for the teleop role entries.
TELEOP_INPUT_KEY = "teleop.input"
TELEOP_FEEDBACK_KEY = "teleop.feedback"

#: The frame-clock sections of a contract, in resolution order.
FRAME_SECTIONS = ("observations", "actions", "rewards", "signals", "info", "complementary_data")


@dataclass(frozen=True, slots=True)
class Contract:
    """Top-level contract describing a robot's policy I/O surface."""

    robot_type: str
    robot_interface: str
    fps: int
    observations: list[FrameEntry]
    actions: list[FrameEntry]
    tasks: list[Task]
    adjunct: list[Channel]
    rewards: list[FrameEntry]
    signals: list[FrameEntry]
    info: list[FrameEntry]
    complementary_data: list[FrameEntry]
    teleop: Teleop | None = None

    def frame_entries(self) -> Iterable[tuple[str, FrameEntry]]:
        """Yield ``(section, entry)`` over all frame-clock sections."""
        for section in FRAME_SECTIONS:
            for entry in getattr(self, section):
                yield section, entry


# =============================================================================
# YAML Loading - Validation Helpers
# =============================================================================


def _validate_enum(value: str, enum_cls: type, field_name: str, context: str) -> str:
    """Validate that a string is a valid enum value."""
    value = str(value).lower().strip()
    valid = {e.value for e in enum_cls}
    if value not in valid:
        raise ContractValidationError(f"Invalid {field_name} '{value}' in {context}. Must be one of: {sorted(valid)}")
    return value


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


def _validate_dtype(dtype: str | None, context: str, required: bool = False) -> str | None:
    """Validate dtype if provided."""
    if dtype is None:
        if required:
            raise ContractValidationError(f"Missing required field 'dtype' in {context}")
        return None

    dtype = str(dtype).lower()
    if not is_valid_lerobot_dtype(dtype):
        raise ContractValidationError(
            f"Invalid dtype '{dtype}' in {context}. "
            f"Must be a valid numpy dtype or one of: {sorted(LEROBOT_SPECIAL_DTYPES)}"
        )
    return dtype


def _validate_kind(kind: str | None, names: list[str] | None, context: str) -> str:
    """Validate a field kind. Returns the representation, default 'continuous'.

    Errors on an unknown kind or a dim count that doesn't match. Warns when an
    untagged field looks like a quaternion.
    """
    kind = _validate_enum(kind, FieldKind, "kind", context) if kind is not None else "continuous"

    n = len(names) if names else 0
    expected = FIELD_KIND_DIMS.get(kind)
    if expected is not None and n != expected:
        raise ContractValidationError(
            f"kind '{kind}' in {context} requires {expected} dim(s) but select has {n} ({names})."
        )

    # Warn if an untagged field has an x/y/z/w run (likely a quaternion getting
    # min_max'd). Catches it inside a larger select too, e.g. an IMU whose first
    # 4 dims are orientation.{x,y,z,w}. Split that into its own quaternion spec.
    if kind == "continuous" and names:
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


def _validate_converter_path(path: str | None, context: str) -> str | None:
    """
    Validate converter path exists at contract load time.

    Path format: "module.path:function_name"

    Args:
    ----
        path: Converter path or None
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

    if ":" not in path:
        raise ContractValidationError(
            f"Invalid converter path '{path}' in {context}. Expected format: 'module.path:function_name'"
        )

    module_path, func_name = path.rsplit(":", 1)
    if not module_path or not func_name:
        raise ContractValidationError(
            f"Invalid converter path '{path}' in {context}. Expected format: 'module.path:function_name'"
        )

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ContractValidationError(f"Cannot import converter module '{module_path}' in {context}: {e}") from e

    if not hasattr(module, func_name):
        raise ContractValidationError(f"Function '{func_name}' not found in module '{module_path}' ({context})")

    return path


def _check_timeline_provided(msg_type: str, timeline: str, ctx: str) -> None:
    """Reject a timeline the channel cannot provide — at load, not first message.

    Timeline production is robot-interface knowledge, so the check consults
    the interface implementation (lazily — the contract layer must stay
    importable without ROS). In an environment where the interface isn't
    importable, the check is deferred to runtime ingest, which drops
    messages missing their named timeline.
    """
    try:
        from ..robots.ros2.ros2_utils import provided_timelines
    except ImportError:
        return  # non-ROS tooling env: defer to runtime enforcement

    try:
        provided = provided_timelines(msg_type)
    except ValueError as e:
        raise ContractValidationError(f"{ctx}: {e}") from e

    if timeline not in provided:
        raise ContractValidationError(
            f"{ctx}: timeline '{timeline}' is not provided by '{msg_type}'. This channel provides: {sorted(provided)}"
        )


# =============================================================================
# YAML Loading - select / apply helpers
# =============================================================================


def _parse_select(raw: Any, ctx: str, *, dict_form: bool = False) -> "list[str] | dict[str, str] | None":
    """
    Parse a ``select`` field.

    List form (observations/actions) projects field paths: ``[a, b, c]``.
    Dict form (Joy events) maps names to field paths: ``{name: buttons.10}``.
    """
    if raw is None:
        return None
    if dict_form:
        if not isinstance(raw, dict):
            raise ContractValidationError(f"'select' must be a mapping in {ctx}")
        return {str(k): str(v) for k, v in raw.items()}
    if not isinstance(raw, list):
        raise ContractValidationError(f"'select' must be a list in {ctx}")
    return [str(x) for x in raw]


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
    # Pull in entry-point operator plugins so custom operators resolve by name here.
    discover_operators()

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

        cls = OPERATOR_REGISTRY.get(name)
        if cls is None:
            known = ", ".join(sorted(OPERATOR_REGISTRY)) or "(none)"
            raise ContractValidationError(
                f"Unknown operator '{name}' in apply[{i}] of {ctx}. Registered operators: {known}"
            )
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

# Per-section channel-field rules. `safety` is a command concern (actions
# only — declaring it on teleop feedback is an error, not a silent override);
# `encoder` exists only where Rosetta publishes.
_OBS_CHANNEL_KEYS = frozenset({"topic", "type", "qos", "dtype", "decoder"})
_ACTION_CHANNEL_KEYS = frozenset({"topic", "type", "qos", "dtype", "decoder", "encoder", "safety"})
_FEEDBACK_CHANNEL_KEYS = _ACTION_CHANNEL_KEYS - {"safety"}
_DATA_CHANNEL_KEYS = frozenset({"topic", "type", "qos", "dtype", "decoder"})
_BARE_CHANNEL_KEYS = frozenset({"topic", "type", "qos"})

_SOURCE_KEYS = frozenset({"channel", "align", "select", "apply", "kind"})


@dataclass(frozen=True, slots=True)
class _SectionRules:
    """How one contract section parses its sources."""

    channel_keys: frozenset[str]
    serveable: bool = False  # apply must run in the serve direction (publishes)
    dtype_required: bool = False  # extended sections declare their dtype

    def parse_channel(self, data: Any, ctx: str) -> Channel:
        data = _require_mapping(data, ctx)
        _check_keys(data, self.channel_keys, ctx)
        _require_fields(data, ["topic", "type"], ctx)
        if not data["topic"]:
            raise ContractValidationError(f"Empty topic in {ctx}")

        dtype = _validate_dtype(data.get("dtype"), ctx, required=self.dtype_required)
        if self.dtype_required and dtype in ("video", "image"):
            raise ContractValidationError(f"Invalid dtype '{dtype}' in {ctx}: extended sections are never images.")

        safety = "none"
        if "safety" in data:
            safety = _validate_enum(data["safety"], SafetyBehavior, "safety", ctx)

        return Channel(
            topic=data["topic"],
            type=data["type"],
            qos=data.get("qos"),
            dtype=dtype,
            decoder=_validate_converter_path(data.get("decoder"), f"{ctx}.decoder"),
            encoder=_validate_converter_path(data.get("encoder"), f"{ctx}.encoder"),
            safety=safety,
        )


_SECTION_RULES: dict[str, _SectionRules] = {
    "observations": _SectionRules(_OBS_CHANNEL_KEYS),
    "actions": _SectionRules(_ACTION_CHANNEL_KEYS, serveable=True),
    "rewards": _SectionRules(_DATA_CHANNEL_KEYS, dtype_required=True),
    "signals": _SectionRules(_DATA_CHANNEL_KEYS, dtype_required=True),
    "info": _SectionRules(_DATA_CHANNEL_KEYS, dtype_required=True),
    "complementary_data": _SectionRules(_DATA_CHANNEL_KEYS, dtype_required=True),
    "teleop.input": _SectionRules(_DATA_CHANNEL_KEYS),
    "teleop.feedback": _SectionRules(_FEEDBACK_CHANNEL_KEYS, serveable=True),
}

# Extended sections: record-only frame entries (never images, dtype mandatory).
_EXTENDED_SECTIONS = frozenset({"rewards", "signals", "info", "complementary_data"})


def _parse_align(data: Any, ctx: str) -> Align:
    """Parse a mandatory ``align`` block. No defaults: strategy and timeline are required."""
    if data is None:
        raise ContractValidationError(
            f"Missing required 'align' block in {ctx} — every frame-clock entry must choose a strategy and a timeline."
        )
    data = _require_mapping(data, ctx)
    _check_keys(data, {"strategy", "timeline", "tolerance_ms"}, ctx)
    _require_fields(data, ["strategy", "timeline"], ctx)

    strategy = _validate_enum(data["strategy"], ResamplePolicy, "strategy", ctx)
    timeline = str(data["timeline"]).lower().strip()
    if not timeline:
        raise ContractValidationError(f"Empty timeline in {ctx}")

    tol_raw = data.get("tolerance_ms")
    if strategy == ResamplePolicy.ASOF.value:
        if tol_raw is None:
            raise ContractValidationError(f"'asof' alignment in {ctx} requires 'tolerance_ms'")
        tolerance_ms = int(tol_raw)
        if tolerance_ms <= 0:
            raise ContractValidationError(f"'tolerance_ms' must be positive in {ctx}, got {tolerance_ms}")
    else:
        if tol_raw is not None:
            raise ContractValidationError(
                f"'tolerance_ms' in {ctx} is only valid with strategy 'asof' (got '{strategy}')"
            )
        tolerance_ms = 0

    return Align(strategy=strategy, timeline=timeline, tolerance_ms=tolerance_ms)


def _parse_source(data: Any, ctx: str, rules: _SectionRules) -> Source:
    """Parse one channel -> align -> select -> apply source."""
    data = _require_mapping(data, ctx)
    _check_keys(data, _SOURCE_KEYS, ctx)
    _require_fields(data, ["channel", "align"], ctx)

    channel = rules.parse_channel(data["channel"], f"{ctx}.channel")
    align = _parse_align(data["align"], f"{ctx}.align")
    _check_timeline_provided(channel.type, align.timeline, f"{ctx}.align")

    select = _parse_select(data.get("select"), ctx)
    apply = _parse_apply(data.get("apply"), ctx, require_serveable=rules.serveable)
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
    rules = _SECTION_RULES[section]
    ctx = f"{section}.{key}"
    if isinstance(value, list):
        if not value:
            raise ContractValidationError(f"{ctx} has an empty source list")
        sources = tuple(_parse_source(item, f"{ctx}[{i}]", rules) for i, item in enumerate(value))
    else:
        sources = (_parse_source(value, ctx, rules),)
    return FrameEntry(key=str(key), sources=sources)


def _parse_frame_section(data: Any, section: str) -> list[FrameEntry]:
    """Parse a frame-clock section: a mapping keyed by frame key."""
    if data is None:
        return []
    if isinstance(data, list):
        raise ContractValidationError(
            f"'{section}' must be a mapping keyed by frame key, got a list "
            f"(the v1 list-of-entries contract format is no longer supported)"
        )
    data = _require_mapping(data, f"'{section}'")

    entries: list[FrameEntry] = []
    for key, value in data.items():
        if section in _EXTENDED_SECTIONS and str(key).startswith("observation.images."):
            raise ContractValidationError(
                f"{section}.{key}: extended sections are never images "
                f"(keys under 'observation.images.' are not allowed here)"
            )
        entries.append(_parse_entry(str(key), value, section))
    return entries


def _parse_tasks(data: Any) -> list[Task]:
    """Parse the tasks section: a mapping of task key -> {channel}."""
    if data is None:
        return []
    if isinstance(data, list):
        raise ContractValidationError(
            "'tasks' must be a mapping keyed by task key, got a list "
            "(the v1 list-of-entries contract format is no longer supported)"
        )
    data = _require_mapping(data, "'tasks'")

    tasks: list[Task] = []
    for key, value in data.items():
        ctx = f"tasks.{key}"
        entry = _require_mapping(value, ctx)
        _check_keys(entry, {"channel"}, ctx)
        _require_fields(entry, ["channel"], ctx)
        channel = _SectionRules(_BARE_CHANNEL_KEYS).parse_channel(entry["channel"], f"{ctx}.channel")
        tasks.append(Task(key=str(key), channel=channel))
    return tasks


def _parse_adjunct(data: Any) -> list[Channel]:
    """Parse the adjunct section: a list of bare channels (record-only, no key, no align)."""
    if data is None:
        return []
    if not isinstance(data, list):
        raise ContractValidationError(f"'adjunct' must be a list of channel entries, got {type(data).__name__}")

    channels: list[Channel] = []
    for i, item in enumerate(data):
        ctx = f"adjunct[{i}]"
        entry = _require_mapping(item, ctx)
        _check_keys(entry, {"channel"}, ctx)
        _require_fields(entry, ["channel"], ctx)
        channels.append(_SectionRules(_BARE_CHANNEL_KEYS).parse_channel(entry["channel"], f"{ctx}.channel"))
    return channels


def _parse_teleop_events(data: Any) -> TeleopEventMap:
    """Parse teleop events: a bare channel plus a dict-form select. No align."""
    ctx = "teleop.events"
    data = _require_mapping(data, ctx)
    _check_keys(data, {"channel", "select"}, ctx)
    _require_fields(data, ["channel", "select"], ctx)

    channel = _SectionRules(_BARE_CHANNEL_KEYS).parse_channel(data["channel"], f"{ctx}.channel")
    select = _parse_select(data["select"], ctx, dict_form=True)
    return TeleopEventMap(channel=channel, select=select)


def _parse_teleop(data: Any) -> Teleop | None:
    """Parse teleop role sections: input / events / feedback."""
    if not data:
        return None
    data = _require_mapping(data, "'teleop'")
    _check_keys(data, {"input", "events", "feedback"}, "'teleop'")

    input_entry = None
    if data.get("input") is not None:
        input_entry = _parse_entry(TELEOP_INPUT_KEY, data["input"], "teleop.input")

    events = None
    if data.get("events") is not None:
        events = _parse_teleop_events(data["events"])

    feedback_entry = None
    if data.get("feedback") is not None:
        feedback_entry = _parse_entry(TELEOP_FEEDBACK_KEY, data["feedback"], "teleop.feedback")

    return Teleop(input=input_entry, events=events, feedback=feedback_entry)


# =============================================================================
# Main Loader
# =============================================================================

_TOP_LEVEL_KEYS = frozenset(
    {
        "robot_type",
        "robot_interface",
        "fps",
        "observations",
        "actions",
        "tasks",
        "adjunct",
        "rewards",
        "signals",
        "info",
        "complementary_data",
        "teleop",
    }
)


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
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ContractValidationError(f"Invalid YAML in {path}: {e}") from e

    if not isinstance(data, dict):
        raise ContractValidationError(f"Contract must be a YAML mapping, got {type(data).__name__}")

    # Top-level x-* keys hold YAML anchors (e.g. shared QoS blocks) — ignored.
    data = {k: v for k, v in data.items() if not str(k).startswith("x-")}
    _check_keys(data, _TOP_LEVEL_KEYS, "contract")

    robot_type = data.get("robot_type")
    if not robot_type:
        raise ContractValidationError("robot_type is required")

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
    try:
        fps = int(data["fps"])
    except (TypeError, ValueError) as e:
        raise ContractValidationError(f"fps must be an integer, got {data['fps']!r}") from e
    if fps <= 0:
        raise ContractValidationError(f"fps must be positive, got {fps}")

    # Pull in entry-point codec plugins so registry-keyed dtype/encoder lookups
    # during spec building see plugin-provided codecs.
    discover_codecs()

    return Contract(
        robot_type=str(robot_type),
        robot_interface=robot_interface,
        fps=fps,
        observations=_parse_frame_section(data.get("observations"), "observations"),
        actions=_parse_frame_section(data.get("actions"), "actions"),
        tasks=_parse_tasks(data.get("tasks")),
        adjunct=_parse_adjunct(data.get("adjunct")),
        rewards=_parse_frame_section(data.get("rewards"), "rewards"),
        signals=_parse_frame_section(data.get("signals"), "signals"),
        info=_parse_frame_section(data.get("info"), "info"),
        complementary_data=_parse_frame_section(data.get("complementary_data"), "complementary_data"),
        teleop=_parse_teleop(data.get("teleop")),
    )
