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
Contract: types, validation, and loading for robot I/O contracts.

This module defines the Contract dataclass and all related types, plus
YAML loading and validation. The types have no ROS dependencies, making
them easy to use in offline tooling, tests, and type checking.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from lerobot.utils.utils import is_valid_numpy_dtype_string


# =============================================================================
# Constants
# =============================================================================

LEROBOT_SPECIAL_DTYPES = frozenset(["video", "image", "string"])
"""Special LeRobot dtypes that aren't numpy dtypes."""

DEPTH_ENCODINGS = frozenset({"mono16", "16uc1", "32fc1", "32fc"})
"""Depth image encodings - not supported due to LeRobot limitations.

LeRobot currently lacks proper depth image handling:
- Forces all images through PIL.convert("RGB")
- No depth-specific normalization or transforms
- Precision loss when converting uint16/float32 to uint8
"""


def is_valid_lerobot_dtype(dtype: str) -> bool:
    """Check if dtype is valid for LeRobot datasets.

    Valid dtypes are:
    - Any valid numpy dtype string (float32, float64, int32, int64, bool, etc.)
    - Special LeRobot types: video, image, string
    """
    return dtype in LEROBOT_SPECIAL_DTYPES or is_valid_numpy_dtype_string(dtype)


# =============================================================================
# Exceptions
# =============================================================================


class ContractValidationError(ValueError):
    """Raised when contract YAML is invalid."""

    pass


# =============================================================================
# Enums
# =============================================================================


class ResamplePolicy(str, Enum):
    """Resampling strategy for observation streams."""

    HOLD = "hold"  # Carry forward last value
    ASOF = "asof"  # Last value within tolerance window
    DROP = "drop"  # Only value if arrived within step window


class StampSource(str, Enum):
    """Timestamp source for observations."""

    RECEIVE = "receive"  # Use receive time
    HEADER = "header"  # Use message header stamp


class SafetyBehavior(str, Enum):
    """Safety behavior when policy fails to produce actions."""

    NONE = "none"  # No safety action (default)
    ZEROS = "zeros"  # Send zero commands
    HOLD = "hold"  # Hold last action


class BufferingStrategy(str, Enum):
    """Buffering strategy for TRANSIENT_LOCAL topics."""

    NO_BUFFER = "no_buffer"  # Don't buffer (default for non-transient topics)
    ACCUMULATE = "accumulate"  # Keep all messages up to history depth
    RESUBSCRIBE_ON_START = "resubscribe_on_start"  # Re-subscribe when recording starts


class ResetMode(str, Enum):
    """Reset mechanism mode."""

    MANUAL = "manual"  # User manually resets environment
    SERVICE = "service"  # Call ROS2 service to reset
    TOPIC = "topic"  # Publish to topic to trigger reset


# =============================================================================
# Contract Dataclasses
# =============================================================================


@dataclass(frozen=True, slots=True)
class AlignSpec:
    """Timestamp alignment behavior for an observation stream."""

    strategy: str = "hold"  # "hold" | "asof" | "drop"
    tol_ms: int = 0  # as-of tolerance in ms
    stamp: str = "receive"  # "receive" | "header"


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    """Observation stream description from contract YAML."""

    key: str
    topic: str
    type: str
    selector: dict[str, Any] | None = None
    image: dict[str, Any] | None = None
    align: AlignSpec | None = None
    qos: dict[str, Any] | None = None
    dtype: str | None = None
    decoder: str | None = None  # Custom decoder path: "module.path:function_name"


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """Action stream description from contract YAML."""

    key: str
    publish_topic: str
    type: str
    selector: dict[str, Any] | None = None
    from_tensor: dict[str, Any] | None = None
    publish_qos: dict[str, Any] | None = None
    publish_strategy: dict[str, Any] | None = None
    safety_behavior: str = "none"
    decoder: str | None = None  # Custom decoder path: "module.path:function_name"
    encoder: str | None = None  # Custom encoder path: "module.path:function_name"


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Task channel description (e.g., prompts)."""

    key: str
    topic: str
    type: str
    qos: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class AdjunctSpec:
    """Adjunct topic for recording only (not processed at inference)."""

    topic: str
    type: str
    qos: dict[str, Any] | None = None
    buffering_strategy: str | None = None  # "no_buffer" | "accumulate" | "resubscribe_on_start"


@dataclass(frozen=True, slots=True)
class TeleopEventsSpec:
    """Teleoperator event button mappings for HIL-SERL."""

    topic: str
    msg_type: str
    mappings: dict[str, str]  # event_name -> selector
    qos: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class TeleopSpec:
    """Teleoperator configuration."""

    inputs: list[ObservationSpec]
    events: TeleopEventsSpec | None
    feedback: list[ActionSpec]


@dataclass(frozen=True, slots=True)
class ResetSpec:
    """Reset mechanism configuration."""

    mode: str = "manual"
    service: str | None = None
    service_type: str = "std_srvs/srv/Trigger"
    topic: str | None = None
    topic_type: str = "std_msgs/msg/Empty"
    reset_time_s: float = 5.0


@dataclass(frozen=True, slots=True)
class VisualizationSpec:
    """Visualization configuration for 3D robot rendering."""

    urdf_path: str | None = None
    urdf_parameter: str | None = None
    tf_topics: list[str] | None = None
    tf_base_frame: str = "world"


@dataclass(frozen=True, slots=True)
class Contract:
    """Top-level contract describing a policy's ROS 2 I/O surface."""

    robot_type: str
    fps: int
    max_duration_s: float
    observations: list[ObservationSpec]
    actions: list[ActionSpec]
    tasks: list[TaskSpec]
    recording: dict[str, Any]
    adjunct: list[AdjunctSpec]
    rewards: list[ObservationSpec]
    signals: list[ObservationSpec]
    info: list[ObservationSpec]
    complementary_data: list[ObservationSpec]
    teleop: TeleopSpec | None = None
    reset: ResetSpec | None = None
    visualization: VisualizationSpec | None = None
    timestamp_source: str = "receive"


# =============================================================================
# Runtime Stream Specs
# =============================================================================


@dataclass(frozen=True, slots=True)
class StreamSpec:
    """Base configuration for observation and action streams."""

    key: str
    topic: str
    msg_type: str
    names: list[str]
    fps: int
    stamp_src: str


@dataclass(frozen=True, slots=True)
class ObservationStreamSpec(StreamSpec):
    """Resolved observation stream configuration for runtime use."""

    is_image: bool
    image_resize: tuple[int, int] | None
    image_encoding: str
    image_channels: int
    resample_policy: str
    asof_tol_ms: int
    dtype: str = "float32"
    qos: dict[str, Any] | None = None
    namespace: str | None = None
    decoder: str | None = None  # Custom decoder path


@dataclass(frozen=True, slots=True)
class ActionStreamSpec(StreamSpec):
    """Resolved action stream configuration for runtime use."""

    clamp: tuple[float, float] | None
    safety_behavior: str
    qos: dict[str, Any] | None = None
    namespace: str | None = None
    decoder: str | None = None  # Custom decoder path
    encoder: str | None = None  # Custom encoder path


# =============================================================================
# YAML Loading - Validation Helpers
# =============================================================================


def _validate_enum(value: str, enum_cls: type, field_name: str, context: str) -> str:
    """Validate that a string is a valid enum value."""
    value = str(value).lower().strip()
    valid = {e.value for e in enum_cls}
    if value not in valid:
        raise ContractValidationError(
            f"Invalid {field_name} '{value}' in {context}. Must be one of: {sorted(valid)}"
        )
    return value


def _parse_align(d: dict[str, Any] | None, context: str) -> AlignSpec | None:
    """Parse align configuration."""
    if not d:
        return None
    strategy = _validate_enum(d.get("strategy", "hold"), ResamplePolicy, "strategy", context)
    stamp = _validate_enum(d.get("stamp", "receive"), StampSource, "stamp", context)
    return AlignSpec(strategy=strategy, tol_ms=int(d.get("tol_ms", 0)), stamp=stamp)


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


def _validate_converter_path(path: str | None, context: str) -> str | None:
    """Validate converter path exists at contract load time.

    Path format: "module.path:function_name"

    Args:
        path: Converter path or None
        context: Error context string

    Returns:
        Validated path or None

    Raises:
        ContractValidationError: If path format is invalid or module/function not found
    """
    if path is None:
        return None

    path = str(path).strip()
    if not path:
        return None

    if ":" not in path:
        raise ContractValidationError(
            f"Invalid converter path '{path}' in {context}. "
            f"Expected format: 'module.path:function_name'"
        )

    module_path, func_name = path.rsplit(":", 1)
    if not module_path or not func_name:
        raise ContractValidationError(
            f"Invalid converter path '{path}' in {context}. "
            f"Expected format: 'module.path:function_name'"
        )

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ContractValidationError(
            f"Cannot import converter module '{module_path}' in {context}: {e}"
        ) from e

    if not hasattr(module, func_name):
        raise ContractValidationError(
            f"Function '{func_name}' not found in module '{module_path}' ({context})"
        )

    return path


# =============================================================================
# YAML Loading - Section Parsers
# =============================================================================


def _parse_observation(data: dict[str, Any], idx: int, section: str = "observations") -> ObservationSpec:
    """Parse an observation spec from YAML."""
    ctx = f"{section}[{idx}]"
    _require_fields(data, ["key", "topic", "type"], ctx)

    if not data["topic"]:
        raise ContractValidationError(f"Empty topic in {ctx}")

    return ObservationSpec(
        key=data["key"],
        topic=data["topic"],
        type=data["type"],
        selector=data.get("selector"),
        image=data.get("image"),
        align=_parse_align(data.get("align"), ctx),
        qos=data.get("qos"),
        dtype=_validate_dtype(data.get("dtype"), ctx),
        decoder=_validate_converter_path(data.get("decoder"), f"{ctx}.decoder"),
    )


def _parse_data_spec(data: dict[str, Any], idx: int, section: str) -> ObservationSpec:
    """Parse extended data spec (rewards, signals, etc.) - dtype required."""
    ctx = f"{section}[{idx}]"
    _require_fields(data, ["key", "topic", "type", "dtype"], ctx)

    if not data["topic"]:
        raise ContractValidationError(f"Empty topic in {ctx}")

    return ObservationSpec(
        key=data["key"],
        topic=data["topic"],
        type=data["type"],
        selector=data.get("selector"),
        image=data.get("image"),
        align=_parse_align(data.get("align"), ctx),
        qos=data.get("qos"),
        dtype=_validate_dtype(data["dtype"], ctx, required=True),
        decoder=_validate_converter_path(data.get("decoder"), f"{ctx}.decoder"),
    )


def _parse_action(data: dict[str, Any], idx: int, section: str = "actions") -> ActionSpec:
    """Parse an action spec from YAML."""
    ctx = f"{section}[{idx}]"
    _require_fields(data, ["key", "publish"], ctx)

    pub = data["publish"]
    if not isinstance(pub, dict):
        raise ContractValidationError(f"'publish' must be a mapping in {ctx}")

    _require_fields(pub, ["topic", "type"], f"{ctx}.publish")

    if not pub["topic"]:
        raise ContractValidationError(f"Empty publish.topic in {ctx}")

    safety = _validate_enum(
        data.get("safety_behavior", "zeros"), SafetyBehavior, "safety_behavior", ctx
    )

    return ActionSpec(
        key=data["key"],
        publish_topic=pub["topic"],
        type=pub["type"],
        selector=data.get("selector"),
        from_tensor=data.get("from_tensor"),
        publish_qos=pub.get("qos"),
        publish_strategy=pub.get("strategy"),
        safety_behavior=safety,
        decoder=_validate_converter_path(data.get("decoder"), f"{ctx}.decoder"),
        encoder=_validate_converter_path(data.get("encoder"), f"{ctx}.encoder"),
    )


def _parse_task(data: dict[str, Any], idx: int) -> TaskSpec:
    """Parse a task spec from YAML."""
    ctx = f"tasks[{idx}]"
    _require_fields(data, ["topic", "type"], ctx)

    return TaskSpec(
        key=data.get("key", data["topic"]),
        topic=data["topic"],
        type=data["type"],
        qos=data.get("qos"),
    )


def _parse_adjunct(data: dict[str, Any], idx: int) -> AdjunctSpec:
    """Parse an adjunct topic spec from YAML."""
    ctx = f"adjunct[{idx}]"
    _require_fields(data, ["topic", "type"], ctx)

    buffering_strategy = data.get("buffering_strategy")
    
    # Validate buffering strategy if provided
    if buffering_strategy is not None:
        buffering_strategy = _validate_enum(
            buffering_strategy, BufferingStrategy, "buffering_strategy", ctx
        )
        
        # Validate that buffering strategy is only used with TRANSIENT_LOCAL topics
        qos = data.get("qos", {})
        durability = qos.get("durability", "volatile").lower()
        
        if buffering_strategy != BufferingStrategy.NO_BUFFER.value and durability != "transient_local":
            raise ContractValidationError(
                f"buffering_strategy '{buffering_strategy}' can only be used with "
                f"transient_local durability in {ctx}. Current durability: {durability}"
            )

    return AdjunctSpec(
        topic=data["topic"],
        type=data["type"],
        qos=data.get("qos"),
        buffering_strategy=buffering_strategy,
    )


def _parse_teleop_events(data: dict[str, Any] | None) -> TeleopEventsSpec | None:
    """Parse teleop events section."""
    if not data:
        return None

    ctx = "teleop.events"
    _require_fields(data, ["topic", "type", "mappings"], ctx)

    mappings = data["mappings"]
    if not isinstance(mappings, dict):
        raise ContractValidationError(f"'mappings' must be a mapping in {ctx}")

    return TeleopEventsSpec(
        topic=data["topic"],
        msg_type=data["type"],
        mappings={str(k): str(v) for k, v in mappings.items()},
        qos=data.get("qos"),
    )


def _parse_teleop(data: dict[str, Any] | None) -> TeleopSpec | None:
    """Parse teleop section."""
    if not data:
        return None

    inputs = [
        _parse_observation(it, i, "teleop.inputs")
        for i, it in enumerate(data.get("inputs") or [])
    ]
    events = _parse_teleop_events(data.get("events"))
    feedback = [
        _parse_action(it, i, "teleop.feedback")
        for i, it in enumerate(data.get("feedback") or [])
    ]

    # Override safety_behavior for feedback (always none)
    feedback = [
        ActionSpec(
            key=f.key,
            publish_topic=f.publish_topic,
            type=f.type,
            selector=f.selector,
            from_tensor=f.from_tensor,
            publish_qos=f.publish_qos,
            publish_strategy=f.publish_strategy,
            safety_behavior="none",
            decoder=f.decoder,
            encoder=f.encoder,
        )
        for f in feedback
    ]

    return TeleopSpec(inputs=inputs, events=events, feedback=feedback)


def _parse_reset(data: dict[str, Any] | None) -> ResetSpec | None:
    """Parse reset section."""
    if not data:
        return None

    ctx = "reset"
    mode = _validate_enum(data.get("mode", "manual"), ResetMode, "mode", ctx)

    if mode == "service" and not data.get("service"):
        raise ContractValidationError(f"'service' is required when mode is 'service' in {ctx}")
    if mode == "topic" and not data.get("topic"):
        raise ContractValidationError(f"'topic' is required when mode is 'topic' in {ctx}")

    return ResetSpec(
        mode=mode,
        service=data.get("service"),
        service_type=data.get("service_type", "std_srvs/srv/Trigger"),
        topic=data.get("topic"),
        topic_type=data.get("topic_type", "std_msgs/msg/Empty"),
        reset_time_s=float(data.get("reset_time_s", 5.0)),
    )


def _parse_visualization(data: dict[str, Any] | None) -> VisualizationSpec | None:
    """Parse visualization section."""
    if not data:
        return None

    urdf = data.get("urdf") or {}
    tf = data.get("tf") or {}

    return VisualizationSpec(
        urdf_path=urdf.get("path"),
        urdf_parameter=urdf.get("parameter"),
        tf_topics=tf.get("topics"),
        tf_base_frame=tf.get("base_frame", "world"),
    )


# =============================================================================
# Main Loader
# =============================================================================


def load_contract(path: Path | str) -> Contract:
    """Load and validate a contract YAML file.

    Args:
        path: Path to the contract YAML file.

    Returns:
        Validated Contract dataclass.

    Raises:
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

    # Required fields
    robot_type = data.get("robot_type")
    if not robot_type:
        raise ContractValidationError("robot_type is required")

    fps = int(data.get("fps", 30))
    if fps <= 0:
        raise ContractValidationError(f"fps must be positive, got {fps}")

    # Parse sections
    observations = [_parse_observation(it, i) for i, it in enumerate(data.get("observations") or [])]
    actions = [_parse_action(it, i) for i, it in enumerate(data.get("actions") or [])]
    tasks = [_parse_task(it, i) for i, it in enumerate(data.get("tasks") or [])]

    # Parse adjunct topics (can be single object or list)
    adjunct_raw = data.get("adjunct")
    if adjunct_raw is None:
        adjunct = []
    elif isinstance(adjunct_raw, list):
        adjunct = [_parse_adjunct(it, i) for i, it in enumerate(adjunct_raw)]
    else:
        # Single adjunct object - wrap in list
        adjunct = [_parse_adjunct(adjunct_raw, 0)]

    # Extended data (dtype required)
    rewards = [_parse_data_spec(it, i, "rewards") for i, it in enumerate(data.get("rewards") or [])]
    signals = [_parse_data_spec(it, i, "signals") for i, it in enumerate(data.get("signals") or [])]
    info = [_parse_data_spec(it, i, "info") for i, it in enumerate(data.get("info") or [])]
    comp_data = [
        _parse_data_spec(it, i, "complementary_data")
        for i, it in enumerate(data.get("complementary_data") or [])
    ]

    return Contract(
        robot_type=robot_type,
        fps=fps,
        max_duration_s=float(data.get("max_duration_s", 30.0)),
        observations=observations,
        actions=actions,
        tasks=tasks,
        recording=data.get("recording") or {},
        adjunct=adjunct,
        rewards=rewards,
        signals=signals,
        info=info,
        complementary_data=comp_data,
        teleop=_parse_teleop(data.get("teleop")),
        reset=_parse_reset(data.get("reset")),
        visualization=_parse_visualization(data.get("visualization")),
        timestamp_source=str(data.get("timestamp_source", "receive")).lower(),
    )
