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
The contract document model: enums and dataclasses, no parsing.

:mod:`.schema` parses YAML into these types and re-exports them;
:mod:`.specs` resolves them into runtime stream specs. This module imports
only :mod:`.errors`, so any layer can import it without cycles.

A contract binds channels to frame keys. Every frame-clock entry reads::

    channel (provides) -> align (chooses a timeline; mandatory) -> select
    -> apply -> the mapping key

A :class:`Channel` is the robot-interface half (topic/type/qos), an
:class:`Align` picks a timeline by name, a :class:`Source` is one
channel+align+select+apply pipeline, and a :class:`FrameEntry` is a frame key
with one or more ordered sources (concatenated for observations, split for
actions).

The parser builds tuples and copied dicts. The dataclasses don't coerce, so
tests may pass lists; consumers only iterate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable

from .errors import ContractValidationError


class ResamplePolicy(StrEnum):
    """Alignment strategy: how a stream's samples map onto frame ticks."""

    HOLD = "hold"  # Carry forward last value
    ASOF = "asof"  # Last value within tolerance window
    DROP = "drop"  # Only value if arrived within step window


class SafetyBehavior(StrEnum):
    """What to publish when the policy stops producing actions.

    ``ZEROS`` are action-space zeros: they go through the inverse operator
    pipeline like any command, so a declared ``clamp`` still bounds them.

    ``HOLD`` re-sends the stream's last command. Before the stream has sent
    one (since the bridge was set up, or since its last state reset between
    episodes) there is nothing to hold, so it sends zeros, exactly like
    ``ZEROS``. On a position-controlled arm that zero is a pose the arm will
    move to.
    """

    NONE = "none"  # No safety action (default)
    ZEROS = "zeros"  # Send zero commands (action-space)
    HOLD = "hold"  # Re-send the last command; zeros if none was sent yet


class FieldKind(StrEnum):
    """Value type of a state/action field.

    Tells the VLA frameworks how to normalize or rotate the values. LeRobot
    ignores it. Default is continuous (plain min_max).
    """

    CONTINUOUS = "continuous"  # plain scalar/vector (joints, positions, velocities)
    QUATERNION = "quaternion"  # 4D rotation [x, y, z, w]
    EULER_RPY = "euler_rpy"  # 3D roll/pitch/yaw
    AXIS_ANGLE = "axis_angle"  # 3D rotation vector
    ROTATION_6D = "rotation_6d"  # 6D continuous rotation representation
    BINARY = "binary"  # discrete on/off (e.g. gripper open/close)

    @property
    def dims(self) -> int | None:
        """Required select length for this kind (None = any), checked at contract load."""
        return _FIELD_KIND_DIMS[self]


_FIELD_KIND_DIMS: dict[FieldKind, int | None] = {
    FieldKind.CONTINUOUS: None,
    FieldKind.QUATERNION: 4,
    FieldKind.EULER_RPY: 3,
    FieldKind.AXIS_ANGLE: 3,
    FieldKind.ROTATION_6D: 6,
    FieldKind.BINARY: None,
}


@dataclass(frozen=True, slots=True)
class Channel:
    """One robot-interface channel plus its codec settings.

    topic/type/qos speak the interface's dialect (ros2). Image pixel encoding
    is not declared: it is read from the message at decode time. Which fields
    a section may set is enforced at parse, so e.g. an observation's
    ``safety`` is always ``NONE``.
    """

    topic: str
    type: str
    qos: dict[str, Any] | None = None
    dtype: str | None = None
    decoder: str | None = None  # Custom decoder path: "module.path:function_name"
    encoder: str | None = None  # Custom encoder path (actions/feedback only)
    safety: SafetyBehavior = SafetyBehavior.NONE  # action channels only


@dataclass(frozen=True, slots=True)
class Align:
    """How a source's samples land on the contract's ``fps`` clock.

    ``timeline`` names one of the timelines the robot interface provides for
    the channel's type.
    """

    strategy: ResamplePolicy  # "hold" | "asof" | "drop"
    timeline: str  # open timeline name, e.g. "receive" | "header"
    tolerance_ms: int = 0  # required iff strategy == "asof"


@dataclass(frozen=True, slots=True)
class Source:
    """One channel -> align -> select -> apply pipeline feeding a frame key."""

    channel: Channel
    align: Align
    select: tuple[str, ...] | None = None  # field paths to project (list form in YAML)
    apply: tuple[tuple[str, Any], ...] = ()  # ordered operators: ((name, args), ...)
    kind: FieldKind = FieldKind.CONTINUOUS  # representation for VLA frameworks


@dataclass(frozen=True, slots=True)
class FrameEntry:
    """A frame key and its ordered sources (concatenated for observations, split for actions)."""

    key: str
    sources: tuple[Source, ...]


@dataclass(frozen=True, slots=True)
class Task:
    """Task channel (e.g. prompts). Not sampled on the frame clock, so no align."""

    key: str
    channel: Channel


@dataclass(frozen=True, slots=True)
class TeleopEventMap:
    """Teleoperator event button mappings. No align: events are edges.

    Event names are open: each consumer acts on the names it knows and ignores
    the rest, so one contract can serve several consumers.
    """

    channel: Channel
    select: dict[str, str]  # event_name -> field path (dict form of select)


@dataclass(frozen=True, slots=True)
class TeleopInputSource:
    """A teleop input source, decoded like an observation.

    ``target`` is the topic of the action entry it drives; it must belong to
    exactly one action entry.
    """

    source: Source
    target: str


@dataclass(frozen=True, slots=True)
class TeleopFeedbackSource:
    """A teleop feedback source, encoded like an action.

    ``origin`` is the topic of the observation entry whose decoded value is
    forwarded here; it must belong to exactly one observation entry.
    """

    source: Source
    origin: str


@dataclass(frozen=True, slots=True)
class Teleop:
    """Teleoperator sections: input / events / feedback.

    Each input/feedback source names its own action/observation topic, so a
    contract can teleop any subset of its actions.
    """

    input: tuple[TeleopInputSource, ...] = ()
    events: TeleopEventMap | None = None
    feedback: tuple[TeleopFeedbackSource, ...] = ()


#: Synthesized key prefixes for teleop diagnostic recording columns (see specs.py).
TELEOP_INPUT_KEY = "teleop.input"
TELEOP_FEEDBACK_KEY = "teleop.feedback"


#: The frame-clock sections, in resolution order (YAML key == Contract attribute).
FRAME_SECTIONS = ("observations", "actions", "rewards", "signals", "info", "complementary_data")

#: Extended (record-only) frame sections: never images, dtype mandatory.
EXTENDED_SECTIONS = FRAME_SECTIONS[2:]


def topic_owners(entries: Iterable[FrameEntry]) -> dict[str, list[str]]:
    """Topic -> distinct entry keys whose sources include it."""
    owners: dict[str, list[str]] = {}
    for entry in entries:
        for src in entry.sources:
            keys = owners.setdefault(src.channel.topic, [])
            if entry.key not in keys:
                keys.append(entry.key)
    return owners


def owning_key(entries: Iterable[FrameEntry], topic: str, ref: str) -> str:
    """Key of the one entry whose sources include ``topic``.

    A teleop ``target``/``origin`` must name exactly one entry: its recording
    column is named after that entry, and teleop consumers route by topic.
    Sharing a topic across entries stays legal everywhere else. ``ref`` names
    the reference in errors, e.g. ``"teleop.input[0]: target"``.
    """
    owners = topic_owners(entries)
    keys = owners.get(topic, [])
    if not keys:
        raise ContractValidationError(
            f"{ref} '{topic}' does not match any entry's channel topic. Known topics: {sorted(owners)}"
        )
    if len(keys) > 1:
        raise ContractValidationError(
            f"{ref} '{topic}' belongs to multiple entries ({keys}); a teleop target/origin must belong to exactly one."
        )
    return keys[0]


@dataclass(frozen=True, slots=True)
class Contract:
    """Top-level contract describing a robot's policy I/O surface."""

    robot_type: str
    robot_interface: str
    fps: int
    observations: tuple[FrameEntry, ...]
    actions: tuple[FrameEntry, ...]
    tasks: tuple[Task, ...]
    adjunct: tuple[Channel, ...]
    rewards: tuple[FrameEntry, ...]
    signals: tuple[FrameEntry, ...]
    info: tuple[FrameEntry, ...]
    complementary_data: tuple[FrameEntry, ...]
    teleop: Teleop | None = None

    def frame_entries(self) -> Iterable[tuple[str, FrameEntry]]:
        """Yield ``(section, entry)`` over all frame-clock sections."""
        for section in FRAME_SECTIONS:
            for entry in getattr(self, section):
                yield section, entry
