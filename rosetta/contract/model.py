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
The contract document model: enums and dataclasses.

Pure data — no YAML, no numpy, no parsing. :mod:`.schema` parses and
validates YAML into these types (and re-exports them, so external code
imports from schema); :mod:`.specs` projects a built document into runtime
stream specs. Like :mod:`.errors`, this is a leaf every layer can import
without cycles.

A contract binds channels to frame keys. Every frame-clock entry reads::

    channel (provides) -> align (chooses a timeline; mandatory) -> select
    -> apply -> the mapping key

The dataclasses mirror that shape exactly: a :class:`Channel` is the
robot-interface half of an entry (topic/type/qos speak the interface's
dialect), an :class:`Align` chooses a timeline by name, a :class:`Source` is
one channel+align+select+apply pipeline, and a :class:`FrameEntry` is a frame
key with one or more ordered sources (several sources = concatenation for
observations, splitting for actions).

All types are frozen with slots. Parsers build fully-immutable instances
(tuples for sequences, copied dicts for qos); tests constructing them directly
may pass lists — the dataclasses don't coerce, and every consumer only
iterates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable

# =============================================================================
# Enums
# =============================================================================


class ResamplePolicy(StrEnum):
    """Alignment strategy: how a stream's samples map onto frame ticks."""

    HOLD = "hold"  # Carry forward last value
    ASOF = "asof"  # Last value within tolerance window
    DROP = "drop"  # Only value if arrived within step window


class SafetyBehavior(StrEnum):
    """Safety behavior when policy fails to produce actions.

    ``ZEROS`` means action-space (dataset-space) zeros: the zero vector runs
    through ``encode_value``'s inverse operator pipeline like any command, so
    a declared ``clamp`` maps it into the safe wire range. The contract
    author puts the safe range in ``apply``; the safety path deliberately
    honors it rather than bypassing to raw wire zeros.
    """

    NONE = "none"  # No safety action (default)
    ZEROS = "zeros"  # Send zero commands (action-space; see class docstring)
    HOLD = "hold"  # Hold last action


class FieldKind(StrEnum):
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

    Which fields a section may declare is enforced at parse, so e.g. an
    observation's ``safety`` is always ``NONE`` by construction.
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
    """How a source's samples land on the clock defined by the contract's ``fps``.

    ``timeline`` is an open name selecting one of the timelines the channel
    provides (the robot interface produces them; align only chooses).
    Mandatory on every observation/action entry — there are no defaults.
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


#: The teleop event vocabulary: every name the HIL manager's edge dispatch
#: acts on. ``teleop.events.select`` keys are validated against this set at
#: load, so a typo'd event is a contract error, not a button that silently
#: does nothing.
TELEOP_EVENT_NAMES = frozenset(
    {
        "is_intervention",  # level: mux teleop/policy while held
        "start_episode",  # press: start an episode (fire-and-forget)
        "success",  # press: latch positive reward override
        "failure",  # press: latch negative reward override
        "end_success",  # press: end episode with positive reward
        "end_failure",  # press: end episode with negative reward
    }
)


@dataclass(frozen=True, slots=True)
class TeleopEventMap:
    """Teleoperator event button mappings for HIL-SERL. No align — events are edges."""

    channel: Channel
    select: dict[str, str]  # event_name -> field path (dict form of select)


@dataclass(frozen=True, slots=True)
class TeleopInputSource:
    """One teleop input source: a channel -> align -> select -> apply pipeline, decoded like an observation.

    ``target`` is the topic of the action-section Source it drives, validated
    at load time against every declared action channel topic, so a typo'd
    target is a load-time error, not a message silently going nowhere.
    """

    source: Source
    target: str


@dataclass(frozen=True, slots=True)
class TeleopFeedbackSource:
    """One teleop feedback source: a channel -> align -> select -> apply pipeline, encoded like an action.

    ``origin`` is the topic of the observation-section Source whose decoded
    value is forwarded here, validated at load time against every declared
    observation channel topic.
    """

    source: Source
    origin: str


@dataclass(frozen=True, slots=True)
class Teleop:
    """Teleoperator role sections: input / events / feedback.

    ``input``/``feedback`` are tuples, not single entries: each source
    independently names the action/observation topic it drives via
    ``target``/``origin``, so a contract can teleop just one action, several,
    or none, without touching the actions/observations sections themselves.
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
    """Topic -> distinct entry keys whose sources include it.

    The single home for the "topic owned by exactly one entry" rule: schema
    resolves teleop ``target``/``origin`` references against it at parse, and
    specs resolves the synthesized teleop recording key from it. A topic owned
    by several entries makes those resolutions ambiguous and is rejected by
    the callers; sharing a topic across entries stays legal everywhere else.
    """
    owners: dict[str, list[str]] = {}
    for entry in entries:
        for src in entry.sources:
            keys = owners.setdefault(src.channel.topic, [])
            if entry.key not in keys:
                keys.append(entry.key)
    return owners


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
