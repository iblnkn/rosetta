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

"""Timelines the ros2 robot interface provides, and per-message extraction.

Data arriving on a channel can carry several timestamps at once — the
receive time, a header stamp, conceivably a publish time or others. The
robot interface (here) is responsible for producing those timelines under
names; align only *selects* one by name (`align.timeline` in the contract).
A new timeline is one new entry in :data:`TIMELINES`, nothing else —
:func:`provided_timelines` and extraction both derive from it, so the
attested set and the extractors cannot drift apart.

Imports without ROS (rosidl is a call-time dependency of
:func:`provided_timelines` only): contract loading uses this module as the
ros2 interface's capability surface, and StreamIngest's unit tests exercise
timestamp extraction with plain fakes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

if TYPE_CHECKING:
    from rosetta.contract.specs import StreamSpec


def stamp_from_header_ns(msg) -> int | None:
    """Nanosecond timestamp from a std_msgs Header stamp, or None when unset.

    rosidl zero-initializes stamps, so (0, 0) is indistinguishable from a
    never-filled header and is treated as unset — at the cost of dropping a
    message legitimately stamped at exactly t=0.
    """
    st = msg.header.stamp
    if st.sec == 0 and st.nanosec == 0:
        return None
    return st.sec * 1_000_000_000 + st.nanosec


class Timeline(NamedTuple):
    """One named timeline: whether a message class carries it, and how to read it."""

    provides: Callable[[type], bool]
    # extract returns None when *this message* is missing the timeline
    # (e.g. an unset header stamp) — the caller's signal to drop it.
    extract: Callable[[Any, int], int | None]


TIMELINES: dict[str, Timeline] = {
    "receive": Timeline(
        provides=lambda msg_cls: True,
        extract=lambda msg, receive_ns: receive_ns,
    ),
    "header": Timeline(
        # Attest by field *type*, not name: a message with e.g. `uint32 header`
        # must not validate a timeline its extractor can never produce.
        provides=lambda msg_cls: msg_cls.get_fields_and_field_types().get("header") == "std_msgs/Header",
        extract=lambda msg, receive_ns: stamp_from_header_ns(msg),
    ),
}


@lru_cache(maxsize=None)
def provided_timelines(msg_type: str) -> frozenset[str]:
    """
    Timelines a ros2 channel of ``msg_type`` provides, by name.

    Every channel provides ``receive``; a message type carrying a std_msgs
    Header also provides ``header``. Contract loading validates
    ``align.timeline`` against this set. Cached: message types are immutable
    within a process, and contract loading asks once per source.

    Raises
    ------
        ValueError: If ``msg_type`` does not name an importable message type.

    """
    from rosidl_runtime_py.utilities import get_message

    try:
        msg_cls = get_message(msg_type)
    except (AttributeError, ModuleNotFoundError, ValueError) as e:
        raise ValueError(f"Unknown message type '{msg_type}': {e}") from e

    return frozenset(name for name, timeline in TIMELINES.items() if timeline.provides(msg_cls))


def get_message_timestamp_ns(msg, spec: "StreamSpec", receive_ns: int) -> int | None:
    """
    Extract the timestamp of ``spec``'s chosen timeline from a message.

    Args:
    ----
        msg: ROS message
        spec: Stream spec; ``spec.source.align.timeline`` names the timeline
        receive_ns: When the message arrived (node clock live, bag time offline)

    Returns:
    -------
        Timestamp in nanoseconds, or None when *this message* does not carry
        the named timeline (e.g. an uninitialized header stamp). There is no
        silent fallback — a missing timeline is the caller's signal to drop.

    Raises:
    ------
        KeyError: If the timeline name is not one this interface produces.
            Contract loading validates ``align.timeline`` against
            :func:`provided_timelines`, so this is a programming error, not
            a data condition — fail loudly rather than dropping forever.

    """
    return TIMELINES[spec.source.align.timeline].extract(msg, receive_ns)
