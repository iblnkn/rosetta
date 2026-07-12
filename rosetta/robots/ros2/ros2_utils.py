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

"""Shared ROS2 utilities for the bridge, codecs, and adapters.

QoS profiles from contract dicts, lifecycle state labels, dotted message
field access (decoders/encoders/teleop), and message timestamp extraction.
Node-only helpers (BusyGuard, finish_goal, distro gates, QoS introspection,
wait_until) live in rosetta.robots.ros2.nodes.node_utils.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

if TYPE_CHECKING:
    from rosetta.contract.specs import StreamSpec


# =============================================================================
# QoS Utilities
# =============================================================================


def qos_profile_from_dict(d: dict[str, Any] | None) -> QoSProfile | None:
    """
    Convert a dictionary to a ROS QoS profile.

    Supported keys:
    - reliability: "reliable" (default) or "best_effort"
    - history: "keep_last" (default) or "keep_all"
    - durability: "volatile" (default) or "transient_local"
    - depth: int (default 10)
    """
    if not d:
        return None

    rel = str(d.get("reliability", "reliable")).lower()
    hist = str(d.get("history", "keep_last")).lower()
    dur = str(d.get("durability", "volatile")).lower()
    depth = int(d.get("depth", 10))

    reliability = ReliabilityPolicy.BEST_EFFORT if rel == "best_effort" else ReliabilityPolicy.RELIABLE
    history = HistoryPolicy.KEEP_ALL if hist == "keep_all" else HistoryPolicy.KEEP_LAST
    durability = DurabilityPolicy.TRANSIENT_LOCAL if dur == "transient_local" else DurabilityPolicy.VOLATILE
    return QoSProfile(
        reliability=reliability,
        history=history,
        depth=depth,
        durability=durability,
    )


# =============================================================================
# Lifecycle state access
# =============================================================================

LIFECYCLE_CONFIGURED_LABELS = frozenset({"inactive", "active", "activating", "deactivating"})


def lifecycle_state_label(node) -> str:
    """Current lifecycle state label of a rclpy LifecycleNode ('active', ...).

    Reads rclpy's private ``_state_machine`` because rclpy exposes no public
    synchronous state accessor (the public route is the GetState service,
    which needs an executor round-trip). This helper is the single sanctioned
    reach-in, shared by TopicBridge and the teleop node — never read
    ``_state_machine`` anywhere else.
    """
    return node._state_machine.current_state[1]


# =============================================================================
# Dotted Attribute Access
# =============================================================================
#
# Resolve a contract selector string (e.g. "position.elbow") against a ROS
# message. The dotted-selector *syntax* is the framework-agnostic contract
# convention (see rosetta.contract.schema); this is the ROS *interpretation* of
# it -- plain attribute walking (parallel-array messages like JointState have
# dedicated codecs instead). A non-ROS binding (protobuf, dict-backed
# dataset, ...) would resolve the same selector against its own message
# structure, so this lives in the ros2 adapter, not in core.


def dot_get(obj, path: str):
    """
    Resolve a dotted attribute path on a ROS message.

    Example:
    -------
        dot_get(msg, "linear.x") -> msg.linear.x

    Parallel-array messages (JointState, JointTrajectory, MultiDOF) have
    dedicated codecs and never route through here.

    """
    cur = obj
    for p in path.split("."):
        cur = getattr(cur, p)
    return cur


def dot_set(obj, path: str, value: float) -> None:
    """
    Set a dotted attribute on a ROS message.

    Example:
    -------
        dot_set(msg, "linear.x", 2.0) -> msg.linear.x = 2.0

    Parallel-array messages (JointState, JointTrajectory, MultiDOF) have
    dedicated codecs and never route through here.

    """
    parts = path.split(".")
    cur = obj
    for p in parts[:-1]:
        cur = getattr(cur, p)
    setattr(cur, parts[-1], float(value))


# =============================================================================
# Timestamp Utilities
# =============================================================================


def stamp_from_header_ns(msg) -> int | None:
    """
    Extract nanosecond timestamp from a ROS message header.

    Returns
    -------
        Positive integer nanoseconds, or None if unavailable/zero.

    """
    try:
        st = msg.header.stamp
    except AttributeError:
        return None

    try:
        sec = int(st.sec)
        nsec = int(st.nanosec)
    except (TypeError, ValueError, AttributeError):
        return None

    # Accept timestamps >= 0 (simulation starts at time 0)
    # Only reject if both sec and nanosec are 0 (uninitialized)
    if sec == 0 and nsec == 0:
        return None

    return sec * 1_000_000_000 + nsec


# =============================================================================
# Timelines
# =============================================================================
#
# Data arriving on a channel can carry several timestamps at once — the
# receive time, a header stamp, conceivably a publish time or others. The
# robot interface (here) is responsible for producing those timelines under
# names; align only *selects* one by name (`align.timeline` in the contract).
# A new timeline is a new entry in these two structures, nothing else.

# timeline name -> extractor(msg, receive_ns) -> int | None (None = the
# message is missing this timeline, e.g. an uninitialized header stamp).
TIMELINE_EXTRACTORS: dict[str, Any] = {
    "receive": lambda msg, receive_ns: receive_ns,
    "header": lambda msg, receive_ns: stamp_from_header_ns(msg),
}


def provided_timelines(msg_type: str) -> set[str]:
    """
    Timelines a ros2 channel of ``msg_type`` provides, by name.

    Every channel provides ``receive``; a message type carrying a std_msgs
    Header also provides ``header``. Contract loading validates
    ``align.timeline`` against this set.

    Raises
    ------
        ValueError: If ``msg_type`` does not name an importable message type.

    """
    from rosidl_runtime_py.utilities import get_message

    try:
        msg_cls = get_message(msg_type)
    except (AttributeError, ModuleNotFoundError, ValueError) as e:
        raise ValueError(f"Unknown message type '{msg_type}': {e}") from e

    timelines = {"receive"}
    if "header" in msg_cls.get_fields_and_field_types():
        timelines.add("header")
    return timelines


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
        Timestamp in nanoseconds, or None when the message does not carry
        the named timeline (e.g. an uninitialized header stamp). There is no
        silent fallback — a missing timeline is the caller's signal to drop.

    """
    extractor = TIMELINE_EXTRACTORS.get(spec.source.align.timeline)
    if extractor is None:
        return None
    return extractor(msg, receive_ns)
