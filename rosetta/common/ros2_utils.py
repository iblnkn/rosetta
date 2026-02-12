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
ROS2 utilities: QoS profiles, message field access, timestamp helpers, and distribution compatibility.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    LivelinessPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

if TYPE_CHECKING:
    from .contract import ObservationStreamSpec


# =============================================================================
# QoS Utilities
# =============================================================================


def qos_profile_from_dict(d: dict[str, Any] | None) -> QoSProfile | None:
    """Convert a dictionary to a ROS QoS profile.

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

    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT if rel == "best_effort" else ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_ALL if hist == "keep_all" else HistoryPolicy.KEEP_LAST,
        depth=depth,
        durability=DurabilityPolicy.TRANSIENT_LOCAL if dur == "transient_local" else DurabilityPolicy.VOLATILE,
    )


# =============================================================================
# ROS2 Distribution Compatibility
# =============================================================================


def detect_ros_distro() -> str:
    """
    Detect ROS2 distribution from environment.
    
    Returns:
        Distribution name in lowercase (e.g., 'humble', 'jazzy', 'rolling').
        Defaults to 'humble' if ROS_DISTRO is not set.
    """
    return os.environ.get("ROS_DISTRO", "humble").lower()


def is_jazzy_or_newer() -> bool:
    """
    Check if running on Jazzy, Rolling, or newer distributions.
    
    Jazzy introduced several API changes:
    - rosbag2_py uses QoS objects instead of YAML strings in TopicMetadata
    - LifecycleNode supports enable_logger_service parameter
    
    Returns:
        True if Jazzy/Rolling or newer, False otherwise (Humble, etc.)
    """
    distro = detect_ros_distro()
    return distro in ("jazzy", "rolling")


def extract_qos_numeric_values(q: QoSProfile | int) -> dict[str, int]:
    """
    Extract numeric RMW QoS policy values from QoSProfile.
    
    Uses rclpy.qos enums and extracts their .value attributes to get
    the underlying RMW numeric values. This approach works consistently
    across Humble and Jazzy.
    
    Args:
        q: Either a QoSProfile object or an integer depth value
    
    Returns:
        dict with keys: depth, history, reliability, durability, liveliness
        All values are integers matching RMW QoS policy constants.
    """
    if isinstance(q, int):
        # Just depth provided, use common defaults
        return {
            "depth": q,
            "history": HistoryPolicy.KEEP_LAST.value,
            "reliability": ReliabilityPolicy.RELIABLE.value,
            "durability": DurabilityPolicy.VOLATILE.value,
            "liveliness": LivelinessPolicy.AUTOMATIC.value,
        }
    
    # Extract depth directly (not an enum)
    depth = int(getattr(q, "depth", 10) or 10)
    
    # Extract enum .value attributes with fallback defaults
    # Using the rclpy enums ensures we get the correct RMW numeric values
    history = getattr(q.history, "value", HistoryPolicy.KEEP_LAST.value)
    reliability = getattr(q.reliability, "value", ReliabilityPolicy.RELIABLE.value)
    durability = getattr(q.durability, "value", DurabilityPolicy.VOLATILE.value)
    liveliness = getattr(q.liveliness, "value", LivelinessPolicy.AUTOMATIC.value)
    
    return {
        "depth": depth,
        "history": history,
        "reliability": reliability,
        "durability": durability,
        "liveliness": liveliness,
    }


def is_transient_local(qos: QoSProfile | int) -> bool:
    """
    Check if QoS profile has TRANSIENT_LOCAL durability.
    
    Args:
        qos: Either a QoSProfile object or an integer depth value
        
    Returns:
        True if QoS uses TRANSIENT_LOCAL durability, False otherwise
    """
    if isinstance(qos, int):
        return False
    
    try:
        return qos.durability == DurabilityPolicy.TRANSIENT_LOCAL
    except Exception:
        return False


def get_qos_depth(qos: QoSProfile | int) -> int:
    """
    Extract history depth from QoS profile.
    
    Args:
        qos: Either a QoSProfile object or an integer depth value
        
    Returns:
        History depth as integer
    """
    if isinstance(qos, int):
        return qos
    return int(getattr(qos, "depth", 10) or 10)


# =============================================================================
# Dotted Attribute Access (for ROS messages)
# =============================================================================


def dot_get(obj, path: str):
    """Resolve a dotted attribute path on a ROS message.

    Supports JointState-style pattern: "<field>.<joint_name>".

    Example:
        dot_get(msg, "position.elbow") -> msg.position[msg.name.index("elbow")]
        dot_get(msg, "linear.x") -> msg.linear.x
    """
    parts = path.split(".")

    # JointState-like: "field.joint_name" -> field[name.index(joint_name)]
    if len(parts) == 2 and hasattr(obj, "name") and hasattr(obj, parts[0]):
        field, key = parts
        idx = list(obj.name).index(key)
        return getattr(obj, field)[idx]

    # Generic nested getattr
    cur = obj
    for p in parts:
        cur = getattr(cur, p)
    return cur


def dot_set(obj, path: str, value: float) -> None:
    """Set a dotted attribute on a ROS message.

    Supports JointState-style pattern: "<field>.<joint_name>".

    Example:
        dot_set(msg, "position.elbow", 1.5) -> msg.position[index] = 1.5
        dot_set(msg, "linear.x", 2.0) -> msg.linear.x = 2.0
    """
    parts = path.split(".")

    # JointState-like: "field.joint_name" -> field[name.index(joint_name)] = value
    if len(parts) == 2 and hasattr(obj, "name") and hasattr(obj, parts[0]):
        field, key = parts
        arr = getattr(obj, field)
        if isinstance(arr, (list, tuple)) and key in list(obj.name):
            idx = list(obj.name).index(key)
            arr[idx] = float(value)
            return

    # Generic nested setattr
    cur = obj
    for p in parts[:-1]:
        cur = getattr(cur, p)
    setattr(cur, parts[-1], float(value))


# =============================================================================
# Timestamp Utilities
# =============================================================================


def stamp_from_header_ns(msg) -> int | None:
    """Extract nanosecond timestamp from a ROS message header.

    Returns:
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


def get_message_timestamp_ns(msg, spec: "ObservationStreamSpec", fallback_ns: int) -> tuple[int, bool]:
    """Extract timestamp from message based on spec configuration.

    Args:
        msg: ROS message with optional header
        spec: Stream spec with stamp_src ("header" or "receive")
        fallback_ns: Fallback timestamp if header unavailable

    Returns:
        Tuple of (timestamp_ns, used_fallback).
        used_fallback is True if header was unavailable/invalid.
    """
    if spec.stamp_src == "header":
        ts_ns = stamp_from_header_ns(msg)
        if ts_ns is not None:
            return (ts_ns, False)  # Successfully extracted from header
    return (fallback_ns, True)  # Used fallback (receive time)
