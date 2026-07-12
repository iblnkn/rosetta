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

"""Helpers used only by the lifecycle nodes in rosetta.robots.ros2.nodes.

Node-side concerns: goal/service mutual exclusion (BusyGuard), rcl-legal
action termination (finish_goal), distro feature gates (is_jazzy_or_newer),
QoS introspection for bag topic metadata, and polling waits (wait_until).
Message/QoS/timestamp helpers shared with the wider ros2 layer stay in
rosetta.robots.ros2.ros2_utils.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable

from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    LivelinessPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

# =============================================================================
# Goal / service concurrency
# =============================================================================


class BusyGuard:
    """Accept-time mutual exclusion for one-goal-at-a-time nodes.

    Under a MultiThreadedExecutor with a ReentrantCallbackGroup, two goal
    requests can race a bare ``self._active is not None`` check (the field is
    only set later, in the execute callback). Call :meth:`try_acquire` inside
    the goal/service *accept* callback — an atomic check-and-set — and
    :meth:`release` when the work fully ends (or fails to start).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._busy = False

    def try_acquire(self) -> bool:
        """Atomically claim the guard; False if already claimed."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True
            return True

    def release(self) -> None:
        """Release the guard. Idempotent."""
        with self._lock:
            self._busy = False

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy


def finish_goal(
    goal_handle,
    result,
    *,
    service_cancelled: bool = False,
    success_message: str | None = None,
) -> None:
    """Set an action goal's terminal state using only rcl-legal transitions.

    ``canceled()`` is legal only from CANCELING (``is_cancel_requested``
    True); a service-initiated cancel leaves the goal EXECUTING and must
    ``abort()`` instead. ``succeed()``/``abort()`` are legal from both
    EXECUTING and CANCELING, so a cancel racing in after these checks cannot
    produce an illegal transition. ``result`` must carry ``success`` and
    ``message`` fields (true for all rosetta action results).
    """
    if goal_handle.is_cancel_requested:
        result.success = False
        result.message = "Cancelled"
        goal_handle.canceled()
    elif service_cancelled:
        result.success = False
        result.message = "Cancelled via service"
        goal_handle.abort()
    elif result.success:
        if success_message is not None:
            result.message = success_message
        goal_handle.succeed()
    else:
        goal_handle.abort()


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

    ROS2 distro names are alphabetical (humble < jazzy < kilted < ...),
    so any distro name >= "jazzy" (including "rolling") is considered newer.
    Only known pre-Jazzy distros (humble, iron, galactic, foxy, etc.) return False.

    Returns:
        True if Jazzy/Rolling/Kilted or newer, False otherwise (Humble, etc.)

    """
    distro = detect_ros_distro()
    _PRE_JAZZY = {
        "humble",
        "iron",
        "galactic",
        "foxy",
        "eloquent",
        "dashing",
        "crystal",
        "bouncy",
        "ardent",
    }
    return distro not in _PRE_JAZZY


# =============================================================================
# QoS introspection (bag topic metadata)
# =============================================================================


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
# Timing Utilities
# =============================================================================


def wait_until(predicate: Callable[[], bool], timeout: float, poll: float = 0.1) -> bool:
    """
    Block until ``predicate()`` is true or ``timeout`` seconds elapse.

    Polls at ``poll``-second intervals. Used by lifecycle ``on_deactivate``
    callbacks to wait briefly for in-progress work to wind down. Returns True
    if the predicate became true, False if it timed out.
    """
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(poll)
    return predicate()
