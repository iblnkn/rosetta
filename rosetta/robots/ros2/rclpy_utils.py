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

"""Shared helpers that need rclpy at import time but no node instance.

QoS profiles from contract dicts and lifecycle state labels, used by the
bridge, the lifecycle nodes, and contract-load qos validation. Importing this
module pulls in rclpy; the pure-Python siblings exist so ROS-less paths can
avoid that — timeline production and message timestamp extraction live in
rosetta.robots.ros2.timelines, dotted message field access in
rosetta.robots.ros2.field_access, and ingest and the codecs use those without
rclpy. Node-only helpers (finish_goal, QoS introspection, wait_until,
spin_lifecycle_node) live in rosetta.robots.ros2.nodes.node_utils.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    LivelinessPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

if TYPE_CHECKING:
    from rclpy.lifecycle import LifecycleNode

# =============================================================================
# QoS Utilities
# =============================================================================


# Contract qos maps straight onto rclpy's QoS policies, and the value strings
# ARE rclpy's own short keys, so parsing delegates to rclpy rather than keeping
# a private vocabulary in sync. Every short-key (enum) QoS policy is accepted.
# reliability and durability gate whether a subscription connects at all
# (best_effort sensors, latched config). history and depth shape the queue.
# liveliness covers publisher-liveliness assertion. The three Duration-valued
# policies (deadline, lifespan, liveliness_lease_duration) are deliberately
# omitted. They have no short-key form, so accepting them would mean inventing
# a seconds-to-Duration convention, which is the custom handling this avoids.
_QOS_ENUMS = {
    "reliability": ReliabilityPolicy,
    "durability": DurabilityPolicy,
    "history": HistoryPolicy,
    "liveliness": LivelinessPolicy,
}
_ALLOWED_QOS_KEYS = _QOS_ENUMS.keys() | {"depth"}


def qos_profile_from_dict(d: dict[str, Any] | None) -> QoSProfile:
    """
    Convert a contract ``qos`` mapping to a ROS QoS profile.

    Accepted keys are the four short-key (enum) QoS policies plus ``depth``:
    ``reliability`` (reliable/best_effort), ``durability``
    (volatile/transient_local), ``history`` (keep_last/keep_all),
    ``liveliness`` (automatic/manual_by_topic), and ``depth`` (int, default
    10). Policy values parse via rclpy's own short-key lookup, so there is no
    private QoS vocabulary to keep in sync. ``None`` and ``{}`` yield
    ``QoSProfile(depth=10)`` (rclpy's default profile).

    Raises
    ------
        ValueError: On an unknown key, an unrecognized policy value, or a
            non-integer depth. Contract loading runs this on every declared
            qos mapping and catches ValueError, so a typo dies at load with
            contract context instead of silently becoming the default policy.
            A mistyped ``reliability`` key would otherwise leave a best_effort
            sensor on the reliable default and receive nothing.

    """
    d = d or {}
    unknown = sorted(d.keys() - _ALLOWED_QOS_KEYS)
    if unknown:
        raise ValueError(f"Unknown qos key(s) {unknown}. Allowed: {sorted(_ALLOWED_QOS_KEYS)}")

    policies = {}
    for key, enum_cls in _QOS_ENUMS.items():
        if key in d:
            value = str(d[key]).strip()
            try:
                # get_from_short_key is case-insensitive and raises KeyError on
                # an unknown value; remap it to the ValueError the loader expects.
                policies[key] = enum_cls.get_from_short_key(value)
            except KeyError:
                raise ValueError(f"Invalid qos {key} '{value}'. Valid: {enum_cls.short_keys()}") from None

    try:
        depth = int(d.get("depth", 10))
    except (TypeError, ValueError):
        raise ValueError(f"Invalid qos depth {d.get('depth')!r}: must be an integer") from None
    return QoSProfile(depth=depth, **policies)


# =============================================================================
# Lifecycle state access
# =============================================================================

LIFECYCLE_CONFIGURED_LABELS = frozenset({"inactive", "active", "activating", "deactivating"})


def lifecycle_state_label(node: LifecycleNode) -> str:
    """Current lifecycle state label of a rclpy LifecycleNode ('active', ...).

    Reads rclpy's private ``_state_machine`` because rclpy exposes no public
    synchronous state accessor (the public route is the GetState service,
    which needs an executor round-trip). This helper is the single sanctioned
    reach-in. Never read ``_state_machine`` anywhere else. Being private, its
    shape can shift across ROS distros, so keeping the access in one place
    limits the blast radius.
    """
    return node._state_machine.current_state[1]


def require_transition_success(result: TransitionCallbackReturn, transition: str) -> None:
    """Raise when a ``trigger_*()`` lifecycle transition reports failure.

    ``trigger_*()`` returns the transition callback's return code rather than
    raising, so a failed transition is otherwise silent. This turns anything
    other than SUCCESS into a hard stop.

    Args:
        result: Return code from ``node.trigger_configure()`` and friends.
        transition: Transition name, used in the error message ('configure').

    Raises:
        RuntimeError: When ``result`` is not SUCCESS. The message points at the
            node log. A deliberate FAILURE return usually logged its own reason
            first. An exception inside the callback is a weaker case: rclpy
            catches it, turns it into ERROR, and drops the traceback, so the
            node log can be empty.

    """
    if result != TransitionCallbackReturn.SUCCESS:
        raise RuntimeError(f"Lifecycle transition '{transition}' failed (see node log)")
