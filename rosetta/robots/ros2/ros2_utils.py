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

QoS profiles from contract dicts and lifecycle state labels. Timeline
production and message timestamp extraction live in
rosetta.robots.ros2.timelines, dotted message field access in
rosetta.robots.ros2.field_access (both pure Python -- ingest and the codecs
need them without rclpy). Node-only helpers (BusyGuard, finish_goal, QoS
introspection, wait_until, spin_lifecycle_node) live in
rosetta.robots.ros2.nodes.node_utils.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

if TYPE_CHECKING:
    from rclpy.lifecycle import LifecycleNode

# =============================================================================
# QoS Utilities
# =============================================================================


_QOS_POLICIES: dict[str, dict[str, Any]] = {
    "reliability": {"reliable": ReliabilityPolicy.RELIABLE, "best_effort": ReliabilityPolicy.BEST_EFFORT},
    "history": {"keep_last": HistoryPolicy.KEEP_LAST, "keep_all": HistoryPolicy.KEEP_ALL},
    "durability": {"volatile": DurabilityPolicy.VOLATILE, "transient_local": DurabilityPolicy.TRANSIENT_LOCAL},
}

_QOS_DEFAULTS = {"reliability": "reliable", "history": "keep_last", "durability": "volatile"}


def _qos_depth(value: Any) -> int:
    # Same integer strictness as the contract loader's _parse_strict_int:
    # integral floats are accepted (YAML writers produce 10.0), everything
    # else that isn't a plain int is an error.
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Invalid qos depth {value!r}: must be a non-negative integer")
    return value


def qos_profile_from_dict(d: dict[str, Any] | None) -> QoSProfile:
    """
    Convert a contract ``qos`` mapping to a ROS QoS profile.

    Supported keys (anything else is an error):
    - reliability: "reliable" (default) or "best_effort"
    - history: "keep_last" (default) or "keep_all"
    - durability: "volatile" (default) or "transient_local"
    - depth: non-negative int (default 10)

    ``None`` and ``{}`` yield the default profile — the same profile rclpy
    builds for the int-depth shorthand, ``QoSProfile(depth=10)``.

    Raises
    ------
        ValueError: On an unknown key, an unrecognized policy value, or a
            bad depth. Contract loading runs this on every declared qos
            mapping, so a typo dies at load with contract context instead
            of silently becoming the default policy.

    """
    d = d or {}
    unknown = sorted(d.keys() - (_QOS_POLICIES.keys() | {"depth"}))
    if unknown:
        raise ValueError(f"Unknown qos key(s) {unknown}. Allowed: {sorted(_QOS_POLICIES.keys() | {'depth'})}")

    policies = {}
    for key, allowed in _QOS_POLICIES.items():
        value = str(d.get(key, _QOS_DEFAULTS[key])).lower().strip()
        if value not in allowed:
            raise ValueError(f"Invalid qos {key} '{value}'. Must be one of: {sorted(allowed)}")
        policies[key] = allowed[value]

    return QoSProfile(depth=_qos_depth(d.get("depth", 10)), **policies)


# =============================================================================
# Lifecycle state access
# =============================================================================

LIFECYCLE_CONFIGURED_LABELS = frozenset({"inactive", "active", "activating", "deactivating"})


def lifecycle_state_label(node: LifecycleNode) -> str:
    """Current lifecycle state label of a rclpy LifecycleNode ('active', ...).

    Reads rclpy's private ``_state_machine`` because rclpy exposes no public
    synchronous state accessor (the public route is the GetState service,
    which needs an executor round-trip). This helper is the single sanctioned
    reach-in, shared by TopicBridge and the teleop node — never read
    ``_state_machine`` anywhere else.
    """
    return node._state_machine.current_state[1]


def require_transition_success(result: TransitionCallbackReturn, transition: str) -> None:
    """Raise when a ``trigger_*()`` lifecycle transition reports failure.

    ``trigger_*()`` returns the transition callback's return code; the
    callback already logged why it failed (rclpy itself drops the
    traceback), so all that is left here is to refuse to continue.
    """
    if result != TransitionCallbackReturn.SUCCESS:
        raise RuntimeError(f"Lifecycle transition '{transition}' failed (see node log)")
