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

"""Dotted-selector resolution against ROS messages (get/set).

Resolve a contract selector string (e.g. "position.elbow") against a ROS
message. The dotted-selector *syntax* is the framework-agnostic contract
convention (see rosetta.contract.schema); this is the ROS *interpretation* of
it -- plain attribute walking (parallel-array messages like JointState have
dedicated codecs instead). A non-ROS binding (protobuf, dict-backed
dataset, ...) would resolve the same selector against its own message
structure, so this lives in the ros2 adapter, not in core.

Pure Python by design -- no rclpy/rosidl imports (numpy only). The codec
modules (decoders/encoders) depend on this, and their ROS-less importability
is what keeps contract loading free of a ROS environment.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


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


def resolve_indexed(obj, path: str):
    """
    Resolve a dotted path where numeric segments index into sequences.

    Example:
    -------
        resolve_indexed(msg, "buttons.5") -> msg.buttons[5]

    Unlike :func:`dot_get`, a purely-numeric segment is a sequence index.
    Used by teleop event selectors (Joy buttons/axes); lives here so the
    indexed-selector grammar has one home. Negative indices are rejected for
    the same reason as :func:`parse_joy_selector`.

    """
    cur = obj
    for p in path.split("."):
        if p.lstrip("-").isdigit():
            idx = int(p)
            if idx < 0:
                raise ValueError(f"Selector index must be non-negative, got {idx} in '{path}'")
            cur = cur[idx]
        else:
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


# Field tables for the parallel-array message families. Keys are the accepted
# selector spellings, values the message attribute they map to. Shared by each
# family's decoder and encoder so the selector grammar cannot drift between
# the two (same rationale as parse_joy_selector below).
JOINT_STATE_FIELDS = {"position": "position", "velocity": "velocity", "effort": "effort"}
JOINT_TRAJECTORY_FIELDS = {
    "position": "positions",
    "positions": "positions",
    "velocity": "velocities",
    "velocities": "velocities",
    "acceleration": "accelerations",
    "accelerations": "accelerations",
    "effort": "effort",
}
MULTIDOF_FIELDS = {"values": "values", "values_dot": "values_dot"}


def parse_field_selector(selector: str, fields: dict[str, str], *, default: str, msg_type: str) -> tuple[str, str]:
    """
    Parse a parallel-array selector "<field>.<name>" into ``(attribute, name)``.

    A bare name means ``default`` (e.g. "elbow" -> position). ``fields`` maps
    accepted field spellings to the message attribute they select; an unknown
    field is an error rather than a silently mis-parsed name.
    """
    if "." in selector:
        field, name = selector.split(".", 1)
    else:
        field, name = default, selector

    attr = fields.get(field)
    if attr is None:
        raise ValueError(
            f"Unknown {msg_type} field '{field}' in selector '{selector}'. "
            f"Valid fields: {', '.join(sorted(set(fields)))}"
        )
    return attr, name


def gather_named_fields(
    selectors: Iterable[str],
    msg_names: Iterable[str],
    container,
    fields: dict[str, str],
    *,
    default: str,
    msg_type: str,
) -> np.ndarray:
    """
    Gather selector values from a parallel-array message family (decode side).

    ``msg_names`` is the message's name list; each selected field's array on
    ``container`` aligns 1:1 with it (JointState: the message itself;
    JointTrajectory: the point). Shared by every parallel-array decoder so
    the lookup/bounds semantics cannot drift between them.
    """
    name_to_idx = {name: i for i, name in enumerate(msg_names)}
    out = []
    for selector in selectors:
        attr, name = parse_field_selector(selector, fields, default=default, msg_type=msg_type)
        if name not in name_to_idx:
            raise ValueError(f"{msg_type} name '{name}' not in message. Available: {list(name_to_idx)}")
        idx = name_to_idx[name]
        arr = getattr(container, attr)
        if idx >= len(arr):
            raise ValueError(f"Index {idx} out of range for '{attr}' (len={len(arr)})")
        out.append(float(arr[idx]))
    return np.asarray(out, dtype=np.float64)


def build_field_map(
    selectors: Iterable[str],
    fields: dict[str, str],
    *,
    default: str,
    msg_type: str,
) -> tuple[dict[str, dict[str, int]], list[str]]:
    """
    Map selectors of a parallel-array message to fields (encode side).

    Returns ``(field -> {name -> vector index}, first-occurrence name order)``
    — the scatter plan every parallel-array encoder builds before filling the
    message's name list and per-field arrays.
    """
    field_to_names: dict[str, dict[str, int]] = {}
    name_order: list[str] = []
    seen: set[str] = set()
    for i, path in enumerate(selectors):
        attr, name = parse_field_selector(path, fields, default=default, msg_type=msg_type)
        field_to_names.setdefault(attr, {})[name] = i
        if name not in seen:
            name_order.append(name)
            seen.add(name)
    return field_to_names, name_order


def parse_joy_selector(selector: str) -> tuple[str, int]:
    """
    Parse a Joy selector "<field>.<index>" into ``(field, index)``.

    ``field`` is "axes" or "buttons"; a bare index means "axes". Shared by
    the Joy decoder and encoder so the selector syntax cannot drift between
    the two. Negative indices are rejected: Python's wrap-from-the-end
    semantics would make decode silently read the wrong element while encode
    mis-sizes its output arrays.
    """
    if "." in selector:
        field, idx_str = selector.split(".", 1)
    else:
        field, idx_str = "axes", selector

    try:
        idx = int(idx_str)
    except ValueError:
        raise ValueError(f"Joy selector index must be an integer, got '{idx_str}' in selector '{selector}'") from None
    if idx < 0:
        raise ValueError(f"Joy selector index must be non-negative, got {idx} in selector '{selector}'")
    if field not in ("axes", "buttons"):
        raise ValueError(f"Unknown Joy field '{field}'. Valid fields: axes, buttons")
    return field, idx
