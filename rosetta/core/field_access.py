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
Pure dotted field access on message-like objects.

These helpers walk attributes by a dotted path and have no ROS or LeRobot
dependency -- they work on any object that exposes the addressed attributes.
The two-part ``<field>.<name>`` form additionally supports the JointState-style
lookup (resolve ``name`` via the object's ``name`` list). Adapters (e.g. the
ROS decoders/encoders) use these to read/write message fields.
"""

from __future__ import annotations


def dot_get(obj, path: str):
    """
    Resolve a dotted attribute path on a message-like object.

    Supports JointState-style pattern: "<field>.<joint_name>".

    Example:
    -------
        dot_get(msg, "position.elbow") -> msg.position[msg.name.index("elbow")]
        dot_get(msg, "linear.x") -> msg.linear.x

    """
    parts = path.split('.')

    # JointState-like: "field.joint_name" -> field[name.index(joint_name)]
    if len(parts) == 2 and hasattr(obj, 'name') and hasattr(obj, parts[0]):
        field, key = parts
        idx = list(obj.name).index(key)
        return getattr(obj, field)[idx]

    # Generic nested getattr
    cur = obj
    for p in parts:
        cur = getattr(cur, p)
    return cur


def dot_set(obj, path: str, value: float) -> None:
    """
    Set a dotted attribute on a message-like object.

    Supports JointState-style pattern: "<field>.<joint_name>".

    Example:
    -------
        dot_set(msg, "position.elbow", 1.5) -> msg.position[index] = 1.5
        dot_set(msg, "linear.x", 2.0) -> msg.linear.x = 2.0

    """
    parts = path.split('.')

    # JointState-like: "field.joint_name" -> field[name.index(joint_name)] = value
    if len(parts) == 2 and hasattr(obj, 'name') and hasattr(obj, parts[0]):
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
