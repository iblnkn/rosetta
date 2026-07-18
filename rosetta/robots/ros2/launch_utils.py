# Copyright 2026 Isaac Blankenau
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

"""Shared helpers for rosetta's lifecycle-node launch files.

Imported only by the files under ``launch/``, never by the contract/frames
layers, so launch and rclpy stay out of the contract import graph. The
lifecycle autostart chain lives here once instead of being copied into every
launch file, so a fix or a new gotcha applies to all of them at once.
"""

from __future__ import annotations

import yaml
from launch.actions import EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.events import matches_action
from launch.substitutions import EqualsSubstitution, LaunchConfiguration
from launch.utilities.type_utils import normalize_typed_substitution, perform_typed_substitution
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition


def typed_config(context, name: str, data_type):
    """Resolve a LaunchConfiguration and coerce it to ``data_type``.

    Use this instead of hand-parsing ``value.lower() in ("true", "1", ...)``.
    Launch's coercion accepts the same spellings ``IfCondition`` does and
    raises on unparseable input instead of silently reading it as False. The
    substitution must be normalized before ``perform_typed_substitution``, so
    both calls are required.
    """
    normalized = normalize_typed_substitution(LaunchConfiguration(name), data_type)
    return perform_typed_substitution(context, normalized, data_type)


def yaml_params(path: str, node_name: str) -> dict:
    """One node's ``ros__parameters`` mapping from a plain params YAML file.

    For launch files that merge parameter dicts in Python (hil_launch's
    layered defaults). Namespaced nodes also need this instead of passing the
    file path. A params file's bare top-level node key only matches the
    root-namespace node name, so the file is silently inert for a namespaced
    node. A dict applies unconditionally.
    """
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return (data.get(node_name) or {}).get("ros__parameters") or {}


def autostart_handlers(node) -> list:
    """The configure-on-start / activate-on-inactive chain for a LifecycleNode.

    Gated by the ``configure`` and ``activate`` launch configurations (the
    caller declares those arguments). The activate event fires only when the
    node actually reaches INACTIVE: emitting it right after the configure
    request would race the transition and be dropped by the state machine.
    """
    configure_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(node),
            transition_id=Transition.TRANSITION_CONFIGURE,
        ),
        condition=IfCondition(EqualsSubstitution(LaunchConfiguration("configure"), "true")),
    )
    activate_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(node),
            transition_id=Transition.TRANSITION_ACTIVATE,
        ),
        condition=IfCondition(EqualsSubstitution(LaunchConfiguration("activate"), "true")),
    )
    return [
        RegisterEventHandler(OnProcessStart(target_action=node, on_start=[configure_event])),
        RegisterEventHandler(
            OnStateTransition(target_lifecycle_node=node, goal_state="inactive", entities=[activate_event])
        ),
    ]
