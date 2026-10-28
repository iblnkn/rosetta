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

from rosetta_ros.rclpy_utils import prefixed_topic

__all__ = [
    "autostart_handlers",
    "contract_output_remaps",
    "typed_config",
    "yaml_params",
]


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


def contract_output_remaps(contract_path: str, prefix: str) -> list[tuple[str, str]]:
    """Remaps that move every topic a policy runner can publish under ``prefix``.

    Those are the contract's action and reward topics: a runner publishes one
    or the other (rewards when ``is_classifier`` is set), and a remap for a
    topic a node never uses has no effect, so both are remapped. Use it to put
    something between the runner and the robot, such as a mux. An empty
    ``prefix`` returns no remaps, and a prefix needs an explicit ``contract_path``.
    """
    if not prefix:
        return []
    if not contract_path:
        # The node can fall back to a model's embedded contract; launch can't
        # read the topics to remap from it.
        raise ValueError("output_prefix needs an explicit contract_path: the topics to remap are read from it")
    from rosetta.contract.schema import load_contract

    contract = load_contract(contract_path)
    topics = dict.fromkeys(
        src.channel.topic for entry in (*contract.actions, *contract.rewards) for src in entry.sources
    )
    return [(topic, prefixed_topic(prefix, topic)) for topic in topics]


def yaml_params(path: str, node_name: str) -> dict:
    """One node's ``ros__parameters`` mapping from a plain params YAML file.

    For launch files that merge parameter dicts in Python (layered defaults,
    one launch including another). Namespaced nodes also need this instead of passing the
    file path. A params file's bare top-level node key only matches the
    root-namespace node name, so the file is silently inert for a namespaced
    node. A dict applies unconditionally.

    Sections keyed ``/**``, ``/**/node_name``, ``node_name`` (or
    ``/node_name``) and ``/<namespace>/node_name`` all apply, merged in that
    order so the more specific key wins. A file with sections, none of which names this node,
    raises: ignoring it would silently run the node on defaults.
    """
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    def rank(key: str) -> int | None:
        if key == "/**":
            return 0
        if key.lstrip("/") == node_name:
            return 2
        if key.startswith("/") and key.rsplit("/", 1)[-1] == node_name:
            return 1 if key == f"/**/{node_name}" else 3
        return None

    matched = sorted((r, key) for key in data if (r := rank(str(key))) is not None)
    if data and not matched:
        raise ValueError(f"{path}: no section applies to node '{node_name}' (sections: {sorted(map(str, data))})")
    params: dict = {}
    for _, key in matched:
        params.update((data[key] or {}).get("ros__parameters") or {})
    return params


def _flag_condition(name: str, context) -> IfCondition:
    """Condition for a boolean launch flag on a deferred event.

    With a context, the flag is resolved to its final value here, at setup
    time. Left as a lazy substitution, the condition is evaluated only when
    the event handler eventually fires, in whatever configuration scope
    exists then — which read the wrong value when the launch file was
    included from a launch XML on Jazzy. Without a context (a declaratively
    built launch file), the lazy substitution is the only option.
    """
    if context is None:
        return IfCondition(EqualsSubstitution(LaunchConfiguration(name), "true"))
    return IfCondition("true" if typed_config(context, name, bool) else "false")


def autostart_handlers(node, context=None) -> list:
    """The configure-on-start / activate-on-inactive chain for a LifecycleNode.

    Gated by the ``configure`` and ``activate`` launch configurations (the
    caller declares those arguments), resolved at setup time when ``context``
    is passed — see ``_flag_condition``. The activate event fires only when
    the node actually reaches INACTIVE: emitting it right after the configure
    request would race the transition and be dropped by the state machine.
    """
    configure_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(node),
            transition_id=Transition.TRANSITION_CONFIGURE,
        ),
        condition=_flag_condition("configure", context),
    )
    activate_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(node),
            transition_id=Transition.TRANSITION_ACTIVATE,
        ),
        condition=_flag_condition("activate", context),
    )
    return [
        RegisterEventHandler(OnProcessStart(target_action=node, on_start=[configure_event])),
        RegisterEventHandler(
            OnStateTransition(target_lifecycle_node=node, goal_state="inactive", entities=[activate_event])
        ),
    ]
