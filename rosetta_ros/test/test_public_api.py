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

"""rosetta_ros's public Python API, the ROS 2 node toolkit: the modules and names other packages may import.

Each public module declares ``__all__``; everything else in rosetta_ros is internal.
This pins the exact names, so adding, renaming or removing one is a deliberate
change to this file rather than a side effect of a refactor.
"""

import importlib

import pytest

PUBLIC_API = {
    "rosetta_ros.child_goals": {"cancel_goal", "send_goal"},
    "rosetta_ros.field_access": {"dot_get", "dot_set", "resolve_indexed"},
    "rosetta_ros.launch_utils": {"autostart_handlers", "typed_config", "yaml_params"},
    "rosetta_ros.node_host": {"NodeHost"},
    "rosetta_ros.node_utils": {
        "RESULT_QOS",
        "STATUS_QOS",
        "TERMINATION_CANCELLED",
        "TERMINATION_COMPLETED",
        "TERMINATION_ERROR",
        "TERMINATION_NODE_DEACTIVATED",
        "TERMINATION_STOPPED",
        "TERMINATION_TIMEOUT",
        "copy_fields",
        "finish_goal",
        "positive_rate_descriptor",
        "spin_lifecycle_node",
        "wait_until",
    },
    "rosetta_ros.rclpy_utils": {"lifecycle_state_label", "qos_profile_from_dict", "require_transition_success"},
    "rosetta_ros.rosetta_lifecycle_node": {"BridgeLifecycleNode", "RosettaLifecycleNode"},
    "rosetta_ros.runner_host": {"RosParameterHost"},
    "rosetta_ros.topic_bridge": {"TopicBridge"},
}


@pytest.mark.parametrize("module_name", sorted(PUBLIC_API))
def test_public_module_exports_exactly_the_pinned_names(module_name):
    module = importlib.import_module(module_name)
    assert set(module.__all__) == PUBLIC_API[module_name]


@pytest.mark.parametrize("module_name", sorted(PUBLIC_API))
def test_every_exported_name_exists(module_name):
    module = importlib.import_module(module_name)
    assert [name for name in module.__all__ if not hasattr(module, name)] == []
