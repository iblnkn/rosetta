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

"""Shared rclpy context for the whole test session, and HilManagerNode fixtures.

rclpy.init() is global-context and process-wide — calling it a second time
before a prior call's rclpy.shutdown() raises, so test modules must not
initialize their own contexts in the shared pytest process. autouse=True so
it applies session-wide without every test file (pytest-native or
unittest.TestCase) needing to request it explicitly; unittest-style tests
can't take fixture parameters.
"""

import os
from types import SimpleNamespace

import pytest
import rclpy
from forwarder_fakes import FakePublisher
from std_msgs.msg import Int8

from rosetta_interfaces.msg import ManageEpisodeResult, ManageEpisodeStatus
from rosetta_ros_hil.hil_manager_node import HilManagerNode

# This package's tests run on their own domain, so a robot or sim on this
# machine, or another package's tests run in parallel by colcon, never shares
# their topics. Set at import, before any test creates a context.
os.environ["ROS_DOMAIN_ID"] = "82"


@pytest.fixture(scope="session", autouse=True)
def rclpy_ctx():
    rclpy.init()
    yield
    rclpy.try_shutdown()


@pytest.fixture
def make_node(rclpy_ctx):
    """Build HilManagerNodes with the ~/status, ~/result and hil_intervention publishers setup() would create.

    No contract is loaded; each node is destroyed after the test.
    """
    nodes = []

    def make(**kwargs):
        node = HilManagerNode(**kwargs)
        node._status_pub = FakePublisher(ManageEpisodeStatus)
        node._result_pub = FakePublisher(ManageEpisodeResult)
        node._intervention_pub = FakePublisher(Int8)
        nodes.append(node)
        return node

    yield make
    for node in nodes:
        node.destroy_node()


@pytest.fixture
def node(make_node):
    return make_node()


@pytest.fixture
def child_prompts(node, monkeypatch):
    """Every child goal (recorder, policy, classifier) is accepted without a server; returns the prompts sent."""
    prompts = []
    monkeypatch.setattr(
        node,
        "_send_child_goal",
        lambda _c, goal, *a, **k: prompts.append(goal.prompt) or SimpleNamespace(accepted=True),
    )
    monkeypatch.setattr(node, "_cancel_child", lambda *a, **k: None)
    monkeypatch.setattr(node, "_stop_recorder", lambda: None)
    return prompts
