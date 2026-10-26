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

"""RosParameterHost: a runner's settings are the hosting node's parameters."""

import pytest
from rclpy.node import Node
from rclpy.parameter import Parameter

from rosetta_ros.runner_host import RosParameterHost


@pytest.fixture
def node():
    n = Node("runner_host_test", parameter_overrides=[Parameter("from_launch", value=7)])
    yield n
    n.destroy_node()


def test_declare_then_get_reads_the_default(node):
    host = RosParameterHost(node)
    host.declare("steps", 50, "Steps per chunk")
    assert host.get("steps") == 50


def test_a_launch_override_wins_over_the_default(node):
    host = RosParameterHost(node)
    host.declare("from_launch", 1)
    assert host.get("from_launch") == 7


def test_declaring_an_existing_name_is_a_no_op(node):
    """Hosts declare shared settings first, and setup() may run again after cleanup."""
    host = RosParameterHost(node)
    host.declare("steps", 50)
    host.declare("steps", 99)
    assert host.get("steps") == 50


def test_read_only_reaches_the_parameter(node):
    RosParameterHost(node).declare("device", "cpu", read_only=True)
    (result,) = node.set_parameters([Parameter("device", value="cuda")])
    assert not result.successful
    assert node.get_parameter("device").value == "cpu"


def test_logger_is_the_nodes(node):
    assert RosParameterHost(node).logger.name == node.get_logger().name
