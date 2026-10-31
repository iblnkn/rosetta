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

"""Ros2Interface.connect: the live robot a framework opens, against a real ROS graph."""

import time
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import JointState, Joy

from rosetta.contract.schema import load_contract
from rosetta.contract.specs import iter_action_specs, iter_observation_specs
from rosetta.frames.protocols import Connection, RobotClock
from rosetta_ros.interface import Ros2Interface
from rosetta_ros.live import read_events

CONTRACT = """\
robot_type: connection_test
robot_interface: ros2
fps: 20
observations:
  observation.state:
    channel: {topic: /connection_test/joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.a, position.b]
actions:
  action:
    channel: {topic: /connection_test/cmd, type: std_msgs/msg/Float64MultiArray}
    align: {strategy: hold, timeline: receive}
    select: [a, b]
teleop:
  events:
    channel: {topic: /connection_test/joy, type: sensor_msgs/msg/Joy}
    select: {go: buttons.0, stop: buttons.1}
"""


@pytest.fixture
def contract(tmp_path):
    path = tmp_path / "contract.yaml"
    path.write_text(CONTRACT)
    return load_contract(path)


def _connect(contract, use_sim_time=False):
    return Ros2Interface().connect(
        observations=list(iter_observation_specs(contract)),
        actions=list(iter_action_specs(contract)),
        fps=contract.fps,
        name="connection_test",
        events=contract.teleop.events,
        use_sim_time=use_sim_time,
    )


@pytest.fixture
def connection(contract):
    conn = _connect(contract)
    yield conn
    conn.close()


@pytest.fixture
def publisher_node():
    node = rclpy.create_node("connection_test_pub")
    yield node
    node.destroy_node()


def _publish_until(node, msg_type, topic, msg, done, timeout_s=5.0):
    pub = node.create_publisher(msg_type, topic, 10)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pub.publish(msg)
        if done():
            return True
        time.sleep(0.05)
    return False


def test_connection_is_a_connection(connection):
    assert isinstance(connection, Connection)
    assert isinstance(connection.clock, RobotClock)
    assert not connection.clock.simulated


def test_observations_sample_and_missing_keys_are_left_out_on_request(connection, publisher_node):
    assert connection.sample_frame(fill_missing=False) == {}
    assert connection.sample_frame()["observation.state"].tolist() == [0.0, 0.0]

    msg = JointState(name=["a", "b"], position=[0.25, -0.5])
    assert _publish_until(
        publisher_node, JointState, "/connection_test/joint_states", msg, lambda: connection.warmed_up
    )
    np.testing.assert_allclose(connection.sample_frame(fill_missing=False)["observation.state"], [0.25, -0.5])


def test_events_report_the_latest_value_per_contract_name(connection, publisher_node):
    assert connection.events() == {}
    msg = Joy(buttons=[1, 0])
    assert _publish_until(publisher_node, Joy, "/connection_test/joy", msg, lambda: connection.events() != {})
    assert connection.events() == {"go": True, "stop": False}


def test_read_events_keeps_the_last_value_on_a_malformed_field():
    select = {"go": "buttons.0", "stop": "buttons.1"}
    state = read_events({}, SimpleNamespace(buttons=[1, 1]), select)
    assert read_events(state, SimpleNamespace(buttons=[0]), select) == {"go": False, "stop": True}


def test_use_sim_time_runs_the_connection_on_sim_time(contract):
    conn = _connect(contract, use_sim_time=True)
    try:
        assert conn.clock.simulated
    finally:
        conn.close()


def test_close_is_idempotent(contract):
    conn = _connect(contract)
    conn.close()
    conn.close()


def test_close_stops_the_host_when_a_transition_raises(contract, monkeypatch):
    conn = _connect(contract)
    node = conn._host.node

    def refuse():
        raise RuntimeError("transition refused")

    monkeypatch.setattr(node, "trigger_deactivate", refuse)
    with pytest.raises(RuntimeError, match="transition refused"):
        conn.close()
    assert conn._host.node is None
