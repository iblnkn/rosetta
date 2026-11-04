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

import logging
import time
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import Float64MultiArray

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
    channel: {topic: /connection_test/cmd, type: std_msgs/msg/Float64MultiArray, safety: zeros}
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


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")  # the crash is the point
def test_a_connection_whose_spin_thread_died_raises_instead_of_serving_stale_frames(contract):
    conn = _connect(contract)
    node = conn._host.node
    node.create_timer(0.01, lambda: (_ for _ in ()).throw(RuntimeError("callback boom")))
    conn._host._spin_thread.join(timeout=5.0)

    for call in (
        conn.sample_frame,
        lambda: conn.publish_frame({"action": np.zeros(2)}),
        conn.events,
        conn.clock.now_ns,
        lambda: conn.warmed_up,
    ):
        with pytest.raises(RuntimeError, match="spin thread died"):
            call()
    conn.close()


def test_a_connection_whose_context_was_shut_down_raises_instead_of_serving_stale_frames(contract, caplog, monkeypatch):
    """rclpy.shutdown on the context, or rclpy's signal handler where a process installs it, outside close()."""
    # launch_testing's pytest plugin installs launch's Logger class, which stops
    # every new logger propagating to the root, where caplog listens.
    monkeypatch.setattr(logging.getLogger("rosetta_ros.live"), "propagate", True)
    conn = _connect(contract)
    rclpy.shutdown(context=conn._host._context)
    conn._host._spin_thread.join(timeout=5.0)
    assert not conn._host._spin_thread.is_alive()
    with pytest.raises(RuntimeError, match="context was shut down"):
        conn.sample_frame()
    with pytest.raises(RuntimeError, match="context was shut down"):
        conn.clock.now_ns()
    with caplog.at_level(logging.WARNING, logger="rosetta_ros.live"):
        conn.close()
    assert "the safety action could not be sent" in caplog.text
    assert conn._host.node is None


def test_a_context_shut_down_during_a_callback_still_stops_the_connection(contract):
    """rclpy's spin() returns without raising when the context goes down while a callback runs."""
    import threading

    conn = _connect(contract)
    inside, release = threading.Event(), threading.Event()

    def blocking_callback():
        inside.set()
        release.wait(5.0)

    conn._host.node.create_timer(0.01, blocking_callback)
    assert inside.wait(5.0)
    rclpy.shutdown(context=conn._host._context)
    release.set()
    conn._host._spin_thread.join(timeout=5.0)
    assert not conn._host._spin_thread.is_alive()
    with pytest.raises(RuntimeError, match="context was shut down"):
        conn.sample_frame()
    conn.close()  # the shutdown destroyed the timer with the node's other entities


def test_a_failed_start_raises_its_own_error_when_cleanup_fails_too(contract, monkeypatch):
    from rosetta_ros import live

    def refuse(result, transition):
        raise RuntimeError(f"{transition} refused")

    close = live.RosConnection.close

    def close_then_fail(self):
        close(self)
        raise RuntimeError("spin thread did not exit")

    monkeypatch.setattr(live, "require_transition_success", refuse)
    monkeypatch.setattr(live.RosConnection, "close", close_then_fail)
    with pytest.raises(RuntimeError, match="configure refused") as excinfo:
        _connect(contract)
    assert any("spin thread did not exit" in note for note in excinfo.value.__notes__)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")  # the crash is the point
def test_close_after_the_spin_thread_died_still_sends_the_safety_action(contract, publisher_node):
    # A callback that raises kills the executor and the bridge's watchdog with
    # it. LeRobot then disconnects, and the robot must still get its safety
    # command rather than keep executing the last one.
    received = []
    publisher_node.create_subscription(Float64MultiArray, "/connection_test/cmd", received.append, 10)
    conn = _connect(contract)
    # A command that arrives proves the publisher and subscription are matched.
    deadline = time.monotonic() + 5.0
    while not received and time.monotonic() < deadline:
        conn.publish_frame({"action": np.array([1.0, 2.0])})
        rclpy.spin_once(publisher_node, timeout_sec=0.05)
    assert received, "the connection's commands never arrived"

    conn._host.node.create_timer(0.01, lambda: (_ for _ in ()).throw(RuntimeError("callback boom")))
    conn._host._spin_thread.join(timeout=5.0)
    conn.close()

    deadline = time.monotonic() + 5.0
    while list(received[-1].data) != [0.0, 0.0] and time.monotonic() < deadline:
        rclpy.spin_once(publisher_node, timeout_sec=0.05)
    assert list(received[-1].data) == [0.0, 0.0]


def test_a_closed_connection_raises(contract):
    conn = _connect(contract)
    clock = conn.clock
    conn.close()
    with pytest.raises(RuntimeError, match="is closed"):
        conn.sample_frame()
    with pytest.raises(RuntimeError, match="is closed"):
        clock.now_ns()
