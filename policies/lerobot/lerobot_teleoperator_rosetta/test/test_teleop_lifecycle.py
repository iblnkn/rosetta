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

"""Input-only teleop lifecycle end-to-end (the so_101_teleop.yaml fixture, feedback: []).

An input-only teleop must report connected, return sampled inputs from
get_action(), see the contract's events, and disconnect cleanly enough to reconnect.
"""

import time
from pathlib import Path

import numpy as np
import pytest
import rclpy
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.errors import DeviceAlreadyConnectedError
from sensor_msgs.msg import JointState, Joy

from lerobot_teleoperator_rosetta.config_rosetta_teleop import RosettaTeleopConfig
from lerobot_teleoperator_rosetta.rosetta_teleop import RosettaTeleop

CONTRACT = Path(__file__).resolve().parent / "so_101_teleop.yaml"
INPUT_TOPIC = "/human/leader_arm/joint_states"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_pitch", "wrist_roll", "wrist_yaw"]


@pytest.fixture
def teleop():
    t = RosettaTeleop(RosettaTeleopConfig(id="test_teleop", config_path=str(CONTRACT)))
    yield t
    # Best-effort teardown if a test failed mid-way.
    try:
        if t._node is not None:
            t.disconnect()
    except Exception:
        pass


def test_input_only_teleop_connects(teleop):
    teleop.connect()
    assert teleop.is_connected
    assert teleop.feedback_features == {}  # genuinely input-only


def test_connect_subscribes_to_the_contract_events(teleop, ros):
    teleop.connect()
    pub_node = rclpy.create_node("teleop_events_fixture_pub")
    pub = pub_node.create_publisher(Joy, "/joy", 10)
    try:
        deadline = time.monotonic() + 5.0
        events = teleop.get_teleop_events()
        while time.monotonic() < deadline and not events[TeleopEvents.TERMINATE_EPISODE]:
            pub.publish(Joy(buttons=[0, 0, 1, 0]))  # buttons.2 is end_success in the fixture
            time.sleep(0.05)  # teleop spins on its own thread
            events = teleop.get_teleop_events()
        assert (events[TeleopEvents.SUCCESS], events[TeleopEvents.TERMINATE_EPISODE]) == (True, True)
    finally:
        pub_node.destroy_node()


def test_double_connect_raises(teleop):
    # lerobot device convention (and the robot adapter's behavior).
    teleop.connect()
    with pytest.raises(DeviceAlreadyConnectedError):
        teleop.connect()
    assert teleop.is_connected


def test_get_action_returns_sampled_inputs(teleop, ros):
    teleop.connect()

    pub_node = rclpy.create_node("teleop_fixture_pub")
    pub = pub_node.create_publisher(JointState, INPUT_TOPIC, 10)
    msg = JointState()
    msg.name = JOINTS
    msg.position = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    try:
        deadline = time.monotonic() + 5.0
        action = {}
        while time.monotonic() < deadline and not action:
            # The contract aligns this input on the header timeline; an
            # unstamped message is dropped at ingest (never fabricated).
            msg.header.stamp = pub_node.get_clock().now().to_msg()
            pub.publish(msg)
            time.sleep(0.05)  # teleop spins on its own thread
            action = teleop.get_action()

        assert action, "get_action() never returned data"
        # Each joint lands under its contract name, in order, after the contract's rad2deg.
        assert list(action) == list(teleop.action_features)
        assert list(action.values()) == pytest.approx(np.rad2deg(msg.position).tolist())
    finally:
        pub_node.destroy_node()


def test_disconnect_clean_and_reconnectable(teleop):
    teleop.connect()
    teleop.disconnect()  # must not raise on the ACTIVE -> cleanup path
    assert not teleop.is_connected
    # Full cycle: reconnect works.
    teleop.connect()
    assert teleop.is_connected
    teleop.disconnect()
