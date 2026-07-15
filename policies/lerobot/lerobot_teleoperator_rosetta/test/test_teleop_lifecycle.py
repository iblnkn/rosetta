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

"""Input-only teleop lifecycle end-to-end (the so_101_hil shape).

Regression guard: is_active used to read the first feedback publisher's
is_activated — an input-only teleop (feedback: [], the repo's only teleop
contract) therefore never reported connected, get_action() returned {}
forever, and disconnect() raised on an invalid ACTIVE->cleanup transition,
leaking the spin thread. Also pins the _on_input timestamp fix (the helper
returns a (ts, used_fallback) tuple).
"""

import time
from pathlib import Path

import pytest
import rclpy
from ament_index_python.packages import get_package_share_directory
from lerobot.utils.errors import DeviceAlreadyConnectedError
from lerobot_teleoperator_rosetta.config_rosetta_teleop import RosettaTeleopConfig
from lerobot_teleoperator_rosetta.rosetta_teleop import RosettaTeleop
from sensor_msgs.msg import JointState

CONTRACT = Path(get_package_share_directory("rosetta")) / "contracts" / "so_101_hil.yaml"
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


@pytest.fixture
def ros():
    """Global context for fixture nodes (the teleop hosts its own private context)."""
    rclpy.init()
    yield
    rclpy.try_shutdown()


def test_input_only_teleop_connects(teleop):
    # The regression: feedback: [] used to mean is_connected was never True.
    teleop.connect()
    assert teleop.is_connected
    assert not teleop._node.bridge._act_publishers  # genuinely input-only


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

        # Pins both the is_active fix (would be {} forever) and the tuple
        # timestamp fix (would raise TypeError in StreamBuffer.sample).
        assert action, "get_action() never returned data"
        assert all(isinstance(v, float) for v in action.values())
        assert len(action) == len(JOINTS)
    finally:
        pub_node.destroy_node()


def test_disconnect_clean_and_reconnectable(teleop):
    teleop.connect()
    teleop.disconnect()  # used to raise ACTIVE -> cleanup
    assert teleop._node is None
    assert teleop._host._spin_thread is None
    # Full cycle: reconnect works.
    teleop.connect()
    assert teleop.is_connected
    teleop.disconnect()
