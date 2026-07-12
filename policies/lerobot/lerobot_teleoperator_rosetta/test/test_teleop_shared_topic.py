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

"""Two teleop input specs on one topic.

Regression: input buffers were once a dict keyed by spec.topic, so a second
input spec reading the same topic silently overwrote the first — only the
last spec's buffer was ever sampled (both subscriptions stayed live). The
node now hosts a TopicBridge, whose buffers are a positional list.
"""

import time
from pathlib import Path

import pytest
import rclpy
from sensor_msgs.msg import JointState

from lerobot_teleoperator_rosetta.config_rosetta_teleop import RosettaTeleopConfig
from lerobot_teleoperator_rosetta.rosetta_teleop import RosettaTeleop

CONTRACT_YAML = """
robot_type: test
robot_interface: ros2
fps: 30
observations:
  observation.state:
    channel: {topic: /follower/joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.j1]
actions:
  action:
    channel: {topic: /leader/joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.j1]
teleop:
  input:
    - channel: {topic: /leader/joint_states, type: sensor_msgs/msg/JointState}
      align: {strategy: hold, timeline: receive}
      select: [position.j1, position.j2]
    - channel: {topic: /leader/joint_states, type: sensor_msgs/msg/JointState}
      align: {strategy: hold, timeline: receive}
      select: [position.grip]
"""

TOPIC = "/leader/joint_states"


@pytest.fixture
def ros():
    """Global context for fixture nodes (the teleop hosts its own private context)."""
    rclpy.init()
    yield
    rclpy.try_shutdown()


def test_two_input_specs_on_one_topic_both_sampled(tmp_path: Path, ros):
    contract = tmp_path / "teleop_shared.yaml"
    contract.write_text(CONTRACT_YAML)
    teleop = RosettaTeleop(RosettaTeleopConfig(id="shared_topic", config_path=str(contract)))
    teleop.connect()
    try:
        assert len(teleop._node.bridge._obs_buffers) == 2  # dict-by-topic kept only 1

        pub_node = rclpy.create_node("teleop_shared_fixture_pub")
        pub = pub_node.create_publisher(JointState, TOPIC, 10)
        msg = JointState()
        msg.name = ["j1", "j2", "grip"]
        msg.position = [0.1, 0.2, 0.9]
        try:
            deadline = time.monotonic() + 5.0
            action = {}
            while time.monotonic() < deadline and len(action) < 3:
                pub.publish(msg)
                time.sleep(0.05)
                action = teleop.get_action()

            # Both specs' selectors present: the second spec no longer
            # clobbers the first spec's buffer.
            assert set(action) == {"position.j1", "position.j2", "position.grip"}
            assert action["position.grip"] == pytest.approx(0.9)
            assert action["position.j1"] == pytest.approx(0.1)
        finally:
            pub_node.destroy_node()
    finally:
        teleop.disconnect()
