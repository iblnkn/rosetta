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

"""CommandForwarder: whoever has control drives every action topic, once per message."""

import pytest
from forwarder_fakes import STONE, Controls, FakeNode, forwarder
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from rosetta.contract.schema import load_contract
from rosetta_ros_hil.forwarding import CommandForwarder
from rosetta_ros_hil.hil_episode import Control

ARM = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_pitch", "wrist_roll", "wrist_yaw"]


@pytest.fixture
def stone():
    return forwarder(load_contract(STONE))


def _arm(positions):
    return JointState(name=ARM, position=positions)


def test_policy_output_reaches_every_action_topic(stone):
    node, _ = stone
    commands = {"/arm/joint_commands": _arm([0.1] * 6), "/gripper/command": Float64(data=1.0)}
    commands["/cmd_vel"] = TwistStamped()
    for topic, msg in commands.items():
        node.deliver(f"/hil/policy{topic}", msg)
    assert {topic: node.published[topic] for topic in commands} == {t: [m] for t, m in commands.items()}


def test_policy_output_is_dropped_while_the_human_drives(stone):
    node, control = stone
    control.now = Control.TELEOP
    node.deliver("/hil/policy/arm/joint_commands", _arm([0.1] * 6))
    assert node.published["/arm/joint_commands"] == []


def test_teleop_input_is_ignored_while_the_policy_drives(stone):
    node, _ = stone
    node.deliver("/human/leader_arm/joint_states", _arm([0.1] * 6))
    assert node.published["/arm/joint_commands"] == []


@pytest.mark.parametrize("who", [Control.POLICY, Control.TELEOP])
def test_feedback_reaches_the_device_whoever_drives(stone, who):
    node, control = stone
    control.now = who
    node.deliver("/arm/joint_states", _arm([0.6, 0.5, 0.4, 0.3, 0.2, 0.1]))
    (feedback,) = node.published["/human/leader_arm/joint_feedback"]
    assert list(feedback.position) == pytest.approx([0.6, 0.5, 0.4, 0.3, 0.2, 0.1])


SHARED_TOPIC = """\
robot_type: t
robot_interface: ros2
fps: 10
observations:
  observation.state:
    channel: {topic: /joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.a]
actions:
  action:
    - channel: {topic: /cmd, type: sensor_msgs/msg/JointState}
      align: {strategy: hold, timeline: receive}
      select: [position.a]
    - channel: {topic: /cmd, type: sensor_msgs/msg/JointState}
      align: {strategy: hold, timeline: receive}
      select: [position.b]
"""

TELEOP_ON_SHARED_TOPIC = """\
teleop:
  input:
    - target: /cmd
      channel: {topic: /leader, type: sensor_msgs/msg/JointState}
      align: {strategy: hold, timeline: receive}
      select: [position.a]
"""


def _load(tmp_path, text):
    path = tmp_path / "c.yaml"
    path.write_text(text)
    return load_contract(path)


def test_a_topic_shared_by_two_sources_is_forwarded_once(tmp_path):
    node, _ = forwarder(_load(tmp_path, SHARED_TOPIC))
    node.deliver("/hil/policy/cmd", JointState(name=["a", "b"], position=[1.0, 2.0]))
    assert len(node.published["/cmd"]) == 1


def test_teleop_onto_a_shared_topic_is_rejected_and_cleaned_up(tmp_path):
    contract = _load(tmp_path, SHARED_TOPIC + TELEOP_ON_SHARED_TOPIC)
    node = FakeNode()
    with pytest.raises(ValueError, match="exactly one"):
        CommandForwarder(node, contract, "/hil/policy", Controls(), callback_group=None)
    assert node.destroyed == 2  # the /cmd publisher and the /hil/policy/cmd subscription it had made
