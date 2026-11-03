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

"""on_configure wiring of the contract fallback: with no contract_path, the
node asks its policy runner's find_contract for the contract
pretrained_name_or_path was trained with, and fails the transition when the
runner finds none. Failures are ERROR, not FAILURE: setup() raises and the
base routes through on_error so partial construction is torn down. The LeRobot
chase itself is covered in lerobot_dataset_rosetta's
test_contract_fallback_chase.py."""

from node_helpers import FakeRunner
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.parameter import Parameter

import rosetta_ros.nodes.policy_runner_node as node_mod
from rosetta_ros.nodes.policy_runner_node import PolicyRunnerNode


def test_configure_without_contract_chases_pretrained_then_fails(monkeypatch):
    seen = {}

    class ChasingRunner(FakeRunner):
        def find_contract(self, pretrained, *, warn):
            seen["pretrained"] = pretrained

    monkeypatch.setattr(node_mod, "load_policy_runner", lambda fw: ChasingRunner())

    node = PolicyRunnerNode(parameter_overrides=[Parameter("pretrained_name_or_path", value="org/ckpt")])
    try:
        result = node.on_configure(None)
    finally:
        node.destroy_node()

    assert result == TransitionCallbackReturn.ERROR
    assert seen == {"pretrained": "org/ckpt"}


def test_configure_with_no_contract_and_no_checkpoint_fails_before_loading_a_runner(monkeypatch):
    # Loading LeRobot's runner imports torch, so the plain configuration
    # error must not wait on it.
    loaded = []
    monkeypatch.setattr(node_mod, "load_policy_runner", lambda fw: loaded.append(fw) or FakeRunner())

    node = PolicyRunnerNode()
    try:
        assert node.on_configure(None) == TransitionCallbackReturn.ERROR
    finally:
        node.destroy_node()

    assert loaded == []


MINIMAL_CONTRACT = """
robot_type: test
robot_interface: ros2
fps: 30
observations:
  observation.state:
    channel: {topic: /joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.j1]
actions:
  action:
    channel: {topic: /cmd, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.j1]
"""


CLASSIFIER_CONTRACT = (
    MINIMAL_CONTRACT
    + """
rewards:
  next.reward:
    channel: {topic: /reward, type: std_msgs/msg/Float64, dtype: float64}
    align: {strategy: hold, timeline: receive}
"""
)


def test_is_classifier_publishes_to_the_reward_topics(monkeypatch, tmp_path):
    """is_classifier=True must build the bridge from the reward section, not actions."""
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(CLASSIFIER_CONTRACT)
    monkeypatch.setattr(node_mod, "load_policy_runner", lambda fw: FakeRunner())

    node = PolicyRunnerNode(
        parameter_overrides=[
            Parameter("contract_path", value=str(contract_path)),
            Parameter("is_classifier", value=True),
        ]
    )
    try:
        assert node.on_configure(None) == TransitionCallbackReturn.SUCCESS
        published = {pub.topic_name for pub in node.publishers}
        assert "/reward" in published
        assert "/cmd" not in published
    finally:
        node.on_cleanup(None)
        node.destroy_node()


def test_sidecar_contract_scanned_for_inline_codecs_before_load(monkeypatch, tmp_path):
    """The hub-trust audit is only honest if it runs BEFORE load_contract:
    load_contract's parse already imports every inline codec path."""
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(MINIMAL_CONTRACT)

    order = []
    real_scan = node_mod.scan_inline_codec_paths
    real_load = node_mod.load_contract

    class FindingRunner(FakeRunner):
        def find_contract(self, pretrained, *, warn):
            return contract_path

    monkeypatch.setattr(node_mod, "scan_inline_codec_paths", lambda p: order.append("scan") or real_scan(p))
    monkeypatch.setattr(node_mod, "load_contract", lambda p: order.append("load") or real_load(p))
    monkeypatch.setattr(node_mod, "load_policy_runner", lambda fw: FindingRunner())

    node = PolicyRunnerNode(parameter_overrides=[Parameter("pretrained_name_or_path", value="org/ckpt")])
    try:
        assert node.on_configure(None) == TransitionCallbackReturn.SUCCESS
        assert order == ["scan", "load"]
    finally:
        node.on_cleanup(None)
        node.destroy_node()
