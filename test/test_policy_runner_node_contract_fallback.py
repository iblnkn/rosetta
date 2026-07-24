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
node hands pretrained_name_or_path to find_contract_for_pretrained and fails
the transition when the chase comes back empty. Failures are ERROR, not
FAILURE: _setup() raises and the base routes through on_error so partial
construction is torn down. The chase itself is pure and covered in
test_contract_fallback_chase.py."""

from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.parameter import Parameter

import rosetta.robots.ros2.nodes.policy_runner_node as node_mod
from rosetta.robots.ros2.nodes.policy_runner_node import PolicyRunnerNode


def test_configure_without_contract_chases_pretrained_then_fails(monkeypatch):
    seen = {}

    def _fake_chase(pretrained, *, warn):
        seen["pretrained"] = pretrained

    monkeypatch.setattr(node_mod, "find_contract_for_pretrained", _fake_chase)

    node = PolicyRunnerNode(parameter_overrides=[Parameter("pretrained_name_or_path", value="org/ckpt")])
    try:
        result = node.on_configure(None)
    finally:
        node.destroy_node()

    assert result == TransitionCallbackReturn.ERROR
    assert seen == {"pretrained": "org/ckpt"}


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


def test_configure_with_unknown_framework_fails(tmp_path):
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(MINIMAL_CONTRACT)

    node = PolicyRunnerNode(
        parameter_overrides=[
            Parameter("contract_path", value=str(contract_path)),
            Parameter("framework", value="nonexistent"),
        ]
    )
    try:
        result = node.on_configure(None)
    finally:
        node.destroy_node()

    assert result == TransitionCallbackReturn.ERROR


class _SetupOnlyRunner:
    """Minimal PolicyRunner stand-in for configure/cleanup wiring tests."""

    def setup(self, node, contract):
        pass

    def request_stop(self):
        pass

    def teardown(self):
        pass


def test_is_classifier_wires_reward_specs_as_actions(monkeypatch, tmp_path):
    """is_classifier=True must build the bridge from the reward section, not actions."""
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(MINIMAL_CONTRACT)

    calls = []
    monkeypatch.setattr(node_mod, "iter_reward_as_action_specs", lambda c: calls.append("reward") or [])
    monkeypatch.setattr(node_mod, "iter_action_specs", lambda c: calls.append("action") or [])
    monkeypatch.setattr(node_mod, "load_policy_runner", lambda fw: _SetupOnlyRunner())

    node = PolicyRunnerNode(
        parameter_overrides=[
            Parameter("contract_path", value=str(contract_path)),
            Parameter("is_classifier", value=True),
        ]
    )
    try:
        assert node.on_configure(None) == TransitionCallbackReturn.SUCCESS
        assert calls == ["reward"]
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
    monkeypatch.setattr(node_mod, "find_contract_for_pretrained", lambda p, *, warn: contract_path)
    monkeypatch.setattr(node_mod, "scan_inline_codec_paths", lambda p: order.append("scan") or real_scan(p))
    monkeypatch.setattr(node_mod, "load_contract", lambda p: order.append("load") or real_load(p))
    monkeypatch.setattr(node_mod, "load_policy_runner", lambda fw: _SetupOnlyRunner())

    node = PolicyRunnerNode(parameter_overrides=[Parameter("pretrained_name_or_path", value="org/ckpt")])
    try:
        assert node.on_configure(None) == TransitionCallbackReturn.SUCCESS
        assert order == ["scan", "load"]
    finally:
        node.on_cleanup(None)
        node.destroy_node()
