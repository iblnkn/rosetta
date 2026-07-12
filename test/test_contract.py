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

"""Tests for contract loading and schema validation."""

from pathlib import Path

import pytest
from rosetta.contract.errors import ContractValidationError
from rosetta.contract.schema import (
    is_valid_lerobot_dtype,
    load_contract,
)

CONTRACTS_DIR = Path(__file__).resolve().parent.parent / "contracts"

VALID_CONTRACT = """
robot_type: test
robot_interface: ros2
fps: 30
observations:
  observation.state:
    channel: {topic: /joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.j1, position.j2]
    apply: [rad2deg]
actions:
  action:
    channel: {topic: /cmd, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.j1, position.j2]
    apply: [rad2deg]
"""

# The pre-channel-block v1 format: sections were lists of flat entries.
V1_CONTRACT = """
robot_type: test
robot_interface: ros2
fps: 30
observations:
  - key: observation.state
    topic: /joint_states
    type: sensor_msgs/msg/JointState
    select: [position.j1]
"""


@pytest.mark.parametrize("name", ["so_101.yaml", "so_101_hil.yaml", "turtlebot3.yaml", "stone.yaml"])
def test_bundled_contracts_load(name):
    contract = load_contract(CONTRACTS_DIR / name)
    assert contract.robot_type
    assert contract.robot_interface == "ros2"
    assert contract.fps > 0
    # Every bundled contract defines at least one observation and one action.
    assert len(contract.observations) >= 1
    assert len(contract.actions) >= 1


def test_stone_showcases_every_feature():
    """stone.yaml is the format tour; pin the structure it demonstrates."""
    c = load_contract(CONTRACTS_DIR / "stone.yaml")

    by_key = {e.key: e for e in c.observations}
    # Shared-key concatenation: ordered sources feeding one key.
    state = by_key["observation.state"]
    assert [s.channel.topic for s in state.sources] == [
        "/arm/joint_states",
        "/gripper/joint_states",
        "/odom",
    ]
    # Action splitting with per-channel safety.
    action = {e.key: e for e in c.actions}["action"]
    assert [(s.channel.topic, s.channel.safety) for s in action.sources] == [
        ("/arm/joint_commands", "hold"),
        ("/gripper/command", "hold"),
    ]
    # All three align strategies appear.
    strategies = {s.align.strategy for _, e in c.frame_entries() for s in e.sources}
    assert strategies == {"hold", "asof", "drop"}
    # Both demonstrated timelines appear.
    timelines = {s.align.timeline for _, e in c.frame_entries() for s in e.sources}
    assert timelines == {"header", "receive"}
    # Tasks, teleop roles, adjunct.
    assert [(t.key, t.channel.topic) for t in c.tasks] == [("task", "/task_prompt")]
    assert c.teleop.input is not None
    assert c.teleop.events is not None and "is_intervention" in c.teleop.events.select
    assert c.teleop.feedback is not None
    assert [a.topic for a in c.adjunct] == ["/tf", "/tf_static", "/scan"]
    # Extended sections are present with mandatory dtypes.
    assert {e.key for e in c.rewards} == {"next.reward"}
    assert {e.key for e in c.signals} == {"next.done", "next.truncated"}
    assert c.signals[0].sources[0].channel.dtype == "bool"


def test_valid_contract_round_trips(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(VALID_CONTRACT)
    contract = load_contract(p)
    assert contract.robot_type == "test"
    assert contract.robot_interface == "ros2"
    assert contract.observations[0].key == "observation.state"
    assert contract.actions[0].key == "action"
    src = contract.observations[0].sources[0]
    assert src.channel.topic == "/joint_states"
    assert src.align.strategy == "hold"
    assert src.align.timeline == "receive"


def test_safety_defaults_to_none(tmp_path):
    """An action channel without `safety` must NOT fabricate safety commands.

    Safety pin: a zero command is only a safe stop under velocity control —
    under position control (the common case) it commands a slam to the zero
    pose. Safety behavior is an explicit, per-channel opt-in.
    """
    p = tmp_path / "c.yaml"
    p.write_text(VALID_CONTRACT)
    contract = load_contract(p)
    assert contract.actions[0].sources[0].channel.safety == "none"


def test_safety_explicit_optin_preserved(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(VALID_CONTRACT.replace("topic: /cmd,", "topic: /cmd, safety: zeros,"))
    contract = load_contract(p)
    assert contract.actions[0].sources[0].channel.safety == "zeros"


def test_v1_list_sections_rejected_with_pointed_error(tmp_path):
    p = tmp_path / "v1.yaml"
    p.write_text(V1_CONTRACT)
    with pytest.raises(ContractValidationError, match="v1"):
        load_contract(p)


@pytest.mark.parametrize(
    "dtype",
    ["float32", "float64", "int32", "int64", "bool", "uint8", "video", "image", "string"],
)
def test_valid_dtypes_accepted(dtype):
    assert is_valid_lerobot_dtype(dtype)


@pytest.mark.parametrize("dtype", ["not_a_dtype", "flot32", ""])
def test_invalid_dtypes_rejected(dtype):
    assert not is_valid_lerobot_dtype(dtype)


def test_missing_robot_type_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("robot_interface: ros2\nfps: 30\n")
    with pytest.raises(ContractValidationError):
        load_contract(p)


TELEOP_FEEDBACK_CONTRACT = (
    VALID_CONTRACT
    + """
teleop:
  feedback:
    channel: {topic: /wrist_cmd, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [orientation.x, orientation.y, orientation.z, orientation.w]
    kind: quaternion
"""
)


def test_teleop_feedback_parses_with_kind(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(TELEOP_FEEDBACK_CONTRACT)
    fb = load_contract(p).teleop.feedback.sources[0]
    assert fb.channel.safety == "none"  # feedback never fabricates safety commands
    assert fb.kind == "quaternion"


def test_teleop_feedback_rejects_safety(tmp_path):
    """Feedback declaring `safety` is a contract lie — error, not a silent override."""
    p = tmp_path / "c.yaml"
    p.write_text(TELEOP_FEEDBACK_CONTRACT.replace("topic: /wrist_cmd,", "topic: /wrist_cmd, safety: zeros,"))
    with pytest.raises(ContractValidationError, match="safety"):
        load_contract(p)
