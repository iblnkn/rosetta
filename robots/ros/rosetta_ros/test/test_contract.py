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

"""Contracts load through the ros2 interface: the bundled contracts, and the
load-time rules that depend on real ROS types (timelines, qos, codecs).

The parser's own rules are tested in core against the fake interface.
"""

from pathlib import Path

import pytest

from rosetta.contract.errors import ContractValidationError
from rosetta.contract.schema import load_contract

CONTRACTS_DIR = Path(__file__).resolve().parent.parent / "contracts"

BASE = """
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


def _expect_error(tmp_path, text, match):
    p = tmp_path / "c.yaml"
    p.write_text(text)
    with pytest.raises(ContractValidationError, match=match):
        load_contract(p)


@pytest.mark.parametrize("name", ["so_101.yaml", "turtlebot3.yaml", "stone.yaml"])
def test_bundled_contracts_load(name):
    assert load_contract(CONTRACTS_DIR / name).robot_interface == "ros2"


def test_stone_showcases_every_feature():
    """stone.yaml is the format tour; pin that it still demonstrates each feature."""
    c = load_contract(CONTRACTS_DIR / "stone.yaml")
    sources = [s for _, e in c.frame_entries() for s in e.sources]

    assert {s.align.strategy for s in sources} == {"hold", "asof", "drop"}
    assert {s.align.timeline for s in sources} == {"header", "receive"}
    assert any(len(e.sources) > 1 for e in c.observations), "shared-key concatenation"
    assert any(len(e.sources) > 1 for e in c.actions), "action splitting"
    assert any(s.channel.safety != "none" for e in c.actions for s in e.sources), "per-channel safety"
    assert c.tasks and c.adjunct and c.rewards and c.signals
    assert c.teleop.input and c.teleop.events and c.teleop.feedback


def test_header_timeline_on_headerless_type_rejected(tmp_path):
    """std_msgs/msg/Float64 has no header, so it provides only `receive`; the error names what it provides."""
    text = (
        BASE
        + """
rewards:
  next.reward:
    channel: {topic: /reward, type: std_msgs/msg/Float64, dtype: float64}
    align: {strategy: hold, timeline: header}
"""
    )
    _expect_error(tmp_path, text, "provides.*receive")


def test_qos_typo_value_rejected_at_load(tmp_path):
    """A typo'd policy value must die at load, not silently become the default
    policy (best-effort camera vs reliable sub = zero messages, invisibly)."""
    text = BASE.replace(
        "channel: {topic: /joint_states, type: sensor_msgs/msg/JointState}",
        "channel: {topic: /joint_states, type: sensor_msgs/msg/JointState, qos: {reliability: best-effort}}",
    )
    _expect_error(tmp_path, text, r"observation\.state\.qos.*Invalid qos reliability")


def test_image_without_resize_rejected_at_load(tmp_path):
    """Spec resolution runs eagerly in load: an image with no output geometry fails here, not at first frame."""
    text = BASE.replace(
        "actions:",
        """  observation.images.cam:
    channel: {topic: /cam, type: sensor_msgs/msg/Image}
    align: {strategy: hold, timeline: receive}
actions:""",
        1,
    )
    _expect_error(tmp_path, text, "output geometry")


def test_unencodable_action_type_rejected_at_load(tmp_path):
    """sensor_msgs/msg/Image decodes but has no encoder; as an action it must
    fail at load, not first publish."""
    old_action = (
        "channel: {topic: /cmd, type: sensor_msgs/msg/JointState}\n"
        "    align: {strategy: hold, timeline: receive}\n"
        "    select: [position.j1]"
    )
    new_action = "channel: {topic: /cmd, type: sensor_msgs/msg/Image}\n    align: {strategy: hold, timeline: receive}"
    _expect_error(tmp_path, BASE.replace(old_action, new_action), "No encoder registered")
