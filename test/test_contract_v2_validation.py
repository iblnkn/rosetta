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

"""The contract loader's strictness matrix.

Validation replaces fallback: unknown keys, missing align, a timeline the
channel cannot provide — every lie a contract could tell is a load-time
error with a dotted-path context, never a silent default. Each test here
pins one rule.
"""

import pytest

from rosetta.contract.errors import ContractValidationError
from rosetta.contract.schema import load_contract

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


def _load(tmp_path, text):
    p = tmp_path / "c.yaml"
    p.write_text(text)
    return load_contract(p)


def _expect_error(tmp_path, text, match):
    with pytest.raises(ContractValidationError, match=match):
        _load(tmp_path, text)


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


def test_unknown_top_level_key_rejected(tmp_path):
    _expect_error(tmp_path, BASE + "visualization: {}\n", "visualization")


def test_x_prefixed_top_level_keys_ignored(tmp_path):
    c = _load(tmp_path, "x-qos:\n  a: &a {depth: 5}\n" + BASE)
    assert c.robot_type == "test"


def test_missing_robot_interface_rejected(tmp_path):
    _expect_error(tmp_path, BASE.replace("robot_interface: ros2\n", ""), "robot_interface")


def test_unsupported_robot_interface_rejected(tmp_path):
    _expect_error(tmp_path, BASE.replace("robot_interface: ros2", "robot_interface: zenoh"), "zenoh")


def test_missing_fps_rejected(tmp_path):
    _expect_error(tmp_path, BASE.replace("fps: 30\n", ""), "fps")


def test_non_positive_fps_rejected(tmp_path):
    _expect_error(tmp_path, BASE.replace("fps: 30", "fps: 0"), "fps")


# ---------------------------------------------------------------------------
# Align: mandatory, no defaults, asof <-> tolerance coupling
# ---------------------------------------------------------------------------


def test_missing_align_rejected(tmp_path):
    _expect_error(tmp_path, BASE.replace("    align: {strategy: hold, timeline: receive}\n", "", 1), "align")


def test_align_missing_strategy_rejected(tmp_path):
    _expect_error(tmp_path, BASE.replace("{strategy: hold, timeline: receive}", "{timeline: receive}", 1), "strategy")


def test_align_missing_timeline_rejected(tmp_path):
    _expect_error(tmp_path, BASE.replace("{strategy: hold, timeline: receive}", "{strategy: hold}", 1), "timeline")


def test_align_unknown_strategy_rejected(tmp_path):
    _expect_error(tmp_path, BASE.replace("strategy: hold", "strategy: nearest", 1), "nearest")


def test_align_unknown_key_rejected(tmp_path):
    _expect_error(
        tmp_path,
        BASE.replace("{strategy: hold, timeline: receive}", "{strategy: hold, timeline: receive, stamp: header}", 1),
        "stamp",
    )


def test_asof_requires_tolerance(tmp_path):
    _expect_error(
        tmp_path,
        BASE.replace("{strategy: hold, timeline: receive}", "{strategy: asof, timeline: receive}", 1),
        "tolerance_ms",
    )


def test_asof_with_tolerance_accepted(tmp_path):
    c = _load(
        tmp_path,
        BASE.replace("{strategy: hold, timeline: receive}", "{strategy: asof, timeline: receive, tolerance_ms: 10}", 1),
    )
    src = c.observations[0].sources[0]
    assert (src.align.strategy, src.align.tolerance_ms) == ("asof", 10)


def test_tolerance_without_asof_rejected(tmp_path):
    _expect_error(
        tmp_path,
        BASE.replace("{strategy: hold, timeline: receive}", "{strategy: hold, timeline: receive, tolerance_ms: 10}", 1),
        "tolerance_ms",
    )


def test_non_positive_tolerance_rejected(tmp_path):
    _expect_error(
        tmp_path,
        BASE.replace("{strategy: hold, timeline: receive}", "{strategy: asof, timeline: receive, tolerance_ms: 0}", 1),
        "tolerance_ms",
    )


# ---------------------------------------------------------------------------
# Timelines: channels provide, align selects — membership checked at load
# ---------------------------------------------------------------------------


def test_header_timeline_on_headerless_type_rejected(tmp_path):
    pytest.importorskip("rosidl_runtime_py")
    text = (
        BASE
        + """
rewards:
  next.reward:
    channel: {topic: /reward, type: std_msgs/msg/Float64, dtype: float64}
    align: {strategy: hold, timeline: header}
"""
    )
    # The error names what the channel does provide.
    _expect_error(tmp_path, text, "provides.*receive")


def test_unknown_timeline_name_rejected(tmp_path):
    pytest.importorskip("rosidl_runtime_py")
    _expect_error(tmp_path, BASE.replace("timeline: receive}", "timeline: publish}", 1), "publish")


def test_unknown_message_type_rejected(tmp_path):
    pytest.importorskip("rosidl_runtime_py")
    _expect_error(
        tmp_path,
        BASE.replace("type: sensor_msgs/msg/JointState}", "type: sensor_msgs/msg/NoSuchThing}", 1),
        "NoSuchThing",
    )


# ---------------------------------------------------------------------------
# Sources and channels: strict keys, per-section field rules
# ---------------------------------------------------------------------------


def test_unknown_source_key_rejected(tmp_path):
    _expect_error(
        tmp_path,
        BASE.replace(
            "    select: [position.j1]\nactions",
            "    select: [position.j1]\n    serve: {safety: zeros}\nactions",
        ),
        "serve",
    )


def test_unknown_channel_key_rejected(tmp_path):
    _expect_error(
        tmp_path,
        BASE.replace("type: sensor_msgs/msg/JointState}", "type: sensor_msgs/msg/JointState, stamp: header}", 1),
        "stamp",
    )


def test_empty_topic_rejected(tmp_path):
    _expect_error(tmp_path, BASE.replace("topic: /joint_states,", "topic: '',", 1), "topic")


def test_safety_on_observation_rejected(tmp_path):
    _expect_error(tmp_path, BASE.replace("topic: /joint_states,", "topic: /joint_states, safety: zeros,", 1), "safety")


def test_encoder_on_observation_rejected(tmp_path):
    _expect_error(
        tmp_path,
        BASE.replace("topic: /joint_states,", "topic: /joint_states, encoder: 'json:dumps',", 1),
        "encoder",
    )


def test_empty_source_list_rejected(tmp_path):
    _expect_error(
        tmp_path,
        BASE.replace(
            """  observation.state:
    channel: {topic: /joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.j1]
""",
            "  observation.state: []\n",
        ),
        "empty",
    )


def test_shared_key_sources_ordered(tmp_path):
    text = BASE.replace(
        """  observation.state:
    channel: {topic: /joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.j1]
""",
        """  observation.state:
    - channel: {topic: /arm/joint_states, type: sensor_msgs/msg/JointState}
      align: {strategy: hold, timeline: receive}
      select: [position.j1]
    - channel: {topic: /base/joint_states, type: sensor_msgs/msg/JointState}
      align: {strategy: drop, timeline: receive}
      select: [position.j2]
""",
    )
    entry = _load(tmp_path, text).observations[0]
    assert [s.channel.topic for s in entry.sources] == ["/arm/joint_states", "/base/joint_states"]
    # Per-source align is real: the two sources differ.
    assert [s.align.strategy for s in entry.sources] == ["hold", "drop"]


# ---------------------------------------------------------------------------
# Extended sections: dtype mandatory, never images
# ---------------------------------------------------------------------------

EXTENDED = (
    BASE
    + """
signals:
  next.done:
    channel: {topic: /done, type: std_msgs/msg/Bool, dtype: bool}
    align: {strategy: hold, timeline: receive}
"""
)


def test_extended_section_accepted(tmp_path):
    c = _load(tmp_path, EXTENDED)
    assert c.signals[0].key == "next.done"
    assert c.signals[0].sources[0].channel.dtype == "bool"


def test_extended_without_dtype_rejected(tmp_path):
    _expect_error(tmp_path, EXTENDED.replace(", dtype: bool", ""), "dtype")


def test_extended_video_dtype_rejected(tmp_path):
    _expect_error(tmp_path, EXTENDED.replace("dtype: bool", "dtype: video"), "video")


def test_extended_image_block_rejected(tmp_path):
    _expect_error(tmp_path, EXTENDED.replace("dtype: bool", "dtype: bool, image: {encoding: rgb8}"), "image")


def test_extended_image_key_rejected(tmp_path):
    _expect_error(tmp_path, EXTENDED.replace("next.done", "observation.images.done"), "images")


# ---------------------------------------------------------------------------
# Tasks / adjunct / teleop shapes
# ---------------------------------------------------------------------------


def test_task_with_align_rejected(tmp_path):
    text = (
        BASE
        + """
tasks:
  task:
    channel: {topic: /task, type: std_msgs/msg/String}
    align: {strategy: hold, timeline: receive}
"""
    )
    _expect_error(tmp_path, text, "align")


def test_adjunct_must_be_list(tmp_path):
    _expect_error(tmp_path, BASE + "adjunct:\n  channel: {topic: /tf, type: tf2_msgs/msg/TFMessage}\n", "list")


def test_adjunct_extra_channel_fields_rejected(tmp_path):
    text = (
        BASE
        + """
adjunct:
  - channel: {topic: /tf, type: tf2_msgs/msg/TFMessage, dtype: float64}
"""
    )
    _expect_error(tmp_path, text, "dtype")


def test_teleop_events_with_align_rejected(tmp_path):
    text = (
        BASE
        + """
teleop:
  events:
    channel: {topic: /joy, type: sensor_msgs/msg/Joy}
    align: {strategy: hold, timeline: receive}
    select: {success: buttons.0}
"""
    )
    _expect_error(tmp_path, text, "align")


def test_teleop_unknown_role_rejected(tmp_path):
    text = (
        BASE
        + """
teleop:
  inputs:
    channel: {topic: /leader, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
"""
    )
    _expect_error(tmp_path, text, "inputs")
