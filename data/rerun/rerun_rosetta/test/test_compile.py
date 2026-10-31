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

"""Contract -> query compilation: shape and fail-fast behavior.

A fake layout with its own column scheme and timeline names stands in for the
robot interface's, so these tests also show that compiling knows nothing about
any middleware's recordings.
"""

from __future__ import annotations

import textwrap

import pytest

pytest.importorskip("rerun")

from rerun_rosetta import RecordingLayout, compile_contract, load_layout

from rosetta.contract.schema import load_contract

BASE = textwrap.dedent(
    """
    robot_type: compile_test
    robot_interface: ros2
    fps: 15

    observations:
      observation.state:
        channel: {{topic: /js, type: sensor_msgs/msg/JointState,
                  qos: {{reliability: best_effort, history: keep_last, depth: 10}}}}
        align: {obs_align}
        select: [position.a]

    actions:
      action:
        channel: {{topic: /cmd, type: sensor_msgs/msg/JointState,
                  qos: {{reliability: best_effort, history: keep_last, depth: 10}}}}
        align: {act_align}
        select: [position.a]
    """
)
HOLD = "{strategy: hold, timeline: receive}"


class FakeLayout:
    """Columns named ``rec<topic>``; ``unreadable<topic>`` marks a form it can't read."""

    def __init__(self, receive_timeline="receive", index="arrival_ns"):
        self.receive_timeline = receive_timeline
        self.index = index

    def column(self, channel, columns):
        if f"unreadable{channel.topic}" in columns:
            raise ValueError(f"{channel.topic}: convert it")
        path = f"rec{channel.topic}"
        return path if path in columns else None

    def message(self, cell):
        return cell


JS, CMD = "rec/js", "rec/cmd"


def _contract(tmp_path, obs_align: str = HOLD, act_align: str = HOLD):
    path = tmp_path / "contract.yaml"
    path.write_text(BASE.format(obs_align=obs_align, act_align=act_align))
    return load_contract(path)


def test_fake_layout_is_a_recording_layout():
    assert isinstance(FakeLayout(), RecordingLayout)


def test_compiles_the_columns_and_index_the_layout_names(tmp_path):
    q = compile_contract(_contract(tmp_path), [JS, CMD], FakeLayout())
    assert q.index == "arrival_ns"
    assert q.sampling.rate_hz == 15.0
    assert [f.path for f in q.fields.values()] == [JS, CMD]
    assert q.required == ("f0",)


def test_the_receive_timeline_is_the_layouts(tmp_path):
    header = "{strategy: hold, timeline: header}"
    q = compile_contract(_contract(tmp_path, header, header), [JS, CMD], FakeLayout(receive_timeline="header"))
    assert q.index == "arrival_ns"


def test_a_form_the_layout_cannot_read_raises_its_error(tmp_path):
    with pytest.raises(ValueError, match="convert it"):
        compile_contract(_contract(tmp_path), [JS, "unreadable/cmd"], FakeLayout())


def test_window_offsets_are_grid_steps_in_seconds(tmp_path):
    q = compile_contract(_contract(tmp_path), [JS, CMD], FakeLayout(), windows={"action": (0, 2)})
    assert q.fields["f1"].window == (0.0, 1 / 15, 2 / 15)
    assert q.fields["f0"].window is None


def test_missing_action_topic_zero_fills(tmp_path):
    q = compile_contract(_contract(tmp_path), [JS], FakeLayout())
    assert list(q.fields) == ["f0"]


def test_missing_observation_topic_raises(tmp_path):
    with pytest.raises(ValueError, match="Observation topics missing"):
        compile_contract(_contract(tmp_path), [CMD], FakeLayout())


def test_asof_rejected(tmp_path):
    contract = _contract(tmp_path, obs_align="{strategy: asof, timeline: receive, tolerance_ms: 50}")
    with pytest.raises(ValueError, match="only 'hold'"):
        compile_contract(contract, [JS, CMD], FakeLayout())


def test_a_timeline_other_than_the_layouts_receive_timeline_is_rejected(tmp_path):
    contract = _contract(tmp_path, obs_align="{strategy: hold, timeline: header}")
    with pytest.raises(ValueError, match="only 'receive'"):
        compile_contract(contract, [JS, CMD], FakeLayout())


def test_unknown_window_key_rejected(tmp_path):
    with pytest.raises(ValueError, match="not in the contract"):
        compile_contract(_contract(tmp_path), [JS, CMD], FakeLayout(), windows={"nope": (0, 1)})


def test_load_layout_finds_the_registered_ros2_layout():
    pytest.importorskip("rosetta_ros")
    assert type(load_layout("ros2")).__name__ == "Rosbag2RrdLayout"


def test_load_layout_names_what_is_installed():
    with pytest.raises(ValueError, match="No Rerun recording layout for robot interface 'nope'"):
        load_layout("nope")
