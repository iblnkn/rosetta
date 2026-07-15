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

"""port()'s wiring: what the contract declares must reach iter_bag_frames.

Regression: port() used to build its episode iterators without task_topics,
so the contract's ``tasks:`` section (per-frame task labels) silently never
fired in an actual port — only the iter_bag_frames level was tested, and the
one code path that passed task_topics correctly (BagFrameSource.episodes,
since deleted) had no callers.
"""

from pathlib import Path

import rosetta.robots.ros2.offline.port as port_mod
from rosetta.robots.ros2.offline.port import port

CONTRACT = """
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
tasks:
  task:
    channel: {topic: /task_prompt, type: std_msgs/msg/String}
"""


class _StubWriter:
    def open(self, **_kw):
        pass

    def add_frame(self, _frame):
        pass

    def save_episode(self):
        pass

    def discard_episode(self):
        pass

    def finalize(self):
        pass


def test_port_passes_task_topics_and_warmup_keys(tmp_path, monkeypatch):
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(CONTRACT)
    seen = {}

    def _fake_iter(bag_dir, specs, *, warmup_keys, task_topics=None):
        seen["warmup_keys"] = warmup_keys
        seen["task_topics"] = task_topics
        return iter([])

    monkeypatch.setattr(port_mod, "find_bag_dirs", lambda raw_dir, **_kw: [Path("ep1")])
    monkeypatch.setattr(port_mod, "iter_bag_frames", _fake_iter)
    monkeypatch.setattr(port_mod, "load_dataset_writer", lambda _fw: _StubWriter())

    port(raw_dir=tmp_path, repo_id="r", contract_path=contract_path)

    assert seen["task_topics"] == {"/task_prompt": "std_msgs/msg/String"}
    assert seen["warmup_keys"] == {"observation.state"}
