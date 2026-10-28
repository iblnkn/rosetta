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

"""rosetta-port's CLI flags reach the dataset writer's open().

main() is driven end to end over a real one-message bag; only the writer is a
stub, so a flag dropped anywhere between argv, port(), write_dataset() and
writer.open() fails here. A hub push is private unless --hub-public is given.
"""

import sys

import pytest
from bridge_helpers import StubWriter, joint_state, write_bag

from rosetta_ros.offline import port as port_mod

CONTRACT = """
robot_type: test
robot_interface: ros2
fps: 30
observations:
  observation.state:
    channel: {topic: /joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.j1]
"""


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], {"hub_private": True, "hub_tags": None, "embed_contract": True}),
        (["--hub-public"], {"hub_private": False}),
        (["--hub-tags", "a, b ,c"], {"hub_tags": ["a", "b", "c"]}),
        (["--no-embed-contract"], {"embed_contract": False}),
    ],
    ids=["defaults", "hub-public", "hub-tags", "no-embed-contract"],
)
def test_cli_flags_reach_writer_open(tmp_path, monkeypatch, flags, expected):
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(CONTRACT)
    write_bag(tmp_path / "raw" / "ep0", [("/joint_states", 1_000_000_000, joint_state(["j1"], [0.5]))])
    writer = StubWriter()
    monkeypatch.setattr(port_mod, "load_dataset_writer", lambda _framework: writer)
    monkeypatch.setattr(
        sys, "argv", ["rosetta-port", "--raw-dir", str(tmp_path / "raw"), "--contract", str(contract_path), *flags]
    )

    port_mod.main()

    assert writer.open_kwargs["contract_path"] == contract_path
    assert {k: writer.open_kwargs.get(k) for k in expected} == expected
