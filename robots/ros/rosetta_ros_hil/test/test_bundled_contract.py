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

"""The contract hil_launch defaults to loads, and every event it wires is one the manager acts on."""

from pathlib import Path

import pytest
from sensor_msgs.msg import JointState

from rosetta.contract.schema import load_contract
from rosetta.contract.specs import iter_action_specs, iter_teleop_input_specs
from rosetta.frames.codecs import decode_value, encode_value
from rosetta_ros_hil.hil_episode import ignored_events

CONTRACT = Path(__file__).resolve().parent.parent / "contracts" / "so_101_hil.yaml"


@pytest.fixture(scope="module")
def contract():
    return load_contract(CONTRACT)


def test_every_wired_event_is_one_the_manager_acts_on(contract):
    assert ignored_events(contract.teleop.events.select) == []


def test_teleop_input_reaches_the_action_topic_in_its_units(contract):
    """The manager re-encodes decoded teleop input onto the action topic; the round trip keeps radians."""
    actions = {s.source.channel.topic: s for s in iter_action_specs(contract)}
    for tis, spec in zip(contract.teleop.input, iter_teleop_input_specs(contract), strict=True):
        names = sorted(path.split(".", 1)[1] for path in spec.source.select)
        leader = JointState(name=names, position=[0.1 * i for i in range(len(names))])
        command = encode_value(decode_value(leader, spec), actions[tis.target])
        assert dict(zip(command.name, command.position, strict=True)) == pytest.approx(
            dict(zip(leader.name, leader.position, strict=True))
        )


def test_teleop_input_is_named_like_the_action_it_drives(contract):
    """Leader-side columns line up with the robot's action columns, namespace included."""
    actions = {s.source.channel.topic: s for s in iter_action_specs(contract)}
    for tis, spec in zip(contract.teleop.input, iter_teleop_input_specs(contract), strict=True):
        target = actions[tis.target]
        assert (spec.namespace, spec.names) == (target.namespace, target.names)
