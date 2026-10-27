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

"""Shipped contracts' teleop wiring works in the HIL manager.

Teleop input goes through the real CommandForwarder (decode with the teleop
source, encode with its target action), so a width or unit mismatch in a
shipped contract fails here instead of on every message at runtime.
"""

import pytest
from forwarder_fakes import SO_101_HIL, STONE, forwarder
from sensor_msgs.msg import JointState

from rosetta.contract.schema import load_contract
from rosetta.contract.specs import iter_action_specs, iter_teleop_input_specs
from rosetta_ros_hil.hil_episode import Control, ignored_events


@pytest.fixture(params=[STONE, SO_101_HIL], ids=lambda path: path.name)
def contract(request):
    return load_contract(request.param)


def _positions(msg: JointState) -> dict[str, float]:
    return dict(zip(msg.name, msg.position, strict=True))


def test_teleop_input_reaches_the_action_topic_in_its_units(contract):
    node, control = forwarder(contract)
    control.now = Control.TELEOP
    for tis, spec in zip(contract.teleop.input, iter_teleop_input_specs(contract), strict=True):
        names = sorted(path.split(".", 1)[1] for path in spec.source.select)
        leader = JointState(name=names, position=[0.1 * (i + 1) for i in range(len(names))])
        node.deliver(spec.source.channel.topic, leader)
        (command,) = node.published[tis.target]
        assert _positions(command) == pytest.approx(_positions(leader))


def test_teleop_input_is_named_like_the_action_it_drives(contract):
    """Leader-side columns line up with the robot's action columns, namespace included."""
    actions = {s.source.channel.topic: s for s in iter_action_specs(contract)}
    for tis, spec in zip(contract.teleop.input, iter_teleop_input_specs(contract), strict=True):
        target = actions[tis.target]
        assert (spec.namespace, spec.names) == (target.namespace, target.names)


def test_every_event_hil_launch_wires_is_one_the_manager_acts_on():
    """so_101_hil.yaml is the contract hil_launch defaults to; a typo'd event name would never fire."""
    assert ignored_events(load_contract(SO_101_HIL).teleop.events.select) == []
