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

"""Shipped teleop wiring survives the HIL node's decode -> encode hop.

rosetta_hil's HIL manager forwards teleop input as decode(teleop spec) -> encode(target
action spec), and feedback as decode(origin observation spec) -> encode(feedback
spec). These tests make the same calls on the shipped contracts, so a width or
unit mismatch fails here instead of on every message at runtime.
"""

from pathlib import Path

import numpy as np
import pytest
from sensor_msgs.msg import JointState

from rosetta.contract.schema import load_contract
from rosetta.contract.specs import (
    iter_action_specs,
    iter_observation_specs,
    iter_teleop_feedback_specs,
    iter_teleop_input_specs,
)
from rosetta.frames.codecs import decode_value, encode_value

CONTRACTS_DIR = Path(__file__).resolve().parent.parent / "contracts"
TELEOP_CONTRACTS = ["stone.yaml", "so_101_hil.yaml"]


def _joint_state(spec) -> JointState:
    """A JointState carrying every field the spec selects, with distinct radian values."""
    names = sorted({path.split(".", 1)[1] for path in spec.source.select})
    values = np.linspace(-1.2, 1.2, len(names))
    return JointState(name=names, position=list(values), velocity=list(values), effort=list(values))


def _positions(msg: JointState) -> dict[str, float]:
    return dict(zip(msg.name, msg.position, strict=True))


@pytest.mark.parametrize("name", TELEOP_CONTRACTS)
def test_teleop_input_reaches_the_action_topic_in_its_units(name):
    contract = load_contract(CONTRACTS_DIR / name)
    actions = {s.source.channel.topic: s for s in iter_action_specs(contract)}
    for tis, teleop_spec in zip(contract.teleop.input, iter_teleop_input_specs(contract), strict=True):
        leader = _joint_state(teleop_spec)
        command = encode_value(decode_value(leader, teleop_spec), actions[tis.target])
        assert _positions(command) == pytest.approx(_positions(leader))


@pytest.mark.parametrize("name", TELEOP_CONTRACTS)
def test_teleop_feedback_reaches_the_device_in_its_units(name):
    contract = load_contract(CONTRACTS_DIR / name)
    if not contract.teleop.feedback:
        pytest.skip(f"{name} declares no teleop feedback")
    observations = {s.source.channel.topic: s for s in iter_observation_specs(contract)}
    for tfs, feedback_spec in zip(contract.teleop.feedback, iter_teleop_feedback_specs(contract), strict=True):
        follower = _joint_state(observations[tfs.origin])
        feedback = encode_value(decode_value(follower, observations[tfs.origin]), feedback_spec)
        sent = _positions(feedback)
        assert sent == pytest.approx({k: v for k, v in _positions(follower).items() if k in sent})
