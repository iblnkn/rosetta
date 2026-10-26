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

"""Core works with any robot interface: a contract for the fake one loads, decodes and encodes."""

import importlib.metadata as ilm
from types import SimpleNamespace

import numpy as np
import pytest
from fake_interface import FakeVector

from rosetta.contract.errors import ContractValidationError
from rosetta.contract.schema import load_contract
from rosetta.contract.specs import iter_action_specs, iter_observation_specs
from rosetta.frames.codecs import decode_value, encode_value
from rosetta.robots import registry

CONTRACT = """\
robot_type: fake_bot
robot_interface: fake
fps: 10
observations:
  observation.state:
    channel: {topic: /state, type: fake/Vector, qos: {level: 1}}
    align: {strategy: hold, timeline: arrival}
    select: [a, b]
actions:
  action:
    channel: {topic: /cmd, type: fake/Vector}
    align: {strategy: hold, timeline: arrival}
    select: [a, b]
"""


@pytest.fixture
def fake_interface(monkeypatch):
    """Make ``fake`` the only installed robot interface for one test."""
    ep = ilm.EntryPoint(name="fake", value="fake_interface:FakeInterface", group=registry.ROBOT_INTERFACE_GROUP)
    monkeypatch.setattr(registry, "_ilm", SimpleNamespace(entry_points=lambda group: [ep]))
    monkeypatch.setattr(registry, "_instances", {})
    monkeypatch.setattr(registry, "_codec_state", {"done": False, "failure": None})


def _load(tmp_path, text):
    path = tmp_path / "c.yaml"
    path.write_text(text)
    return load_contract(path)


def test_contract_for_the_fake_interface_round_trips(fake_interface, tmp_path):
    contract = _load(tmp_path, CONTRACT)
    (obs,) = iter_observation_specs(contract)
    (act,) = iter_action_specs(contract)
    assert decode_value(FakeVector({"a": 1.0, "b": 2.0, "c": 9.0}), obs).tolist() == [1.0, 2.0]
    assert encode_value(np.array([3.0, 4.0]), act).fields == {"a": 3.0, "b": 4.0}


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("robot_interface: fake", "robot_interface: ros2", r"'ros2' is not installed\. Installed: fake"),
        ("type: fake/Vector, qos", "type: fake/Nope, qos", "Unknown message type 'fake/Nope'"),
        ("timeline: arrival}\n    select: [a, b]\nactions", "timeline: header}\n    select: [a, b]\nactions", "header"),
        ("qos: {level: 1}", "qos: {level: 7}", "qos must be"),
    ],
)
def test_the_interface_decides_what_loads(fake_interface, tmp_path, old, new, match):
    assert old in CONTRACT
    with pytest.raises(ContractValidationError, match=match):
        _load(tmp_path, CONTRACT.replace(old, new))
