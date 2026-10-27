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

"""Spec-resolution rules that depend on the ros2 codecs' native dtypes.

String and Float32 resolve to dtypes other than float64, and Float32 decodes
and encodes without ``select``; the rest of spec resolution is tested in core
against the fake interface.
"""

import pytest

from rosetta.contract.errors import ContractValidationError
from rosetta.contract.schema import (
    Align,
    Channel,
    Contract,
    FrameEntry,
    Source,
    Teleop,
    TeleopInputSource,
)
from rosetta.contract.specs import (
    iter_action_specs,
    iter_observation_specs,
    iter_reward_as_action_specs,
    iter_specs,
)

HOLD = Align("hold", "receive")
STRING = "std_msgs/msg/String"
FLOAT32 = "std_msgs/msg/Float32"
JOINTS = "sensor_msgs/msg/JointState"


def _source(topic, msg_type, select=None, apply=(), **channel_kw):
    return Source(
        channel=Channel(topic=topic, type=msg_type, **channel_kw), align=HOLD, select=select, apply=tuple(apply)
    )


def _entry(key, *sources):
    return FrameEntry(key=key, sources=tuple(sources))


def _contract(observations=(), actions=(), rewards=(), teleop=None):
    return Contract(
        robot_type="r",
        robot_interface="ros2",
        fps=30,
        observations=tuple(observations),
        actions=tuple(actions),
        tasks=(),
        adjunct=(),
        rewards=tuple(rewards),
        signals=(),
        info=(),
        complementary_data=(),
        teleop=teleop,
    )


def test_native_codec_dtype_tier():
    """No explicit dtype, no custom decoder: the registered codec's dtype."""
    c = _contract(observations=[_entry("observation.state", _source("/v", FLOAT32, ["data"]))])
    (spec,) = iter_observation_specs(c)
    assert spec.dtype == "float32"


def test_string_stream_with_apply_rejected():
    """Operators transform numeric arrays; `apply` on a string stream fails at
    resolution, not by silently skipping the pipeline at decode time."""
    status = _source("/status", STRING, apply=[("clamp", {"min": 0.0, "max": 1.0})])
    c = _contract(observations=[_entry("observation.environment_state", status)])
    with pytest.raises(ContractValidationError, match="string"):
        list(iter_observation_specs(c))


def test_non_numeric_action_dtype_rejected():
    """Published streams are numeric vectors. A String action channel fails at
    load, not at the first publish_frame inside split's float coercion."""
    c = _contract(actions=[_entry("action", _source("/note", STRING, encoder="my_pkg.enc:encode"))])
    with pytest.raises(ContractValidationError, match="numeric vectors"):
        list(iter_action_specs(c))


def test_teleop_input_non_numeric_dtype_rejected():
    """A String teleop input decodes to str, which the HIL forwarder would try
    to re-encode onto the numeric action topic on every message."""
    act = _entry("action", _source("/cmd", JOINTS, ["position.j1"]))
    teleop = Teleop(input=(TeleopInputSource(source=_source("/leader/text", STRING), target="/cmd"),), events=None)
    with pytest.raises(ContractValidationError, match="numeric action topic"):
        list(iter_specs(_contract(actions=[act], teleop=teleop)))


def test_reward_as_action_multi_entry_namespace_derivation():
    """Select-less rewards on distinct topics share the forced 'action' key but
    get distinguishing namespaces, so their synthesized 'data' names stay unique."""
    c = _contract(
        rewards=[
            _entry("next.reward", _source("/reward_a", FLOAT32, dtype="float32")),
            _entry("next.bonus", _source("/reward_b", FLOAT32, dtype="float32")),
        ]
    )
    specs = list(iter_reward_as_action_specs(c))
    assert [(s.key, s.names, s.namespace) for s in specs] == [
        ("action", ("data",), "reward_a"),
        ("action", ("data",), "reward_b"),
    ]
    assert [s.dtype for s in specs] == ["float32", "float32"]


def test_reward_as_action_duplicate_selectless_names_rejected():
    """Two select-less rewards on one topic derive no namespace and would emit
    identical 'data' feature names, bypassing FrameLayout's shared-key select
    check because the name is synthesized."""
    c = _contract(
        rewards=[
            _entry("next.reward", _source("/reward", FLOAT32, dtype="float32")),
            _entry("next.bonus", _source("/reward", FLOAT32, dtype="float32")),
        ]
    )
    with pytest.raises(ContractValidationError, match="collides"):
        list(iter_reward_as_action_specs(c))
