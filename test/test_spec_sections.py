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

"""Guard: the VLA writers consume observations and actions only, never the
extended sections (reward, signal, info, complementary_data) that the porter's
iter_specs also yields. This keeps RL and record-only columns out of
observation.state."""

import pytest
from rosetta.contract.errors import ContractValidationError
from rosetta.contract.schema import (
    Align,
    Channel,
    Contract,
    FrameEntry,
    Source,
    Teleop,
    TeleopFeedbackSource,
    TeleopInputSource,
)
from rosetta.contract.specs import (
    iter_action_specs,
    iter_observation_specs,
    iter_policy_specs,
    iter_reward_as_action_specs,
    iter_specs,
)

HOLD = Align("hold", "receive")


def _entry(key, topic, msg_type, select=None, **channel_kw):
    return FrameEntry(
        key=key,
        sources=(
            Source(
                channel=Channel(topic=topic, type=msg_type, **channel_kw),
                align=HOLD,
                select=select,
            ),
        ),
    )


def _contract(observations=(), actions=(), rewards=(), teleop=None):
    return Contract(
        robot_type="r",
        robot_interface="ros2",
        fps=30,
        observations=list(observations),
        actions=list(actions),
        tasks=[],
        adjunct=[],
        rewards=list(rewards),
        signals=[],
        info=[],
        complementary_data=[],
        teleop=teleop,
    )


def _contract_with_reward():
    obs = _entry("observation.state", "/joints", "sensor_msgs/msg/JointState", ["position.j1", "position.j2"])
    act = _entry("action", "/cmd", "sensor_msgs/msg/JointState", ["position.j1", "position.j2"])
    # A reward whose key classifies as state ('next.reward' is not image or action).
    rew = _entry("next.reward", "/reward", "std_msgs/msg/Float32", ["data"], dtype="float32")
    return _contract([obs], [act], [rew])


def test_reward_leaks_into_iter_specs_but_not_obs_action():
    c = _contract_with_reward()
    all_keys = {s.key for s in iter_specs(c)}
    oa_keys = {s.key for s in iter_observation_specs(c)} | {s.key for s in iter_action_specs(c)}

    assert "next.reward" in all_keys  # the porter's iter_specs carries it
    assert "next.reward" not in oa_keys  # but the VLA writers (obs+action) don't
    assert oa_keys == {"observation.state", "action"}


def test_reward_as_action_requires_registered_encoder_even_with_decoder():
    """Regression: the encoder check was gated on `decoder is None`, so a
    reward with a custom decoder and an unregistered type passed contract
    load and crashed at first publish (rewards always publish through the
    built-in encoder registry; custom encoders are not supported there)."""
    rew = _entry(
        "next.reward",
        "/reward",
        "my_msgs/msg/NoEncoder",
        ["data"],
        dtype="float32",
        decoder="my_pkg.decoders:decode_reward",
    )
    c = _contract(rewards=[rew])
    with pytest.raises(ContractValidationError, match="No encoder registered"):
        list(iter_reward_as_action_specs(c))


def test_reward_as_action_registered_type_passes():
    c = _contract_with_reward()  # std_msgs/msg/Float32 has a registered encoder
    specs = list(iter_reward_as_action_specs(c))
    assert len(specs) == 1
    assert specs[0].key == "action"
    assert specs[0].dtype == "float32"  # explicit dtype honored


def test_reward_as_action_rejects_empty_rewards_section():
    """Regression: is_classifier=True with no 'rewards' entries used to
    resolve to zero action specs with no error, silently discarding every
    policy output instead of failing at spec-resolution time."""
    c = _contract()  # no rewards
    with pytest.raises(ContractValidationError, match="rewards"):
        list(iter_reward_as_action_specs(c))


def _source(topic, msg_type, select=None, **channel_kw):
    return Source(channel=Channel(topic=topic, type=msg_type, **channel_kw), align=HOLD, select=select)


def test_teleop_leaks_into_iter_specs_but_not_policy_specs():
    """Teleop input/feedback are record-only diagnostic columns (like the
    extended sections): the porter's iter_specs carries them, keyed by the
    action/observation they target (target='/cmd' -> 'teleop.input.action'),
    but iter_policy_specs (observations+actions only) must not, since they
    duplicate/don't belong in a policy's I/O vectors."""
    obs = _entry("observation.state", "/joints", "sensor_msgs/msg/JointState", ["position.j1"])
    act = _entry("action", "/cmd", "sensor_msgs/msg/JointState", ["position.j1"])
    teleop = Teleop(
        input=(
            TeleopInputSource(
                source=_source("/leader/joints", "sensor_msgs/msg/JointState", ["position.j1"]), target="/cmd"
            ),
        ),
        events=None,
        feedback=(
            TeleopFeedbackSource(
                source=_source("/leader/feedback", "sensor_msgs/msg/JointState", ["effort.j1"]), origin="/joints"
            ),
        ),
    )
    c = _contract([obs], [act], teleop=teleop)

    all_keys = {s.key for s in iter_specs(c)}
    policy_keys = {s.key for s in iter_policy_specs(c)}

    assert {"teleop.input.action", "teleop.feedback.observation.state"} <= all_keys
    assert not ({"teleop.input.action", "teleop.feedback.observation.state"} & policy_keys)
    assert policy_keys == {"observation.state", "action"}


def test_teleop_input_non_numeric_dtype_rejected():
    # A String teleop input decodes to str, which hil_manager_node would then
    # try to re-encode onto the numeric action topic — failing on every
    # message at teleop rate. Reject at load instead.
    obs = _entry("observation.state", "/joints", "sensor_msgs/msg/JointState", ["position.j1"])
    act = _entry("action", "/cmd", "sensor_msgs/msg/JointState", ["position.j1"])
    teleop = Teleop(
        input=(TeleopInputSource(source=_source("/leader/text", "std_msgs/msg/String"), target="/cmd"),),
        events=None,
        feedback=(),
    )
    c = _contract([obs], [act], teleop=teleop)
    with pytest.raises(ContractValidationError, match="numeric action topic"):
        list(iter_specs(c))


def test_iter_specs_omits_teleop_when_contract_has_none():
    """No teleop section (the common case) must not raise or synthesize keys."""
    c = _contract()
    assert {s.key for s in iter_specs(c)} == set()
