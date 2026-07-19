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

"""Load-time validation rules of spec resolution (specs.py), one test per rule.

Every rule here exists so a broken contract fails at load/resolution instead
of misbehaving at runtime (dropped messages, corrupted features, first-publish
crashes). Contracts are constructed directly from the schema dataclasses —
codec registration is deterministic (discover_codecs loads the built-ins), so
no registration imports are needed.
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
    TeleopFeedbackSource,
    TeleopInputSource,
)
from rosetta.contract.specs import (
    iter_action_specs,
    iter_extended_specs,
    iter_observation_specs,
    iter_reward_as_action_specs,
    iter_teleop_feedback_specs,
    iter_teleop_input_specs,
)

HOLD = Align("hold", "receive")


def _source(topic, msg_type, select=None, apply=(), **channel_kw):
    return Source(
        channel=Channel(topic=topic, type=msg_type, **channel_kw),
        align=HOLD,
        select=select,
        apply=tuple(apply),
    )


def _entry(key, *sources):
    return FrameEntry(key=key, sources=tuple(sources))


def _contract(observations=(), actions=(), rewards=(), signals=(), teleop=None):
    return Contract(
        robot_type="r",
        robot_interface="ros2",
        fps=30,
        observations=list(observations),
        actions=list(actions),
        tasks=[],
        adjunct=[],
        rewards=list(rewards),
        signals=list(signals),
        info=[],
        complementary_data=[],
        teleop=teleop,
    )


# ---------------------------------------------------------------------------
# Observation decodability + image dtype rules
# ---------------------------------------------------------------------------


def test_multi_source_image_key_rejected():
    c = _contract(
        observations=[
            _entry(
                "observation.images.cam",
                _source("/cam_a", "sensor_msgs/msg/Image", apply=[("resize", [4, 4])]),
                _source("/cam_b", "sensor_msgs/msg/Image", apply=[("resize", [4, 4])]),
            )
        ]
    )
    with pytest.raises(ContractValidationError, match="aggregate multiple channels under image key"):
        list(iter_observation_specs(c))


def test_undecodable_observation_rejected_at_load():
    c = _contract(observations=[_entry("observation.state", _source("/x", "my_msgs/msg/NoDecoder", ["a"]))])
    with pytest.raises(ContractValidationError, match="No decoder registered"):
        list(iter_observation_specs(c))


def test_explicit_dtype_does_not_exempt_decodability():
    """Regression: the old error advice said "specify dtype explicitly", but an
    undecodable channel is undecodable regardless of its declared dtype."""
    c = _contract(
        observations=[_entry("observation.state", _source("/x", "my_msgs/msg/NoDecoder", ["a"], dtype="float64"))]
    )
    with pytest.raises(ContractValidationError, match="No decoder registered"):
        list(iter_observation_specs(c))


def test_undecodable_custom_image_type_rejected_at_load():
    """Regression: is_image used to bypass the decodability check, deferring a
    custom image type without a decoder to a first-frame decode error."""
    c = _contract(
        observations=[
            _entry("observation.images.cam", _source("/cam", "my_msgs/msg/FancyImage", apply=[("resize", [4, 4])]))
        ]
    )
    with pytest.raises(ContractValidationError, match="No decoder registered"):
        list(iter_observation_specs(c))


@pytest.mark.parametrize("bad_dtype", ["float32", "bool", "image"])
def test_explicit_non_video_dtype_on_image_rejected(bad_dtype):
    """Regression: an explicit numeric dtype on an image key used to load and
    then declare a (1,) numeric feature while assemble produced (h, w, 3)
    uint8 — dataset corruption instead of a load error."""
    c = _contract(
        observations=[
            _entry(
                "observation.images.cam",
                _source("/cam", "sensor_msgs/msg/Image", apply=[("resize", [4, 4])], dtype=bad_dtype),
            )
        ]
    )
    with pytest.raises(ContractValidationError, match="conflicts with the image key"):
        list(iter_observation_specs(c))


def test_explicit_video_dtype_on_image_allowed():
    c = _contract(
        observations=[
            _entry(
                "observation.images.cam",
                _source("/cam", "sensor_msgs/msg/Image", apply=[("resize", [4, 4])], dtype="video"),
            )
        ]
    )
    (spec,) = iter_observation_specs(c)
    assert spec.dtype == "video"
    assert spec.is_image


def test_native_codec_dtype_tier():
    """No explicit dtype, no custom decoder -> the registered codec's dtype."""
    c = _contract(observations=[_entry("observation.state", _source("/v", "std_msgs/msg/Float32", ["data"]))])
    (spec,) = iter_observation_specs(c)
    assert spec.dtype == "float32"


def test_explicit_video_dtype_on_non_image_key_rejected():
    """'video' is reserved for observation.images.*; on a numeric key it used
    to load and then die at FrameLayout construction with a generic
    unsupported-dtype message that never named the actual mistake."""
    c = _contract(
        observations=[
            _entry("observation.state", _source("/v", "sensor_msgs/msg/JointState", ["position.j1"], dtype="video"))
        ]
    )
    with pytest.raises(ContractValidationError, match="reserved for observation.images"):
        list(iter_observation_specs(c))


# ---------------------------------------------------------------------------
# Extended sections decode at ingest -> same decodability rule
# ---------------------------------------------------------------------------


def test_undecodable_extended_channel_rejected_at_load():
    """Regression: extended sections used to skip the decodability check, so an
    undecodable signal loaded fine and silently dropped every message at ingest."""
    c = _contract(signals=[_entry("battery", _source("/battery", "my_msgs/msg/NoDecoder", ["v"], dtype="float64"))])
    with pytest.raises(ContractValidationError, match="No decoder registered"):
        list(iter_extended_specs(c))


def test_bool_signal_resolves_bool_dtype():
    """std_msgs/msg/Bool has a built-in decoder (stone.yaml's next.done relies on it)."""
    c = _contract(signals=[_entry("done", _source("/done", "std_msgs/msg/Bool", dtype="bool"))])
    (spec,) = iter_extended_specs(c)
    assert spec.dtype == "bool"


# ---------------------------------------------------------------------------
# Publishing sections need an encoder at load, not at first publish
# ---------------------------------------------------------------------------


def test_action_with_custom_encoder_only_type_allowed():
    """A type with only a custom encoder is legal on actions; the recorded
    column falls back to float64."""
    c = _contract(
        actions=[_entry("action", _source("/cmd", "my_msgs/msg/OnlyEncoded", ["a"], encoder="my_pkg.enc:encode"))]
    )
    (spec,) = iter_action_specs(c)
    assert spec.dtype == "float64"


def test_non_numeric_action_dtype_rejected():
    """Published streams are numeric vectors. A String-typed action channel
    (native dtype 'string') used to build a FrameLayout fine and then crash
    at the first publish_frame, inside split's float coercion."""
    c = _contract(
        actions=[_entry("action", _source("/note", "std_msgs/msg/String", encoder="my_pkg.enc:encode"))]
    )
    with pytest.raises(ContractValidationError, match="numeric vectors"):
        list(iter_action_specs(c))


def _teleop_feedback(feedback_source):
    obs = _entry("observation.state", _source("/joints", "sensor_msgs/msg/JointState", ["position.j1"]))
    teleop = Teleop(input=(), events=None, feedback=(TeleopFeedbackSource(source=feedback_source, origin="/joints"),))
    return _contract(observations=[obs], teleop=teleop)


def test_teleop_feedback_requires_encoder():
    """Regression: feedback skipped the encoder check its sibling publish paths
    (actions, reward-as-action) enforce, deferring failure to first publish in
    hil_manager_node."""
    c = _teleop_feedback(_source("/leader/fb", "my_msgs/msg/NoEncoder", ["a"]))
    with pytest.raises(ContractValidationError, match="No encoder registered"):
        list(iter_teleop_feedback_specs(c))


def test_teleop_feedback_custom_encoder_satisfies():
    c = _teleop_feedback(_source("/leader/fb", "my_msgs/msg/NoEncoder", ["a"], encoder="my_pkg.enc:encode"))
    (spec,) = iter_teleop_feedback_specs(c)
    assert spec.key == "teleop.feedback.observation.state"


# ---------------------------------------------------------------------------
# Reward-as-action projection rules
# ---------------------------------------------------------------------------


def test_reward_as_action_rejects_nonserveable_operator():
    """A reward used as a classifier action publishes, so its pipeline must run
    in the serve direction. This resolution-time check is the only gate:
    rewards parse with serveable=False rules. Uses a scoped FORWARD_ONLY dummy
    (the builtin one, resize, is image-only and now fails its own ctx gate
    first)."""
    from rosetta.contract.operators import OPERATOR_REGISTRY, Invertibility, Operator, register_operator

    @register_operator("_test_fwd_only_reward", kind=Invertibility.FORWARD_ONLY)
    class _FwdOnly(Operator):
        def forward(self, arr):
            return arr

    rew = _entry(
        "next.reward",
        _source("/reward", "std_msgs/msg/Float32", dtype="float32", apply=[("_test_fwd_only_reward", None)]),
    )
    c = _contract(rewards=[rew])
    try:
        with pytest.raises(ContractValidationError, match="no serve direction"):
            list(iter_reward_as_action_specs(c))
    finally:
        OPERATOR_REGISTRY.pop("_test_fwd_only_reward", None)


def test_reward_as_action_multi_entry_namespace_derivation():
    """Select-less rewards on distinct topics share the forced 'action' key but
    get distinguishing namespaces, so their synthesized 'data' names stay unique."""
    c = _contract(
        rewards=[
            _entry("next.reward", _source("/reward_a", "std_msgs/msg/Float32", dtype="float32")),
            _entry("next.bonus", _source("/reward_b", "std_msgs/msg/Float32", dtype="float32")),
        ]
    )
    specs = list(iter_reward_as_action_specs(c))
    assert [s.key for s in specs] == ["action", "action"]
    assert [s.names for s in specs] == [("data",), ("data",)]
    assert [s.namespace for s in specs] == ["reward_a", "reward_b"]


def test_reward_as_action_duplicate_selectless_names_rejected():
    """Two select-less rewards on ONE topic derive no namespace and would emit
    identical 'data' feature names — bypassing FrameLayout's shared-key select
    check because the name is synthesized. Rejected at resolution instead."""
    c = _contract(
        rewards=[
            _entry("next.reward", _source("/reward", "std_msgs/msg/Float32", dtype="float32")),
            _entry("next.bonus", _source("/reward", "std_msgs/msg/Float32", dtype="float32")),
        ]
    )
    with pytest.raises(ContractValidationError, match="collides"):
        list(iter_reward_as_action_specs(c))


# ---------------------------------------------------------------------------
# Teleop ownership resolution (direct construction bypasses schema validation)
# ---------------------------------------------------------------------------


def _teleop_input(target, actions):
    src = TeleopInputSource(
        source=_source("/leader/joints", "sensor_msgs/msg/JointState", ["position.j1"]), target=target
    )
    return _contract(actions=actions, teleop=Teleop(input=(src,), events=None, feedback=()))


def test_owning_key_unknown_topic_raises_contract_error():
    """A programmatically built contract with a dangling target surfaces as a
    contract problem, not an AssertionError."""
    c = _teleop_input("/nope", [_entry("action", _source("/cmd", "sensor_msgs/msg/JointState", ["position.j1"]))])
    with pytest.raises(ContractValidationError, match="no entry's sources declare"):
        list(iter_teleop_input_specs(c))


def test_owning_key_ambiguous_topic_raises():
    """One topic in two action entries makes the owning recording column
    ambiguous (and would diverge from hil_manager's last-wins topic mux)."""
    actions = [
        _entry("action.arm", _source("/cmd", "sensor_msgs/msg/JointState", ["position.j1"])),
        _entry("action.alt", _source("/cmd", "sensor_msgs/msg/JointState", ["velocity.j1"])),
    ]
    c = _teleop_input("/cmd", actions)
    with pytest.raises(ContractValidationError, match="belongs to multiple entries"):
        list(iter_teleop_input_specs(c))


def test_teleop_inputs_sharing_owning_key_get_namespaces():
    """Two targets inside one multi-source action entry share the synthesized
    'teleop.input.action' key; their leader-side names get distinguishing
    namespaces so flattened feature names stay unique in the teleop adapter."""
    action = _entry(
        "action",
        _source("/arm/cmd", "sensor_msgs/msg/JointState", ["position.j1"]),
        _source("/gripper/cmd", "std_msgs/msg/Float64", ["data"]),
    )
    teleop = Teleop(
        input=(
            TeleopInputSource(
                source=_source("/leader/arm/joints", "sensor_msgs/msg/JointState", ["position.j1"]), target="/arm/cmd"
            ),
            TeleopInputSource(
                source=_source("/leader/gripper/joints", "std_msgs/msg/Float64", ["data"]), target="/gripper/cmd"
            ),
        ),
        events=None,
        feedback=(),
    )
    c = _contract(actions=[action], teleop=teleop)
    specs = list(iter_teleop_input_specs(c))
    assert [s.key for s in specs] == ["teleop.input.action", "teleop.input.action"]
    assert [s.namespace for s in specs] == ["arm", "gripper"]


def test_single_teleop_input_gets_no_namespace():
    """The common case (one input per owning key) is unchanged: no namespace."""
    c = _teleop_input("/cmd", [_entry("action", _source("/cmd", "sensor_msgs/msg/JointState", ["position.j1"]))])
    (spec,) = iter_teleop_input_specs(c)
    assert spec.namespace is None


# ---------------------------------------------------------------------------
# Camera keys: core keeps dotted names first-class, no flattened-name guard
# ---------------------------------------------------------------------------


def test_core_accepts_flatten_colliding_camera_keys():
    """'cam.left' and 'cam_left' collide only once flattened, which only a backend
    whose sink needs flat identifiers does (WebDataset-style tars). Core keeps
    dotted names first-class and resolves both — the collision guard lives in that
    backend now, not here."""
    c = _contract(
        observations=[
            _entry(
                "observation.images.cam.left",
                _source("/c1", "sensor_msgs/msg/Image", apply=[("resize", [4, 4])]),
            ),
            _entry(
                "observation.images.cam_left",
                _source("/c2", "sensor_msgs/msg/Image", apply=[("resize", [4, 4])]),
            ),
        ]
    )
    assert len(list(iter_observation_specs(c))) == 2
