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

"""Spec resolution (specs.py): computed fields derive correctly, and every rule
that keeps a broken contract from misbehaving at runtime fails at resolution.

Contracts are built from the schema dataclasses against the fake interface,
whose ``fake/Vector`` has a decoder and an encoder that both need ``select``;
``fake/Nope`` has neither. Image rules key off ``observation.images.*``, not
the message type, so they hold for any decodable type.
"""

import pytest

from rosetta.contract.errors import ContractValidationError
from rosetta.contract.operators import OPERATOR_REGISTRY, Invertibility, Operator, register_operator
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
    _derive_namespaces,
    iter_action_specs,
    iter_observation_specs,
    iter_policy_specs,
    iter_reward_as_action_specs,
    iter_specs,
    iter_teleop_feedback_specs,
    iter_teleop_input_specs,
)

pytestmark = pytest.mark.usefixtures("fake_interface")

VECTOR = "fake/Vector"
NOPE = "fake/Nope"
HOLD = Align("hold", "arrival")
RESIZE = [("resize", [4, 4])]


def _source(topic, msg_type=VECTOR, select=("a",), apply=(), align=HOLD, **channel_kw):
    return Source(
        channel=Channel(topic=topic, type=msg_type, **channel_kw),
        align=align,
        select=list(select) if select else None,
        apply=tuple(apply),
    )


def _image(topic, msg_type=VECTOR, apply=RESIZE, **channel_kw):
    return _source(topic, msg_type, select=(), apply=apply, **channel_kw)


def _entry(key, *sources):
    return FrameEntry(key=key, sources=tuple(sources))


def _contract(observations=(), actions=(), rewards=(), signals=(), teleop=None, fps=30):
    return Contract(
        robot_type="r",
        robot_interface="fake",
        fps=fps,
        observations=tuple(observations),
        actions=tuple(actions),
        tasks=(),
        adjunct=(),
        rewards=tuple(rewards),
        signals=tuple(signals),
        info=(),
        complementary_data=(),
        teleop=teleop,
    )


def _resolve(iterator, contract):
    return list(iterator(contract))


@pytest.fixture
def scoped_operator():
    """Register a test operator class under a name; unregister it afterwards."""
    names = []

    def register(name, cls, kind):
        names.append(name)
        register_operator(name, kind=kind)(cls)

    yield register
    for name in names:
        OPERATOR_REGISTRY.pop(name, None)


# ---------------------------------------------------------------------------
# Computed fields
# ---------------------------------------------------------------------------


def test_observation_computed_fields_derive():
    state_src = _source(
        "/joint_states", select=("shoulder", "elbow"), apply=[("rad2deg", None)], dtype="float32", decoder="m:decode"
    )
    contract = _contract(
        observations=[_entry("observation.state", state_src), _entry("observation.images.wrist", _image("/wrist"))],
        fps=37,
    )
    state, image = _resolve(iter_observation_specs, contract)

    assert state.key == "observation.state"
    assert state.names == ("shoulder", "elbow")
    assert state.fps == 37
    assert state.dtype == "float32"  # explicit channel dtype outranks the custom decoder
    assert [o.name for o in state.operators] == ["rad2deg"]
    assert (state.is_image, image.is_image) == (False, True)
    assert image.image_resize == (4, 4)
    assert image.dtype == "video"


def test_action_computed_fields_and_namespaces():
    action = _entry(
        "action",
        _source("/cmd", NOPE, select=("x0", "x1"), decoder="m:decode", encoder="m:encode"),
        _source("/gripper_cmd", select=("g",)),
    )
    arm, gripper = _resolve(iter_action_specs, _contract(actions=[action]))
    assert (arm.names, gripper.names) == (("x0", "x1"), ("g",))
    assert arm.dtype == "float64"  # custom-decoder tier
    assert (arm.namespace, gripper.namespace) == ("cmd", "gripper_cmd")


@pytest.mark.parametrize(
    ("topics", "expected"),
    [
        # First-segment tier: earliest segment index where all topics differ.
        (["/arm/state", "/base/state"], {"/arm/state": "arm", "/base/state": "base"}),
        (["/arm/pos", "/arm/vel"], {"/arm/pos": "pos", "/arm/vel": "vel"}),
        # Uneven depths: the short topic gets an empty segment at the winning index.
        (["/cmd_vel", "/cmd_vel/filtered"], {"/cmd_vel": "", "/cmd_vel/filtered": "filtered"}),
        (["/only"], {"/only": ""}),
        # No single segment distinguishes all three: falls through to the depth-2 compound tier.
        (["/a/x/1", "/a/y/1", "/b/x/2"], {"/a/x/1": "a.x", "/a/y/1": "a.y", "/b/x/2": "b.x"}),
    ],
)
def test_derive_namespaces_tiers(topics, expected):
    assert _derive_namespaces(topics) == expected


def test_derive_namespaces_identical_normalization_rejected():
    """Distinct topic strings that normalize to the same segments ('/a/b' vs
    'a/b') have no unique namespace at any tier: a load error, not silent
    duplicate feature names."""
    with pytest.raises(ContractValidationError, match="identical paths"):
        _derive_namespaces(["/a/b", "a/b"])


# ---------------------------------------------------------------------------
# Operators and image geometry
# ---------------------------------------------------------------------------


def test_third_party_operator_with_output_hw_satisfies_image_gate(scoped_operator):
    """The image-geometry gate reads Operator.output_hw, not the name 'resize',
    so any plugin declaring a fixed output size can fulfill it."""

    class _Scale(Operator):
        def __init__(self, args, ctx):
            del args, ctx
            self.output_hw = (8, 8)

        def forward(self, arr):
            return arr

    scoped_operator("_test_scale", _Scale, Invertibility.FORWARD_ONLY)
    contract = _contract(observations=[_entry("observation.images.wrist", _image("/w", apply=[("_test_scale", None)]))])
    (spec,) = _resolve(iter_observation_specs, contract)
    assert spec.image_resize == (8, 8)


def test_last_geometry_operator_wins():
    """Pipelines run front-to-back, so the last declared output_hw is the stream's final geometry."""
    image = _image("/w", apply=[("resize", [48, 64]), ("resize", [24, 32])])
    (spec,) = _resolve(iter_observation_specs, _contract(observations=[_entry("observation.images.wrist", image)]))
    assert spec.image_resize == (24, 32)


def test_resize_on_non_image_stream_rejected():
    """resize on a state vector would crash on every message at runtime; the
    operator's ctx.is_image gate turns it into a resolution error."""
    contract = _contract(observations=[_entry("observation.state", _source("/s", apply=RESIZE))])
    with pytest.raises(ContractValidationError, match="image"):
        _resolve(iter_observation_specs, contract)


def test_multi_source_image_key_rejected():
    contract = _contract(observations=[_entry("observation.images.cam", _image("/cam_a"), _image("/cam_b"))])
    with pytest.raises(ContractValidationError, match="aggregate multiple channels under image key"):
        _resolve(iter_observation_specs, contract)


# ---------------------------------------------------------------------------
# Decodability and dtype rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "contract",
    [
        pytest.param(_contract(observations=[_entry("observation.state", _source("/x", NOPE))]), id="plain"),
        pytest.param(
            _contract(observations=[_entry("observation.state", _source("/x", NOPE, dtype="float64"))]),
            id="explicit-dtype",
        ),
        pytest.param(_contract(observations=[_entry("observation.images.cam", _image("/cam", NOPE))]), id="image-key"),
        pytest.param(_contract(signals=[_entry("battery", _source("/b", NOPE, dtype="float64"))]), id="signals"),
    ],
)
def test_undecodable_observation_rejected(contract):
    """Every stream decoded at ingest must be decodable, whatever its dtype or
    key; otherwise it loads and silently drops every message."""
    with pytest.raises(ContractValidationError, match="No decoder registered"):
        _resolve(iter_specs, contract)


@pytest.mark.parametrize("bad_dtype", ["float32", "bool"])
def test_explicit_non_video_dtype_on_image_rejected(bad_dtype):
    """Otherwise the feature is declared numeric while assemble produces
    (h, w, 3) uint8, corrupting the dataset."""
    contract = _contract(observations=[_entry("observation.images.cam", _image("/cam", dtype=bad_dtype))])
    with pytest.raises(ContractValidationError, match="conflicts with the image key"):
        _resolve(iter_observation_specs, contract)


def test_explicit_video_dtype_on_image_allowed():
    contract = _contract(observations=[_entry("observation.images.cam", _image("/cam", dtype="video"))])
    (spec,) = _resolve(iter_observation_specs, contract)
    assert (spec.dtype, spec.is_image) == ("video", True)


@pytest.mark.parametrize("section", ["observations", "signals"])
def test_explicit_video_dtype_on_non_image_key_rejected(section):
    """'video' is reserved for observation.images.*; on a numeric key it fails
    with an error naming that rule, not later with a generic dtype error."""
    contract = _contract(**{section: [_entry("observation.state", _source("/v", dtype="video"))]})
    with pytest.raises(ContractValidationError, match="reserved for observation.images"):
        _resolve(iter_specs, contract)


# ---------------------------------------------------------------------------
# Publishing streams need an encoder at load, not at first publish
# ---------------------------------------------------------------------------


def test_action_with_custom_encoder_only_type_allowed():
    """A type with only a custom encoder is legal on actions; the recorded column falls back to float64."""
    contract = _contract(actions=[_entry("action", _source("/cmd", NOPE, encoder="m:encode"))])
    (spec,) = _resolve(iter_action_specs, contract)
    assert spec.dtype == "float64"


def _teleop_feedback(feedback_source):
    teleop = Teleop(input=(), events=None, feedback=(TeleopFeedbackSource(source=feedback_source, origin="/joints"),))
    return _contract(observations=[_entry("observation.state", _source("/joints"))], teleop=teleop)


def test_teleop_feedback_requires_encoder():
    with pytest.raises(ContractValidationError, match="No encoder registered"):
        _resolve(iter_teleop_feedback_specs, _teleop_feedback(_source("/leader/fb", NOPE)))


def test_teleop_feedback_custom_encoder_satisfies():
    (spec,) = _resolve(iter_teleop_feedback_specs, _teleop_feedback(_source("/leader/fb", NOPE, encoder="m:encode")))
    assert spec.key == "teleop.feedback.observation.state"


# ---------------------------------------------------------------------------
# Section views and the reward-as-action projection
# ---------------------------------------------------------------------------


def test_policy_specs_exclude_record_only_sections():
    """Rewards and teleop input/feedback are record-only columns: iter_specs
    carries them, iter_policy_specs (what the policy I/O sees) must not."""
    teleop = Teleop(
        input=(TeleopInputSource(source=_source("/leader/joints"), target="/cmd"),),
        events=None,
        feedback=(TeleopFeedbackSource(source=_source("/leader/feedback"), origin="/joints"),),
    )
    contract = _contract(
        observations=[_entry("observation.state", _source("/joints"))],
        actions=[_entry("action", _source("/cmd"))],
        rewards=[_entry("next.reward", _source("/reward", dtype="float32"))],
        teleop=teleop,
    )
    all_keys = {s.key for s in iter_specs(contract)}
    policy_keys = {s.key for s in iter_policy_specs(contract)}
    assert all_keys - policy_keys == {"next.reward", "teleop.input.action", "teleop.feedback.observation.state"}
    assert policy_keys == {"observation.state", "action"}


def test_reward_as_action_requires_registered_encoder_even_with_decoder():
    """Rewards publish through the encoder registry (no custom encoder), so a
    custom decoder must not exempt an unregistered type."""
    contract = _contract(rewards=[_entry("next.reward", _source("/r", NOPE, dtype="float32", decoder="m:decode"))])
    with pytest.raises(ContractValidationError, match="No encoder registered"):
        _resolve(iter_reward_as_action_specs, contract)


def test_reward_as_action_rejects_empty_rewards_section():
    """With no rewards, zero action specs would silently discard every classifier output."""
    with pytest.raises(ContractValidationError, match="rewards"):
        _resolve(iter_reward_as_action_specs, _contract())


def test_reward_as_action_rejects_nonserveable_operator(scoped_operator):
    """A reward used as a classifier action publishes, so its pipeline must run
    in the serve direction. This is the only gate: rewards parse with
    serveable=False rules."""

    class _FwdOnly(Operator):
        def forward(self, arr):
            return arr

    scoped_operator("_test_fwd_only_reward", _FwdOnly, Invertibility.FORWARD_ONLY)
    reward = _source("/reward", dtype="float32", apply=[("_test_fwd_only_reward", None)])
    with pytest.raises(ContractValidationError, match="no serve direction"):
        _resolve(iter_reward_as_action_specs, _contract(rewards=[_entry("next.reward", reward)]))


# ---------------------------------------------------------------------------
# Teleop columns are named like the entries they drive or mirror
# ---------------------------------------------------------------------------


def test_teleop_inputs_sharing_owning_key_get_namespaces():
    """Two targets inside one multi-source action entry share the synthesized
    'teleop.input.action' key; each takes its target's namespace, so the names
    stay unique and match the action's."""
    action = _entry("action", _source("/arm/cmd"), _source("/gripper/cmd"))
    teleop = Teleop(
        input=(
            TeleopInputSource(source=_source("/leader/arm"), target="/arm/cmd"),
            TeleopInputSource(source=_source("/leader/gripper"), target="/gripper/cmd"),
        ),
        events=None,
        feedback=(),
    )
    specs = _resolve(iter_teleop_input_specs, _contract(actions=[action], teleop=teleop))
    assert [(s.key, s.namespace) for s in specs] == [("teleop.input.action", "arm"), ("teleop.input.action", "gripper")]


def test_teleop_feedback_takes_its_origins_namespace():
    state = _entry("observation.state", _source("/arm/joint_states"), _source("/gripper/joint_states"))
    feedback = TeleopFeedbackSource(source=_source("/leader/feedback"), origin="/arm/joint_states")
    contract = _contract(observations=[state], teleop=Teleop(input=(), events=None, feedback=(feedback,)))
    (spec,) = _resolve(iter_teleop_feedback_specs, contract)
    assert spec.namespace == "arm"


def test_single_teleop_input_gets_no_namespace():
    """A single-source target has no namespace, so neither does its input."""
    teleop = Teleop(input=(TeleopInputSource(source=_source("/leader"), target="/cmd"),), events=None, feedback=())
    (spec,) = _resolve(iter_teleop_input_specs, _contract(actions=[_entry("action", _source("/cmd"))], teleop=teleop))
    assert spec.namespace is None
