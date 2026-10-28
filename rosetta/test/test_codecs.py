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

"""The codec registry's override discipline, inline codecs, the decode/encode gates,
and the operator pipeline around the codec.

decode_value and encode_value are the choke points between messages and
frames: every value passes their width, finiteness and None checks whatever
codec produced or consumes it. The gate tests run on the fake robot interface.
"""

import numpy as np
import pytest
from fake_interface import FakeVector

from rosetta.contract.schema import Align, Channel, Contract, FrameEntry, Source, load_contract
from rosetta.contract.specs import iter_action_specs, iter_observation_specs
from rosetta.frames.codecs import (
    DECODERS,
    DTYPES,
    ENCODERS,
    NonFiniteActionError,
    decode_value,
    encode_value,
    register_decoder,
    register_encoder,
)

pytestmark = pytest.mark.usefixtures("scratch_registries")

_T = "_test_msgs/msg/Foo"

_REGISTER = {
    "decoder": (lambda **kw: register_decoder(_T, dtype="float64", **kw), DECODERS),
    "encoder": (lambda **kw: register_encoder(_T, **kw), ENCODERS),
}


def _codec(result):
    return lambda *args: result


def _test_type_spec(role, select=("a",)):
    """Resolve one observation (role "decoder") or action (role "encoder") spec of type _T."""
    source = Source(channel=Channel(topic="/foo", type=_T), align=Align("hold", "arrival"), select=list(select) or None)
    sections = {"observations": (), "actions": ()}
    sections["observations" if role == "decoder" else "actions"] = (FrameEntry(key="k", sources=(source,)),)
    contract = Contract(
        robot_type="r",
        robot_interface="fake",
        fps=10,
        **sections,
        tasks=(),
        adjunct=(),
        rewards=(),
        signals=(),
        info=(),
        complementary_data=(),
        teleop=None,
    )
    (spec,) = (iter_observation_specs if role == "decoder" else iter_action_specs)(contract)
    return spec


# --- registration ------------------------------------------------------------


@pytest.mark.parametrize("role", sorted(_REGISTER))
def test_duplicate_registration_without_override_raises(role):
    register, _ = _REGISTER[role]
    register()(_codec(None))
    with pytest.raises(ValueError, match="already registered"):
        register()(_codec(None))


@pytest.mark.parametrize("role", sorted(_REGISTER))
def test_override_replaces(fake_interface, role):
    register, _ = _REGISTER[role]
    register()(_codec("first"))
    register(override=True)(_codec("second"))
    spec = _test_type_spec(role)
    result = decode_value(None, spec) if role == "decoder" else encode_value([1.0], spec)
    assert result == "second"


def test_decoder_override_updates_dtype(fake_interface):
    register_decoder(_T, dtype="float64")(_codec(None))
    register_decoder(_T, dtype="float32", override=True)(_codec(None))
    assert _test_type_spec("decoder").dtype == "float32"


@pytest.mark.parametrize("role", sorted(_REGISTER))
def test_override_without_target_raises(role):
    # A plugin that means to replace a built-in must fail loudly when that
    # built-in is renamed or removed, not silently register fresh.
    register, registry = _REGISTER[role]
    with pytest.raises(ValueError, match="nothing to override"):
        register(override=True)(_codec(None))
    assert _T not in registry


def test_override_clears_requires_select(fake_interface):
    # An override that no longer needs select must not inherit the flag
    # from the codec it replaced: a select-less channel of the type resolves.
    register_decoder(_T, dtype="float64", requires_select=True)(_codec(None))
    register_decoder(_T, dtype="float64", override=True)(_codec(None))
    assert _test_type_spec("decoder", select=()).names == ()


def test_register_decoder_rejects_invalid_dtype():
    # A typo'd dtype fails the plugin at import, not later as an obscure
    # spec-resolution or numpy error far from the mistake.
    with pytest.raises(ValueError, match="Invalid dtype"):
        register_decoder(_T, dtype="flaot64")(_codec(None))
    assert _T not in DECODERS and _T not in DTYPES


# --- decode/encode gates on the fake interface ----------------------------------


def _specs(tmp_path, *, decoder=None, encoder=None, apply=None, act_apply=None, obs_select="[a]"):
    obs_extra = f", decoder: '{decoder}'" if decoder else ""
    act_extra = f", encoder: '{encoder}'" if encoder else ""
    obs_select_line = f"    select: {obs_select}\n" if obs_select else ""
    apply_line = f"    apply: {apply}\n" if apply else ""
    act_apply_line = f"    apply: {act_apply}\n" if act_apply else ""
    path = tmp_path / "c.yaml"
    path.write_text(
        "robot_type: fake_bot\n"
        "robot_interface: fake\n"
        "fps: 10\n"
        "observations:\n"
        "  observation.state:\n"
        f"    channel: {{topic: /state, type: fake/Vector{obs_extra}}}\n"
        "    align: {strategy: hold, timeline: arrival}\n"
        f"{obs_select_line}"
        f"{apply_line}"
        "actions:\n"
        "  action:\n"
        f"    channel: {{topic: /cmd, type: fake/Vector{act_extra}}}\n"
        "    align: {strategy: hold, timeline: arrival}\n"
        "    select: [a, b]\n"
        f"{act_apply_line}"
    )
    contract = load_contract(path)
    return next(iter(iter_observation_specs(contract))), next(iter(iter_action_specs(contract)))


_inline_calls = []


def _inline_encoder(vec, spec, stamp_ns=None):
    _inline_calls.append(tuple(vec))
    return "inline-encoded"


def _str_decoder(msg, spec):
    return "not an array"


def _two_wide_decoder(msg, spec):
    return np.array([1.0, 2.0])


def _none_decoder(msg, spec):
    return None


def test_inline_encoder_wins_over_registry(fake_interface, tmp_path):
    _, act = _specs(tmp_path, encoder="test_codecs:_inline_encoder")
    assert encode_value([1.0, 2.0], act) == "inline-encoded"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_encode_refuses_non_finite_command(fake_interface, tmp_path, bad):
    _, act = _specs(tmp_path)
    with pytest.raises(NonFiniteActionError, match="'b'"):
        encode_value([0.5, bad], act)


def test_encode_refuses_nan_through_clamp(fake_interface, tmp_path):
    # clamp bounds the range but np.clip passes NaN through, so the finiteness
    # gate after the pipeline is what refuses the command.
    _, act = _specs(tmp_path, act_apply="[{clamp: {min: 0, max: 1}}]")
    with pytest.raises(NonFiniteActionError, match="'b'"):
        encode_value([0.5, float("nan")], act)


def test_finiteness_gate_fires_before_inline_encoder(fake_interface, tmp_path):
    # A custom encoder must never see a non-finite vector.
    _inline_calls.clear()
    _, act = _specs(tmp_path, encoder="test_codecs:_inline_encoder")
    with pytest.raises(NonFiniteActionError):
        encode_value([1.0, float("nan")], act)
    assert _inline_calls == []


def test_encode_rejects_width_mismatch(fake_interface, tmp_path):
    # select declares the vector width. Direct decode->encode callers (a teleop
    # passthrough) bypass FrameLayout, so this gate is their only guard.
    _, act = _specs(tmp_path)
    with pytest.raises(ValueError, match="3 values for its 2-wide"):
        encode_value([0.5, 1.0, 2.0], act)


def test_decode_rejects_none_result(fake_interface, tmp_path):
    # Pushing None would mark the stream warm while sampling still yields "missing".
    obs, _ = _specs(tmp_path, decoder="test_codecs:_none_decoder")
    with pytest.raises(ValueError, match="returned None"):
        decode_value(FakeVector({"a": 1.0}), obs)


def test_decode_rejects_non_array_with_operators(fake_interface, tmp_path):
    # A custom decoder whose real output disagrees with its dtype slips past
    # load-time validation; the declared pipeline must not be skipped silently.
    obs, _ = _specs(tmp_path, decoder="test_codecs:_str_decoder", apply="[{clamp: {min: 0, max: 1}}]")
    with pytest.raises(ValueError, match="numpy array"):
        decode_value(FakeVector({"a": 1.0}), obs)


@pytest.mark.parametrize(
    ("obs_select", "match"),
    [
        ("[a]", "2 values for its 1-wide spec. Width must match"),
        # Without select the spec is 1-wide: a multi-element message is refused.
        (None, "2 values for its 1-wide spec. A select-less stream is a scalar"),
    ],
    ids=["select", "select-less"],
)
def test_decode_rejects_width_mismatch(fake_interface, tmp_path, obs_select, match):
    obs, _ = _specs(tmp_path, decoder="test_codecs:_two_wide_decoder", obs_select=obs_select)
    with pytest.raises(ValueError, match=match):
        decode_value(FakeVector({"a": 1.0}), obs)


# --- the operator pipeline around the codec ------------------------------------


def test_decode_applies_operators_front_to_back(fake_interface, tmp_path):
    # rad2deg then clamp: [pi/2, pi] -> [90, 180] -> [90, 90]. Back-to-front
    # would clamp radians first and yield [90, 180].
    obs, _ = _specs(tmp_path, obs_select="[a, b]", apply="[rad2deg, {clamp: {min: 0, max: 90}}]")
    assert np.allclose(decode_value(FakeVector({"a": np.pi / 2, "b": np.pi}), obs), [90.0, 90.0])


def test_encode_applies_inverses_back_to_front(fake_interface, tmp_path):
    # The dataset speaks clamped degrees, so serving clamps in degrees first,
    # then converts deg->rad: 180 -> 90 -> pi/2. Front-to-back would convert
    # first and clamp pi (< 90) to itself.
    _, act = _specs(tmp_path, act_apply="[rad2deg, {clamp: {min: 0, max: 90}}]")
    msg = encode_value([180.0, 45.0], act)
    assert np.allclose([msg.fields["a"], msg.fields["b"]], [np.pi / 2, np.pi / 4])


def test_rad2deg_reaches_the_wire_in_radians(fake_interface, tmp_path):
    """The dataset speaks degrees; encode publishes radians and decode reads them back as degrees."""
    _, act = _specs(tmp_path, act_apply="[rad2deg]")
    msg = encode_value([90.0, 45.0], act)
    assert np.allclose([msg.fields["a"], msg.fields["b"]], [np.pi / 2, np.pi / 4])
    assert np.allclose(decode_value(msg, act), [90.0, 45.0])
