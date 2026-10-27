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

"""The codec registry's override discipline, inline codecs, and the decode/encode gates.

decode_value and encode_value are the choke points between messages and
frames: every value passes their width, finiteness and None checks whatever
codec produced or consumes it. The gate tests run on the fake robot interface.
"""

import numpy as np
import pytest
from fake_interface import FakeVector

from rosetta.contract.schema import load_contract
from rosetta.contract.specs import iter_action_specs, iter_observation_specs
from rosetta.frames import codecs
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

_T = "_test_msgs/msg/Foo"

_REGISTER = {
    "decoder": (lambda **kw: register_decoder(_T, dtype="float64", **kw), DECODERS),
    "encoder": (lambda **kw: register_encoder(_T, **kw), ENCODERS),
}


@pytest.fixture(autouse=True)
def _clear_test_type():
    yield
    DECODERS.pop(_T, None)
    DTYPES.pop(_T, None)
    ENCODERS.pop(_T, None)
    codecs.DECODER_REQUIRES_SELECT.discard(_T)
    codecs.ENCODER_REQUIRES_SELECT.discard(_T)


def _codec(result):
    return lambda *args: result


# --- registration ------------------------------------------------------------


@pytest.mark.parametrize("role", sorted(_REGISTER))
def test_duplicate_registration_without_override_raises(role):
    register, _ = _REGISTER[role]
    register()(_codec(None))
    with pytest.raises(ValueError, match="already registered"):
        register()(_codec(None))


@pytest.mark.parametrize("role", sorted(_REGISTER))
def test_override_replaces(role):
    register, registry = _REGISTER[role]
    register()(_codec("first"))
    register(override=True)(_codec("second"))
    assert registry[_T](None, None) == "second"


def test_decoder_override_updates_dtype():
    register_decoder(_T, dtype="float64")(_codec(None))
    register_decoder(_T, dtype="float32", override=True)(_codec(None))
    assert DTYPES[_T] == "float32"


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
    # from the codec it replaced.
    register_decoder(_T, dtype="float64", requires_select=True)(_codec(None))
    register_decoder(_T, dtype="float64", override=True)(_codec(None))
    assert not codecs.decoder_requires_select(_T)


def test_register_decoder_rejects_invalid_dtype():
    # A typo'd dtype fails the plugin at import, not later as an obscure
    # spec-resolution or numpy error far from the mistake.
    with pytest.raises(ValueError, match="Invalid dtype"):
        register_decoder(_T, dtype="flaot64")(_codec(None))
    assert _T not in DECODERS and _T not in DTYPES


@pytest.mark.parametrize("path", ["nofunc", "m:", ":f"])
def test_load_codec_rejects_malformed_path(path):
    with pytest.raises(ValueError, match="Expected format"):
        codecs.load_codec(path)


# --- decode/encode gates on the fake interface ----------------------------------


def _specs(tmp_path, *, decoder=None, encoder=None, apply=None):
    obs_extra = f", decoder: '{decoder}'" if decoder else ""
    act_extra = f", encoder: '{encoder}'" if encoder else ""
    apply_line = f"    apply: {apply}\n" if apply else ""
    path = tmp_path / "c.yaml"
    path.write_text(
        "robot_type: fake_bot\n"
        "robot_interface: fake\n"
        "fps: 10\n"
        "observations:\n"
        "  observation.state:\n"
        f"    channel: {{topic: /state, type: fake/Vector{obs_extra}}}\n"
        "    align: {strategy: hold, timeline: arrival}\n"
        "    select: [a]\n"
        f"{apply_line}"
        "actions:\n"
        "  action:\n"
        f"    channel: {{topic: /cmd, type: fake/Vector{act_extra}}}\n"
        "    align: {strategy: hold, timeline: arrival}\n"
        "    select: [a, b]\n"
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


def test_decode_rejects_width_mismatch(fake_interface, tmp_path):
    obs, _ = _specs(tmp_path, decoder="test_codecs:_two_wide_decoder")
    with pytest.raises(ValueError, match="2 values for its 1-wide"):
        decode_value(FakeVector({"a": 1.0}), obs)
