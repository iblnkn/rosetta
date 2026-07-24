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

"""Codec registry behavior: override semantics + entry-point discovery.

Pure (no ROS): registration and discovery just populate dicts.
"""

import types

import pytest

from rosetta.contract.errors import ContractValidationError
from rosetta.frames import codecs
from rosetta.frames.codecs import (
    DECODER_REQUIRES_SELECT,
    DECODERS,
    DTYPES,
    ENCODER_REQUIRES_SELECT,
    ENCODERS,
    NonFiniteActionError,
    register_decoder,
    register_encoder,
)

_T = "_test_msgs/msg/Foo"


def _clear(type_str):
    DECODERS.pop(type_str, None)
    DTYPES.pop(type_str, None)
    ENCODERS.pop(type_str, None)
    DECODER_REQUIRES_SELECT.discard(type_str)
    ENCODER_REQUIRES_SELECT.discard(type_str)


# --- override semantics ---------------------------------------------------


def test_duplicate_decoder_without_override_raises():
    try:
        register_decoder(_T, dtype="float64")(lambda msg, spec: None)
        with pytest.raises(ValueError, match="already registered"):
            register_decoder(_T, dtype="float64")(lambda msg, spec: None)
    finally:
        _clear(_T)


def test_decoder_override_replaces():
    try:
        register_decoder(_T, dtype="float64")(lambda msg, spec: "first")
        register_decoder(_T, dtype="float32", override=True)(lambda msg, spec: "second")
        assert DECODERS[_T](None, None) == "second"
        assert DTYPES[_T] == "float32"  # dtype updated too
    finally:
        _clear(_T)


def test_duplicate_encoder_without_override_raises():
    try:
        register_encoder(_T)(lambda vec, spec, stamp=None: None)
        with pytest.raises(ValueError, match="already registered"):
            register_encoder(_T)(lambda vec, spec, stamp=None: None)
    finally:
        _clear(_T)


def test_encoder_override_replaces():
    try:
        register_encoder(_T)(lambda vec, spec, stamp=None: "first")
        register_encoder(_T, override=True)(lambda vec, spec, stamp=None: "second")
        assert ENCODERS[_T](None, None) == "second"
    finally:
        _clear(_T)


def test_decoder_override_without_target_raises():
    # A plugin that means to replace a built-in must fail loudly when that
    # built-in is renamed or removed, not silently register fresh.
    with pytest.raises(ValueError, match="nothing to override"):
        register_decoder(_T, dtype="float64", override=True)(lambda msg, spec: None)
    assert _T not in DECODERS


def test_encoder_override_without_target_raises():
    with pytest.raises(ValueError, match="nothing to override"):
        register_encoder(_T, override=True)(lambda vec, spec, stamp=None: None)
    assert _T not in ENCODERS


def test_requires_select_recorded_and_cleared_on_override():
    # An override that no longer needs select must not inherit the flag
    # from the codec it replaced.
    try:
        register_decoder(_T, dtype="float64", requires_select=True)(lambda msg, spec: None)
        assert codecs.decoder_requires_select(_T)
        register_decoder(_T, dtype="float64", override=True)(lambda msg, spec: None)
        assert not codecs.decoder_requires_select(_T)
        register_encoder(_T, requires_select=True)(lambda vec, spec, stamp=None: None)
        assert codecs.encoder_requires_select(_T)
    finally:
        _clear(_T)


def test_register_decoder_rejects_invalid_dtype():
    # A typo'd dtype fails the plugin at import, not later as an obscure
    # spec-resolution or numpy error far from the mistake.
    with pytest.raises(ValueError, match="Invalid dtype"):
        register_decoder(_T, dtype="flaot64")(lambda msg, spec: None)
    assert _T not in DECODERS and _T not in DTYPES


# --- inline codec loading ----------------------------------------------


@pytest.mark.parametrize("path", ["nofunc", "m:", ":f"])
def test_load_codec_rejects_malformed_path(path):
    with pytest.raises(ValueError, match="Expected format"):
        codecs.load_codec(path)


def test_load_codec_missing_module_raises_importerror():
    with pytest.raises(ImportError):
        codecs.load_codec("no_such_module_xyz:fn")


def test_load_codec_missing_function_raises_attributeerror():
    with pytest.raises(AttributeError):
        codecs.load_codec("json:nope")


def test_load_codec_resolves_function():
    import json

    assert codecs.load_codec("json:loads") is json.loads


# --- inline encoder dispatch ------------------------------------------------


def _action_spec_ns(encoder=None):
    """Minimal duck-typed ActionStreamSpec: exactly what encode_value reads."""
    channel = types.SimpleNamespace(type=_T, topic="/t", encoder=encoder, decoder=None)
    source = types.SimpleNamespace(channel=channel)
    return types.SimpleNamespace(key="action", names=("a", "b"), dim=2, operators=(), source=source)


_sentinel_calls = []


def _sentinel_encoder(vec, spec, stamp_ns=None):
    _sentinel_calls.append(tuple(vec))
    return "inline-encoded"


def test_inline_encoder_wins_over_registry():
    try:
        register_encoder(_T)(lambda vec, spec, stamp=None: "registry-encoded")
        spec = _action_spec_ns(encoder="test_codec_registry:_sentinel_encoder")
        assert codecs.encode_value([1.0, 2.0], spec) == "inline-encoded"
    finally:
        _clear(_T)


def test_finiteness_gate_fires_before_inline_encoder():
    # The gate is THE choke point: a custom encoder must never see a
    # non-finite vector, so it runs before inline dispatch too.
    _sentinel_calls.clear()
    spec = _action_spec_ns(encoder="test_codec_registry:_sentinel_encoder")
    with pytest.raises(NonFiniteActionError):
        codecs.encode_value([1.0, float("nan")], spec)
    assert _sentinel_calls == []


# --- entry-point discovery ------------------------------------------------


def _fake_ilm(monkeypatch, entry_points):
    """Point the shared plugin loader at a fake entry-point list, latches cleared."""
    import types

    import rosetta.contract.plugins as plugins_mod

    monkeypatch.setattr(plugins_mod, "_loaded_groups", set())
    monkeypatch.setattr(plugins_mod, "_failed_groups", {})
    monkeypatch.setattr(plugins_mod, "_ilm", types.SimpleNamespace(entry_points=lambda group=None: entry_points))


def test_discover_codecs_loads_entry_point_plugins(monkeypatch):
    class _FakeEP:
        name = "demo"
        value = "fake.mod"

        def load(self):
            register_decoder(_T, dtype="float64")(lambda msg, spec: None)

    _fake_ilm(monkeypatch, [_FakeEP()])
    try:
        codecs.discover_codecs()
        assert _T in DECODERS  # plugin codec resolvable by msg_type
    finally:
        _clear(_T)


def test_discover_codecs_runs_once(monkeypatch):
    calls = []

    class _CountingEP:
        name = "counter"
        value = "fake.mod"

        def load(self):
            calls.append(1)

    _fake_ilm(monkeypatch, [_CountingEP()])
    codecs.discover_codecs()
    codecs.discover_codecs()
    assert len(calls) == 1  # idempotent


def test_discover_codecs_raises_on_broken_plugin(monkeypatch):
    class _BadEP:
        name = "bad"
        value = "bad.mod"

        def load(self):
            raise ImportError("boom")

    _fake_ilm(monkeypatch, [_BadEP()])
    with pytest.raises(ContractValidationError, match="codec plugin"):
        codecs.discover_codecs()


def test_discover_codecs_failure_is_latched(monkeypatch):
    # A failed scan latches: every later call re-raises the ORIGINAL error,
    # even if the entry-point list changes afterwards. A rescan could not
    # recover anyway -- a plugin that registered something before failing
    # mid-import would hit the duplicate-registration guard on re-execution
    # and mask the real error. Fixing the environment means restarting.
    class _BadEP:
        name = "bad"
        value = "bad.mod"

        def load(self):
            raise ImportError("boom")

    class _GoodEP:
        name = "good"
        value = "fake.mod"

        def load(self):
            register_decoder(_T, dtype="float64")(lambda msg, spec: None)

    entry_points = [_BadEP()]
    _fake_ilm(monkeypatch, entry_points)
    try:
        with pytest.raises(ContractValidationError, match="boom") as first:
            codecs.discover_codecs()
        entry_points[:] = [_GoodEP()]  # "fixing" the environment in-process...
        with pytest.raises(ContractValidationError) as second:
            codecs.discover_codecs()  # ...does not help: same error, no rescan
        assert second.value is first.value
        assert _T not in DECODERS  # the good plugin was never loaded
    finally:
        _clear(_T)


def test_discovery_registers_builtin_codecs():
    """Regression: the built-ins used to register only via side-effect imports
    of robots.ros2.{decoders,encoders}, so registry checks depended on ambient
    import state (the parity test failed when run alone). Discovery itself now
    loads them; the accessors are the API spec resolution uses."""
    from rosetta.frames.codecs import has_decoder, has_encoder

    assert has_decoder("sensor_msgs/msg/JointState")
    assert has_encoder("geometry_msgs/msg/Twist")
    assert has_decoder("std_msgs/msg/Bool")  # required by stone.yaml's extended sections
