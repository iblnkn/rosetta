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

import pytest

import rosetta.frames.codecs as conv
from rosetta.frames.codecs import (
    DECODERS,
    DTYPES,
    ENCODERS,
    register_decoder,
    register_encoder,
)

_T = "_test_msgs/msg/Foo"


def _clear(type_str):
    DECODERS.pop(type_str, None)
    DTYPES.pop(type_str, None)
    ENCODERS.pop(type_str, None)


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


# --- entry-point discovery ------------------------------------------------


def test_discover_codecs_loads_entry_point_plugins(monkeypatch):
    class _FakeEP:
        name = "demo"
        value = "fake.mod"

        def load(self):
            register_decoder(_T, dtype="float64")(lambda msg, spec: None)

    conv._codecs_discovered = False
    monkeypatch.setattr(conv._ilm, "entry_points", lambda group=None: [_FakeEP()])
    try:
        conv.discover_codecs()
        assert _T in DECODERS  # plugin codec resolvable by msg_type
    finally:
        _clear(_T)
        conv._codecs_discovered = False


def test_discover_codecs_runs_once(monkeypatch):
    calls = []
    conv._codecs_discovered = False
    monkeypatch.setattr(conv._ilm, "entry_points", lambda group=None: calls.append(1) or [])
    try:
        conv.discover_codecs()
        conv.discover_codecs()
        assert len(calls) == 1  # idempotent
    finally:
        conv._codecs_discovered = False


def test_discover_codecs_raises_on_broken_plugin(monkeypatch):
    class _BadEP:
        name = "bad"
        value = "bad.mod"

        def load(self):
            raise ImportError("boom")

    conv._codecs_discovered = False
    monkeypatch.setattr(conv._ilm, "entry_points", lambda group=None: [_BadEP()])
    try:
        with pytest.raises(ValueError, match="codec plugin"):
            conv.discover_codecs()
    finally:
        conv._codecs_discovered = False
