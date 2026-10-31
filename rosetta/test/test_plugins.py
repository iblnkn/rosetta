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

"""Entry-point module plugins (operators, codecs): loaded once per process, failures latched."""

import types
import typing

import pytest

from rosetta import plugins
from rosetta.codecs import CODEC_ENTRY_POINT_GROUP
from rosetta.contract.interfaces import discover_codecs
from rosetta.errors import ContractValidationError
from rosetta.operators import OPERATOR_ENTRY_POINT_GROUP, discover_operators


class _EP:
    def __init__(self, name, load):
        self.name = name
        self.value = f"{name}.mod"
        self.load = load


def _fail():
    raise ImportError("boom")


@pytest.fixture
def entry_points(monkeypatch):
    """The entry points the plugin loader sees, by group; load latches cleared."""
    eps = {}
    monkeypatch.setattr(plugins, "_loaded_groups", set())
    monkeypatch.setattr(plugins, "_failed_groups", {})
    monkeypatch.setattr(plugins, "_ilm", types.SimpleNamespace(entry_points=lambda group: eps.get(group, [])))
    return eps


@pytest.mark.parametrize(
    ("discover", "group"),
    [(discover_operators, OPERATOR_ENTRY_POINT_GROUP), (discover_codecs, CODEC_ENTRY_POINT_GROUP)],
    ids=["operators", "codecs"],
)
def test_discovery_loads_its_entry_point_group(entry_points, fake_interface, discover, group):
    loaded = []
    entry_points[group] = [_EP("demo", lambda: loaded.append(group))]
    discover()
    assert loaded == [group]


def test_plugins_load_once_per_process(entry_points):
    loaded = []
    entry_points[OPERATOR_ENTRY_POINT_GROUP] = [_EP("counter", lambda: loaded.append(1))]
    discover_operators()
    discover_operators()
    assert loaded == [1]


def test_failed_load_names_the_plugin(entry_points):
    entry_points[OPERATOR_ENTRY_POINT_GROUP] = [_EP("bad", _fail)]
    with pytest.raises(ContractValidationError, match="operator plugin 'bad'.*boom"):
        discover_operators()


def test_failed_load_is_latched(entry_points):
    # Every later call re-raises the original error without rescanning. A rescan
    # could not recover: a plugin that registered something before failing would
    # hit the duplicate-registration guard on re-import and hide the real error.
    loaded = []
    entry_points[OPERATOR_ENTRY_POINT_GROUP] = [_EP("bad", _fail)]
    with pytest.raises(ContractValidationError) as first:
        discover_operators()
    entry_points[OPERATOR_ENTRY_POINT_GROUP] = [_EP("good", lambda: loaded.append(1))]
    with pytest.raises(ContractValidationError) as second:
        discover_operators()
    assert second.value is first.value
    assert loaded == []


@typing.runtime_checkable
class _Layout(typing.Protocol):
    """A class-plugin protocol with a data member, like rerun_rosetta's RecordingLayout."""

    index: str

    def column(self) -> str: ...


def _instantiate(cls):
    ep = types.SimpleNamespace(name="x", group="g", dist=types.SimpleNamespace(name="d"), load=lambda: cls)
    return plugins.instantiate_entry_point([ep], "g", "x", _Layout, "layout")


def test_data_member_protocol_accepts_a_conforming_class():
    class Good:
        index = "t"

        def column(self):
            return "c"

    assert isinstance(_instantiate(Good), Good)


def test_data_member_protocol_names_missing_members():
    class NoIndex:
        def column(self):
            return "c"

    class NoneColumn:
        index = "t"
        column = None

    with pytest.raises(TypeError, match=r"missing or non-callable: index$|missing or non-callable: index\)"):
        _instantiate(NoIndex)
    with pytest.raises(TypeError, match=r"missing or non-callable: column\)"):
        _instantiate(NoneColumn)
