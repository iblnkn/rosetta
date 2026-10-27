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

import pytest

from rosetta.contract import plugins
from rosetta.contract.errors import ContractValidationError
from rosetta.contract.operators import OPERATOR_ENTRY_POINT_GROUP, discover_operators
from rosetta.frames.codecs import CODEC_ENTRY_POINT_GROUP, discover_codecs

GROUP = "rosetta.test_plugins"


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
    entry_points[GROUP] = [_EP("counter", lambda: loaded.append(1))]
    plugins.load_entry_point_plugins(GROUP, "test")
    plugins.load_entry_point_plugins(GROUP, "test")
    assert loaded == [1]


def test_failed_load_names_the_plugin(entry_points):
    entry_points[GROUP] = [_EP("bad", _fail)]
    with pytest.raises(ContractValidationError, match="test plugin 'bad'.*boom"):
        plugins.load_entry_point_plugins(GROUP, "test")


def test_failed_load_is_latched(entry_points):
    # Every later call re-raises the original error without rescanning. A rescan
    # could not recover: a plugin that registered something before failing would
    # hit the duplicate-registration guard on re-import and hide the real error.
    loaded = []
    entry_points[GROUP] = [_EP("bad", _fail)]
    with pytest.raises(ContractValidationError) as first:
        plugins.load_entry_point_plugins(GROUP, "test")
    entry_points[GROUP] = [_EP("good", lambda: loaded.append(1))]
    with pytest.raises(ContractValidationError) as second:
        plugins.load_entry_point_plugins(GROUP, "test")
    assert second.value is first.value
    assert loaded == []
