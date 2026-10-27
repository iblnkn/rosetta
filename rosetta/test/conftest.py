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

"""Fixtures shared by the core tests."""

import importlib.metadata as ilm
from types import SimpleNamespace

import pytest

from rosetta.robots import registry


@pytest.fixture
def fake_interface(monkeypatch):
    """Make ``fake`` (fake_interface.py) the only installed robot interface for one test."""
    ep = ilm.EntryPoint(name="fake", value="fake_interface:FakeInterface", group=registry.ROBOT_INTERFACE_GROUP)
    monkeypatch.setattr(registry, "_ilm", SimpleNamespace(entry_points=lambda group: [ep]))
    monkeypatch.setattr(registry, "_instances", {})
    monkeypatch.setattr(registry, "_codec_state", {"done": False, "failure": None})
