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

"""Resolving a :class:`~.protocols.DatasetWriter` / :class:`~.protocols.PolicyRunner` by name.

Adapters register their implementation under one of the two entry-point
groups below (see :mod:`rosetta.policies` for the registration snippet);
these helpers look it up at runtime without this package ever importing the
adapter directly.
"""

from __future__ import annotations

import importlib.metadata as _ilm
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .protocols import DatasetWriter, PolicyRunner

DATASET_WRITER_GROUP = "rosetta.dataset_writers"
"""Entry-point group an adapter registers its :class:`~.protocols.DatasetWriter` under."""

POLICY_RUNNER_GROUP = "rosetta.policy_runners"
"""Entry-point group an adapter registers its :class:`~.protocols.PolicyRunner` under."""


def _load_entry_point(group: str, name: str) -> Any:
    """Load the object an adapter registered under ``group`` as ``name``."""
    matches = {ep.name: ep for ep in _ilm.entry_points(group=group)}
    ep = matches.get(name)
    if ep is None:
        known = ", ".join(sorted(matches)) or "(none installed)"
        raise ValueError(f"No framework '{name}' registered under entry-point group '{group}'. Available: {known}.")
    return ep.load()


def load_dataset_writer(name: str) -> "DatasetWriter":
    """Instantiate the ``DatasetWriter`` registered as ``name``."""
    cls = _load_entry_point(DATASET_WRITER_GROUP, name)
    return cls()


def load_policy_runner(name: str) -> "PolicyRunner":
    """Instantiate the ``PolicyRunner`` registered as ``name``."""
    cls = _load_entry_point(POLICY_RUNNER_GROUP, name)
    return cls()


def available_frameworks(group: str) -> list[str]:
    """List framework names registered under an entry-point ``group``."""
    return sorted(ep.name for ep in _ilm.entry_points(group=group))
