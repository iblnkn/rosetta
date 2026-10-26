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
adapter directly. Loading fails fast: an unknown or ambiguous name raises
``ValueError``; a non-class or non-conforming entry point raises
``TypeError``; adapter import/constructor failures propagate raw with their
full tracebacks (the traceback names the adapter — no wrapper improves it).
"""

from __future__ import annotations

import importlib.metadata as _ilm
from typing import Any

from ..contract.plugins import instantiate_entry_point
from .protocols import DatasetWriter, PolicyRunner

DATASET_WRITER_GROUP = "rosetta.dataset_writers"
"""Entry-point group an adapter registers its :class:`~.protocols.DatasetWriter` under."""

POLICY_RUNNER_GROUP = "rosetta.policy_runners"
"""Entry-point group an adapter registers its :class:`~.protocols.PolicyRunner` under."""


def _instantiate(group: str, name: str, protocol: type) -> Any:
    """Load, instantiate, and structurally validate the ``name`` entry point."""
    return instantiate_entry_point(list(_ilm.entry_points(group=group)), group, name, protocol, "framework")


def load_dataset_writer(name: str) -> DatasetWriter:
    """Instantiate the ``DatasetWriter`` registered as ``name``.

    The entry point must be a class whose zero-argument instances implement
    :class:`~.protocols.DatasetWriter`: a non-class or non-conforming entry
    point raises ``TypeError`` here; import/constructor failures propagate
    raw with their full tracebacks.
    """
    return _instantiate(DATASET_WRITER_GROUP, name, DatasetWriter)


def load_policy_runner(name: str) -> PolicyRunner:
    """Instantiate the ``PolicyRunner`` registered as ``name``.

    The entry point must be a class whose zero-argument instances implement
    :class:`~.protocols.PolicyRunner`: a non-class or non-conforming entry
    point raises ``TypeError`` here; import/constructor failures propagate
    raw with their full tracebacks.
    """
    return _instantiate(POLICY_RUNNER_GROUP, name, PolicyRunner)


def available_dataset_writers() -> list[str]:
    """Framework names registered under :data:`DATASET_WRITER_GROUP`."""
    return sorted({ep.name for ep in _ilm.entry_points(group=DATASET_WRITER_GROUP)})


def available_policy_runners() -> list[str]:
    """Framework names registered under :data:`POLICY_RUNNER_GROUP`."""
    return sorted({ep.name for ep in _ilm.entry_points(group=POLICY_RUNNER_GROUP)})
