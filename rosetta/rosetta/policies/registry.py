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

from .protocols import DatasetWriter, PolicyRunner

DATASET_WRITER_GROUP = "rosetta.dataset_writers"
"""Entry-point group an adapter registers its :class:`~.protocols.DatasetWriter` under."""

POLICY_RUNNER_GROUP = "rosetta.policy_runners"
"""Entry-point group an adapter registers its :class:`~.protocols.PolicyRunner` under."""


def _dist_name(ep: _ilm.EntryPoint) -> str:
    """Name of the distribution that registered ``ep``, best-effort."""
    dist = getattr(ep, "dist", None)
    return dist.name if dist is not None else "(unknown distribution)"


def _resolve(group: str, name: str) -> _ilm.EntryPoint:
    """Find the single entry point registered under ``group`` as ``name``."""
    eps = list(_ilm.entry_points(group=group))
    matches = [ep for ep in eps if ep.name == name]
    if not matches:
        known = ", ".join(sorted({ep.name for ep in eps})) or "(none installed)"
        raise ValueError(f"No framework '{name}' registered under entry-point group '{group}'. Available: {known}.")
    # importlib.metadata dedups same-named distributions (first on sys.path
    # wins), so this guard only catches two DIFFERENTLY-named dists.
    if len(matches) > 1:
        dists = ", ".join(sorted(_dist_name(ep) for ep in matches))
        raise ValueError(
            f"Framework '{name}' is registered more than once under entry-point group "
            f"'{group}' (by: {dists}) — resolution would be arbitrary; remove the stale distribution."
        )
    return matches[0]


def _instantiate(group: str, name: str, protocol: type) -> Any:
    """Load, instantiate, and structurally validate the ``name`` entry point.

    ``ep.load()`` and the zero-argument construction propagate raw: their
    tracebacks name the failing adapter module/class directly, which no
    wrapper improves on. The registry raises only for what it can diagnose
    better than a traceback: a non-class entry point and a non-conforming
    instance.
    """
    ep = _resolve(group, name)
    cls = ep.load()
    if not isinstance(cls, type):
        raise TypeError(
            f"Entry point '{name}' in group '{group}' ({_dist_name(ep)}) must be "
            f"a class; got {type(cls).__name__} ({cls!r})."
        )
    obj = cls()
    if not isinstance(obj, protocol):
        # isinstance is the gate; the sweep names the offenders — absent OR
        # present-but-non-callable (e.g. a member set to None fails isinstance
        # yet passes hasattr). __protocol_attrs__ is undocumented but the only
        # runtime member source before Python 3.13's typing.get_protocol_members;
        # both protocols are all-methods, so callability is the right test.
        problems = sorted(m for m in protocol.__protocol_attrs__ if not callable(getattr(obj, m, None)))
        detail = (
            f"missing or non-callable: {', '.join(problems)}"
            if problems
            else "all members present and callable; structural check failed on a non-method member"
        )
        raise TypeError(
            f"Entry point '{name}' in group '{group}' ({_dist_name(ep)}) does not "
            f"implement {protocol.__name__} ({detail})."
        )
    return obj


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
