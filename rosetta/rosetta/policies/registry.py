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

"""Resolving a :class:`~.protocols.PolicyRunner` by name.

A package registers its runner under :data:`POLICY_RUNNER_GROUP` (see
:mod:`rosetta.policies` for the registration snippet); these helpers look it
up at runtime without this package ever importing it. Loading fails fast: an
unknown or ambiguous name raises ``ValueError``; a non-class or
non-conforming entry point raises ``TypeError``; import and constructor
failures propagate raw with their full tracebacks (the traceback names the
package; no wrapper improves it).
"""

from __future__ import annotations

import importlib.metadata as _ilm

from ..plugins import instantiate_entry_point
from .protocols import PolicyRunner

__all__ = ["POLICY_RUNNER_GROUP", "available_policy_runners", "load_policy_runner"]

POLICY_RUNNER_GROUP = "rosetta.policy_runners"
"""Entry-point group a package registers its :class:`~.protocols.PolicyRunner` under."""


def load_policy_runner(name: str) -> PolicyRunner:
    """Instantiate the ``PolicyRunner`` registered as ``name``.

    The entry point must be a class whose zero-argument instances implement
    :class:`~.protocols.PolicyRunner`: a non-class or non-conforming entry
    point raises ``TypeError`` here; import/constructor failures propagate
    raw with their full tracebacks.
    """
    eps = list(_ilm.entry_points(group=POLICY_RUNNER_GROUP))
    return instantiate_entry_point(eps, POLICY_RUNNER_GROUP, name, PolicyRunner, "policy runner")


def available_policy_runners() -> list[str]:
    """Runner names registered under :data:`POLICY_RUNNER_GROUP`."""
    return sorted({ep.name for ep in _ilm.entry_points(group=POLICY_RUNNER_GROUP)})
