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

"""
Operators: composable value transforms for the contract pipeline.

An *operator* is a small, registered plugin that maps a numeric array to
another numeric array. Operators are the ``apply`` list in a contract entry:
an ordered pipeline run after ``select`` (field projection).

Each operator carries a forward direction (dataset-build / message-decode)
and, when its tier allows, an inverse direction (serve / action-encode).
Actions run the pipeline in reverse via the inverse: ``inverse_pipeline``
walks the operators back-to-front through each operator's ``inverse``.

Each operator declares an :class:`Invertibility` tier at registration -- the
nested property ``FORWARD_ONLY`` < ``BIDIRECTIONAL`` < ``BIJECTIVE``:

- ``rad2deg`` is ``BIJECTIVE`` (degrees<->radians round-trips exactly).
- ``clamp`` is ``BIDIRECTIONAL`` (runs both ways but is lossy -- the bound is
  the point; it does not round-trip).
- ``resize`` is ``FORWARD_ONLY`` (downsampling discards information), so an
  action whose ``apply`` contains ``resize`` is rejected at contract load
  (see ``_parse_apply(require_serveable=True)``).

A ``BIJECTIVE`` operator is verified by a round-trip gate at build time: it
will not load unless ``inverse(forward(x)) == x`` on its
:meth:`Operator.sample_input`. A wrong inverse therefore fails fast at
contract load, not silently at runtime.

This module is the framework only (registry, tiers, round-trip gate,
pipelines); the built-in operators themselves (``rad2deg``, ``resize``,
``clamp``) live in :mod:`.builtin_operators` and register into
``OPERATOR_REGISTRY`` on import -- they hold no special status over a
third-party plugin loaded via :func:`discover_operators`. To add a
capability you register a new operator there; this module does not change.

Usage:
    @register_operator("rad2deg", kind=Invertibility.BIJECTIVE)
    class Rad2DegOperator(Operator):
        def forward(self, arr): return np.rad2deg(arr)
        def inverse(self, arr): return np.deg2rad(arr)

    operators = [build_operator("rad2deg", None, ctx)]   # round-trip gate runs here
    decoded = forward_pipeline(raw, operators)           # build direction
    action  = inverse_pipeline(vec, operators)           # serve direction
"""

from __future__ import annotations

import importlib.metadata as _ilm
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from .errors import ContractValidationError

# =============================================================================
# Invertibility taxonomy
# =============================================================================


class Invertibility(Enum):
    """
    How far an operator can run in the serve / encode (inverse) direction.

    The three tiers are *nested*, from least to most capable -- each tier
    includes the guarantees of the one before it:

    - ``FORWARD_ONLY``: only the build / decode direction is defined. Lossy and
      one-way (e.g. ``resize`` discards pixels). Illegal on an action, because
      the action path runs the inverse.
    - ``BIDIRECTIONAL``: a serve direction is defined and legal on actions, but
      it does *not* round-trip -- it is intentionally lossy (e.g. ``clamp``,
      whose inverse clips just like its forward). Safe to run both ways; the
      bound is the point.
    - ``BIJECTIVE``: the inverse exactly undoes the forward
      (``inverse(forward(x)) == x``, e.g. ``rad2deg``). This is the only tier
      subject to the compile-time round-trip gate (see :func:`build_operator`).

    Two gates read this one field:
    - action-pipeline gate (contract load): ``kind != FORWARD_ONLY``.
    - round-trip gate (contract load): ``kind == BIJECTIVE``.
    """

    FORWARD_ONLY = "forward_only"
    BIDIRECTIONAL = "bidirectional"
    BIJECTIVE = "bijective"

    @property
    def serveable(self) -> bool:
        """True if the operator may run in the serve / action-encode direction."""
        return self is not Invertibility.FORWARD_ONLY


# =============================================================================
# Operator context
# =============================================================================


@dataclass(frozen=True)
class OperatorContext:
    """
    Static context an operator may need to configure itself at build time.

    Resolved once when the contract loads, never at runtime, so operators stay
    process-independent across the dataset-build and serve processes.

    Attributes
    ----------
        is_image: Whether the stream is an image stream.

    """

    is_image: bool = False


# =============================================================================
# Operator base + registry
# =============================================================================


class Operator:
    """
    Base class for a value transform.

    Subclasses set ``name``/``kind`` via the ``@register_operator`` decorator
    and implement ``forward`` (and ``inverse`` when the tier is serveable).
    ``args`` is the operator's own YAML payload (``None`` for bare-string
    operators like ``rad2deg``, a list for ``resize: [h, w]``). ``ctx`` is the
    static :class:`OperatorContext`.
    """

    name: str = ""
    kind: Invertibility = Invertibility.FORWARD_ONLY

    def __init__(self, args: Any, ctx: OperatorContext) -> None:
        del args, ctx

    def forward(self, arr: np.ndarray) -> np.ndarray:
        """Apply the operator in the build / decode direction."""
        raise NotImplementedError

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        """Apply the operator in the serve / encode direction."""
        raise ContractValidationError(f"operator '{self.name}' is {self.kind.name} and has no serve direction")

    def sample_input(self) -> np.ndarray:
        """
        Representative input used by the round-trip gate for ``BIJECTIVE`` operators.

        Returns a 1-D array spanning the operator's valid domain. The default
        spread of reals suits unbounded numeric operators (e.g. ``rad2deg``);
        override for a restricted domain (e.g. a log operator that needs
        strictly positive input).
        """
        return np.array([-2.0, -0.5, 0.0, 0.5, 2.0, 3.14159], dtype=np.float64)


OPERATOR_REGISTRY: dict[str, type[Operator]] = {}
"""Registry mapping operator name -> operator class. Open set; add via @register_operator."""

OPERATOR_ENTRY_POINT_GROUP = "rosetta.operators"
"""Entry-point group third-party operator plugins register under (see discover_operators)."""

_operators_discovered = False


def discover_operators() -> None:
    """
    Import operator plugins advertised under the ``rosetta.operators`` entry point.

    Each entry point's value is a module path; loading it runs that module's
    ``@register_operator`` decorators, so installed plugins populate
    OPERATOR_REGISTRY automatically. The contract therefore references custom
    operators *by name* only -- never by module path -- keeping the contract
    free of implementation wiring. Idempotent: the scan runs once per process.

    A plugin that fails to import is a hard error (raised as
    ContractValidationError), not a silent skip -- a half-registered operator
    set would surface later as a confusing "Unknown operator".
    """
    global _operators_discovered  # noqa: PLW0603 - module-level discovery latch
    if _operators_discovered:
        return
    _operators_discovered = True  # set first: a failure must not trigger a re-scan

    try:
        eps = _ilm.entry_points(group=OPERATOR_ENTRY_POINT_GROUP)
    except TypeError:
        # importlib.metadata < 3.10 dict API (ROS 2 Jazzy is 3.12; defensive).
        eps = _ilm.entry_points().get(OPERATOR_ENTRY_POINT_GROUP, [])

    for ep in eps:
        try:
            ep.load()  # imports the module -> runs its @register_operator decorators
        except Exception as e:
            raise ContractValidationError(
                f"Failed to load operator plugin '{ep.name}' ({ep.value}) from "
                f"entry-point group '{OPERATOR_ENTRY_POINT_GROUP}': {e}"
            ) from e


def register_operator(name: str, *, kind: Invertibility):
    """
    Register an operator class under ``name``.

    Args:
    ----
        name: YAML operator name (the key used in an ``apply`` list).
        kind: The operator's :class:`Invertibility` tier. ``FORWARD_ONLY``
            operators are rejected on actions at contract load; ``BIJECTIVE``
            operators are round-trip verified at build time.

    """

    def _wrap(cls: type[Operator]) -> type[Operator]:
        cls.name = name
        cls.kind = kind
        OPERATOR_REGISTRY[name] = cls
        return cls

    return _wrap


def _verify_round_trip(operator: Operator) -> None:
    """
    Compile-time round-trip gate for ``BIJECTIVE`` operators.

    A ``BIJECTIVE`` declaration is a promise that ``inverse`` exactly undoes
    ``forward``. Verify it on the operator's :meth:`Operator.sample_input` so
    a wrong inverse fails at contract load rather than silently corrupting
    actions at runtime.
    """
    x = np.asarray(operator.sample_input(), dtype=np.float64)
    y = operator.inverse(operator.forward(x))
    if not np.allclose(y, x, rtol=1e-5, atol=1e-8):
        raise ContractValidationError(
            f"operator '{operator.name}' is declared BIJECTIVE but failed the "
            f"round-trip gate: inverse(forward(x)) != x. Fix the inverse, or "
            f"declare it BIDIRECTIONAL (serveable but lossy) / FORWARD_ONLY."
        )


def build_operator(name: str, args: Any, ctx: OperatorContext) -> Operator:
    """
    Instantiate a registered operator, running the round-trip gate for ``BIJECTIVE``.

    Args:
    ----
        name: Registered operator name.
        args: The operator's YAML payload (None / list / mapping / scalar).
        ctx: Static context for build-time configuration.

    Raises:
    ------
        ContractValidationError: If ``name`` is not a registered operator, or
            a ``BIJECTIVE`` operator fails the round-trip gate.

    """
    discover_operators()
    cls = OPERATOR_REGISTRY.get(name)
    if cls is None:
        known = ", ".join(sorted(OPERATOR_REGISTRY)) or "(none)"
        raise ContractValidationError(f"Unknown operator '{name}'. Registered operators: {known}")
    operator = cls(args, ctx)
    if cls.kind is Invertibility.BIJECTIVE:
        _verify_round_trip(operator)
    return operator


# =============================================================================
# Pipelines
# =============================================================================


def forward_pipeline(arr: np.ndarray, operators: "list[Operator] | tuple[Operator, ...]") -> np.ndarray:
    """Run operators front-to-back (build / decode direction)."""
    for operator in operators:
        arr = operator.forward(arr)
    return arr


def inverse_pipeline(arr: np.ndarray, operators: "list[Operator] | tuple[Operator, ...]") -> np.ndarray:
    """Run operators back-to-front via their inverses (serve / encode direction)."""
    for operator in reversed(operators):
        arr = operator.inverse(arr)
    return arr
