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
Operators: the registered array transforms behind a contract entry's ``apply`` list.

``forward`` runs on decode (front to back); ``inverse`` runs on encode, for
actions (back to front). Each operator declares an :class:`Invertibility`
tier. Third-party operators register through the ``rosetta.operators`` entry
point; the built-ins live in :mod:`.builtin_operators`.

A ``BIJECTIVE`` operator must pass a round-trip check when it is built
(``inverse(forward(x))`` recovers ``x``), so a plugin with a wrong inverse
fails at load instead of silently corrupting actions.

Operators should preserve the input dtype unless the transform is inherently
a float conversion (``rad2deg``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import numpy as np

from .errors import ContractValidationError
from .plugins import load_entry_point_plugins

__all__ = [
    "OPERATOR_ENTRY_POINT_GROUP",
    "Invertibility",
    "Operator",
    "OperatorContext",
    "register_operator",
]


class Invertibility(Enum):
    """
    How far an operator can run in the inverse (encode) direction.

    - ``FORWARD_ONLY``: no inverse (e.g. ``resize``). Not allowed on actions.
    - ``BIDIRECTIONAL``: has an inverse that does not round-trip (e.g. ``clamp``).
    - ``BIJECTIVE``: the inverse undoes forward (e.g. ``rad2deg``). Checked by
      the round-trip gate in :func:`build_operator`.
    """

    FORWARD_ONLY = "forward_only"
    BIDIRECTIONAL = "bidirectional"
    BIJECTIVE = "bijective"

    @property
    def serveable(self) -> bool:
        """True if the operator may run in the encode direction."""
        return self is not Invertibility.FORWARD_ONLY


@dataclass(frozen=True)
class OperatorContext:
    """Static facts about the stream an operator is built for."""

    is_image: bool = False


class Operator:
    """
    Base class for a value transform.

    ``@register_operator`` sets ``name``/``kind``. Subclasses implement
    ``forward``, plus ``inverse`` when the tier is serveable. ``args`` is the
    operator's YAML payload (``None`` for a bare name, ``[h, w]`` for
    ``resize``). Operators that take arguments override ``__init__``; the base
    rejects any payload.
    """

    name: str = ""
    kind: Invertibility = Invertibility.FORWARD_ONLY

    output_hw: tuple[int, int] | None = None
    """Output image geometry ``(h, w)`` if the operator sets it. The last
    non-None value in a pipeline is the stream's image size."""

    def __init__(self, args: Any, ctx: OperatorContext) -> None:
        del ctx
        if args is not None:
            raise ContractValidationError(f"operator '{self.name}' takes no arguments, got {args!r}")

    def forward(self, arr: np.ndarray) -> np.ndarray:
        """Apply the operator in the decode direction."""
        raise NotImplementedError

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        """Apply the operator in the encode direction."""
        raise NotImplementedError(f"operator '{self.name}' is {self.kind.name} and has no serve direction")

    def sample_input(self) -> np.ndarray:
        """
        Input for the ``BIJECTIVE`` round-trip gate.

        Override when the operator's domain is restricted (e.g. a log needs
        positive input).
        """
        return np.array([-2.0, -0.5, 0.0, 0.5, 2.0, 3.14159], dtype=np.float64)


OPERATOR_REGISTRY: dict[str, type[Operator]] = {}
"""Operator name -> class. Populated by @register_operator."""

OPERATOR_ENTRY_POINT_GROUP = "rosetta.operators"
"""Entry-point group for third-party operator plugins."""


def discover_operators() -> None:
    """Import every operator plugin module advertised under ``rosetta.operators``."""
    load_entry_point_plugins(OPERATOR_ENTRY_POINT_GROUP, "operator")


def register_operator(name: str, *, kind: Invertibility, override: bool = False):
    """
    Class decorator registering an operator under its YAML name.

    Raises ValueError if ``name`` is taken (unless ``override``), so a plugin
    cannot silently shadow another depending on import order. Raises
    TypeError if the class lacks ``forward``, or lacks ``inverse`` for a
    serveable ``kind``.
    """

    def _wrap(cls: type[Operator]) -> type[Operator]:
        if name in OPERATOR_REGISTRY and not override:
            raise ValueError(
                f"Operator already registered under '{name}'. Pass override=True to replace it intentionally."
            )
        if cls.forward is Operator.forward:
            raise TypeError(f"operator '{name}' does not implement forward()")
        if kind.serveable and cls.inverse is Operator.inverse:
            raise TypeError(
                f"operator '{name}' is declared {kind.name} (serveable) but does not "
                f"implement inverse(); implement it or declare FORWARD_ONLY"
            )
        cls.name = name
        cls.kind = kind
        OPERATOR_REGISTRY[name] = cls
        return cls

    return _wrap


def lookup_operator(name: str, *, ctx: str = "") -> type[Operator]:
    """Look up an operator class by name, discovering plugins first; ``ctx`` locates errors."""
    discover_operators()
    cls = OPERATOR_REGISTRY.get(name)
    if cls is None:
        where = f" in {ctx}" if ctx else ""
        known = ", ".join(sorted(OPERATOR_REGISTRY)) or "(none)"
        raise ContractValidationError(f"Unknown operator '{name}'{where}. Registered operators: {known}")
    return cls


def _verify_round_trip(operator: Operator) -> None:
    """
    Check that ``inverse(forward(x))`` recovers ``x`` on ``sample_input()``.

    Values must match to allclose tolerance and the shape exactly, so
    broadcasting cannot hide a shape bug.
    """
    x = np.asarray(operator.sample_input())
    if x.size == 0:
        raise ContractValidationError(
            f"operator '{operator.name}': sample_input() returned an empty array; "
            f"the round-trip gate needs at least one value"
        )
    try:
        y = np.asarray(operator.inverse(operator.forward(x)))
    except Exception as e:
        raise ContractValidationError(
            f"operator '{operator.name}' raised while running the BIJECTIVE round-trip "
            f"gate: {e}. Fix forward/inverse, or override sample_input() with an "
            f"in-domain sample."
        ) from e
    if y.shape != x.shape:
        raise ContractValidationError(
            f"operator '{operator.name}' is declared BIJECTIVE but the round trip changed shape {x.shape} -> {y.shape}."
        )
    if not np.allclose(y, x, rtol=1e-5, atol=1e-8):
        raise ContractValidationError(
            f"operator '{operator.name}' is declared BIJECTIVE but failed the "
            f"round-trip gate: inverse(forward(x)) != x. Fix the inverse, or "
            f"declare it BIDIRECTIONAL (serveable but lossy) / FORWARD_ONLY."
        )


def build_operator(name: str, args: Any, ctx: OperatorContext) -> Operator:
    """Instantiate a registered operator, running the round-trip gate for ``BIJECTIVE``."""
    cls = lookup_operator(name)
    operator = cls(args, ctx)
    if cls.kind is Invertibility.BIJECTIVE:
        _verify_round_trip(operator)
    return operator


def forward_pipeline(arr: np.ndarray, operators: Sequence[Operator]) -> np.ndarray:
    """Run operators front-to-back (decode direction)."""
    for operator in operators:
        arr = operator.forward(arr)
    return arr


def inverse_pipeline(arr: np.ndarray, operators: Sequence[Operator]) -> np.ndarray:
    """Run operators back-to-front via their inverses (encode direction)."""
    for operator in reversed(operators):
        arr = operator.inverse(arr)
    return arr
