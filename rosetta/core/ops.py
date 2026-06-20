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
Ops: composable value transforms for the contract pipeline.

An *op* is a small, registered plugin that maps a numeric array to another
numeric array. Ops are the ``apply`` list in a contract entry: an ordered
pipeline run after ``select`` (field projection) and before ``align``
(resampling).

Each op carries a forward direction (dataset-build / message-decode) and,
when its tier allows, an inverse direction (serve / action-encode). Actions
run the pipeline in reverse via the inverse: ``inverse_pipeline`` walks the
ops back-to-front through each op's ``inverse``.

Each op declares an :class:`Invertibility` tier at registration -- the nested
property ``FORWARD_ONLY`` < ``BIDIRECTIONAL`` < ``BIJECTIVE``:

- ``rad2deg`` is ``BIJECTIVE`` (degrees<->radians round-trips exactly).
- ``clamp`` is ``BIDIRECTIONAL`` (runs both ways but is lossy -- the bound is
  the point; it does not round-trip).
- ``resize`` is ``FORWARD_ONLY`` (downsampling discards information), so an
  action whose ``apply`` contains ``resize`` is rejected at contract load
  (see ``_parse_apply(require_serveable=True)``).

A ``BIJECTIVE`` op is verified by a round-trip gate at build time: it will not
load unless ``inverse(forward(x)) == x`` on its :meth:`Op.sample_input`. A
wrong inverse therefore fails fast at contract load, not silently at runtime.

To add a capability you register a new op here; the contract frame does not
change.

Usage:
    @register_op("rad2deg", kind=Invertibility.BIJECTIVE)
    class Rad2DegOp(Op):
        def forward(self, arr): return np.rad2deg(arr)
        def inverse(self, arr): return np.deg2rad(arr)

    ops = [build_op("rad2deg", None, ctx)]   # round-trip gate runs here
    decoded = forward_pipeline(raw, ops)        # build direction
    action  = inverse_pipeline(vec, ops)        # serve direction
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import importlib.metadata as _ilm
from typing import Any

import numpy as np

from .contract import ContractValidationError


# =============================================================================
# Invertibility taxonomy
# =============================================================================


class Invertibility(Enum):
    """
    How far an op can run in the serve / encode (inverse) direction.

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
      subject to the compile-time round-trip gate (see :func:`build_op`).

    Two gates read this one field:
    - action-pipeline gate (contract load): ``kind != FORWARD_ONLY``.
    - round-trip gate (contract load): ``kind == BIJECTIVE``.
    """

    FORWARD_ONLY = 'forward_only'
    BIDIRECTIONAL = 'bidirectional'
    BIJECTIVE = 'bijective'

    @property
    def serveable(self) -> bool:
        """True if the op may run in the serve / action-encode direction."""
        return self is not Invertibility.FORWARD_ONLY


# =============================================================================
# Op context
# =============================================================================


@dataclass(frozen=True)
class OpContext:
    """
    Static context an op may need to configure itself at build time.

    Resolved once when the contract loads, never at runtime, so ops stay
    process-independent across the dataset-build and serve processes.

    Attributes
    ----------
        is_image: Whether the stream is an image stream.
        image_channels: Channel count for image streams, when known.

    """

    is_image: bool = False
    image_channels: int | None = None


# =============================================================================
# Op base + registry
# =============================================================================


class Op:
    """
    Base class for a value transform.

    Subclasses set ``name``/``kind`` via the ``@register_op`` decorator and
    implement ``forward`` (and ``inverse`` when the tier is serveable). ``args``
    is the op's own YAML payload (``None`` for bare-string ops like ``rad2deg``,
    a list for ``resize: [h, w]``). ``ctx`` is the static :class:`OpContext`.
    """

    name: str = ''
    kind: Invertibility = Invertibility.FORWARD_ONLY

    def __init__(self, args: Any, ctx: OpContext) -> None:  # noqa: D107
        del args, ctx

    def forward(self, arr: np.ndarray) -> np.ndarray:
        """Apply the op in the build / decode direction."""
        raise NotImplementedError

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        """Apply the op in the serve / encode direction."""
        raise ContractValidationError(
            f"op '{self.name}' is {self.kind.name} and has no serve direction"
        )

    def sample_input(self) -> np.ndarray:
        """
        Representative input used by the round-trip gate for ``BIJECTIVE`` ops.

        Returns a 1-D array spanning the op's valid domain. The default spread
        of reals suits unbounded numeric ops (e.g. ``rad2deg``); override for a
        restricted domain (e.g. a log op that needs strictly positive input).
        """
        return np.array([-2.0, -0.5, 0.0, 0.5, 2.0, 3.14159], dtype=np.float64)


OP_REGISTRY: dict[str, type[Op]] = {}
"""Registry mapping op name -> op class. Open set; add via @register_op."""

OP_ENTRY_POINT_GROUP = 'rosetta.ops'
"""Entry-point group third-party op plugins register under (see discover_ops)."""

_ops_discovered = False


def discover_ops() -> None:
    """
    Import op-plugin packages advertised under the ``rosetta.ops`` entry point.

    Each entry point's value is a module path; loading it runs that module's
    ``@register_op`` decorators, so installed plugins populate OP_REGISTRY
    automatically. The contract therefore references custom ops *by name* only
    -- never by module path -- keeping the contract free of implementation
    wiring. Idempotent: the scan runs once per process.

    A plugin that fails to import is a hard error (raised as
    ContractValidationError), not a silent skip -- a half-registered op set
    would surface later as a confusing "Unknown op".
    """
    global _ops_discovered
    if _ops_discovered:
        return
    _ops_discovered = True  # set first: a failure must not trigger a re-scan

    try:
        eps = _ilm.entry_points(group=OP_ENTRY_POINT_GROUP)
    except TypeError:
        # importlib.metadata < 3.10 dict API (ROS 2 Jazzy is 3.12; defensive).
        eps = _ilm.entry_points().get(OP_ENTRY_POINT_GROUP, [])

    for ep in eps:
        try:
            ep.load()  # imports the module -> runs its @register_op decorators
        except Exception as e:  # noqa: BLE001 -- a broken plugin must be loud
            raise ContractValidationError(
                f"Failed to load op plugin '{ep.name}' ({ep.value}) from "
                f"entry-point group '{OP_ENTRY_POINT_GROUP}': {e}"
            ) from e


def register_op(name: str, *, kind: Invertibility):
    """
    Register an op class under ``name``.

    Args
    ----
        name: YAML op name (the key used in an ``apply`` list).
        kind: The op's :class:`Invertibility` tier. ``FORWARD_ONLY`` ops are
            rejected on actions at contract load; ``BIJECTIVE`` ops are
            round-trip verified at build time.

    """

    def _wrap(cls: type[Op]) -> type[Op]:
        cls.name = name
        cls.kind = kind
        OP_REGISTRY[name] = cls
        return cls

    return _wrap


def _verify_round_trip(op: Op) -> None:
    """
    Compile-time round-trip gate for ``BIJECTIVE`` ops.

    A ``BIJECTIVE`` declaration is a promise that ``inverse`` exactly undoes
    ``forward``. Verify it on the op's :meth:`Op.sample_input` so a wrong
    inverse fails at contract load rather than silently corrupting actions at
    runtime.
    """
    x = np.asarray(op.sample_input(), dtype=np.float64)
    y = op.inverse(op.forward(x))
    if not np.allclose(y, x, rtol=1e-5, atol=1e-8):
        raise ContractValidationError(
            f"op '{op.name}' is declared BIJECTIVE but failed the round-trip "
            f'gate: inverse(forward(x)) != x. Fix the inverse, or declare it '
            f'BIDIRECTIONAL (serveable but lossy) / FORWARD_ONLY.'
        )


def build_op(name: str, args: Any, ctx: OpContext) -> Op:
    """
    Instantiate a registered op, running the round-trip gate for ``BIJECTIVE``.

    Args
    ----
        name: Registered op name.
        args: The op's YAML payload (None / list / scalar).
        ctx: Static context for build-time configuration.

    Raises
    ------
        ContractValidationError: If ``name`` is not a registered op, or a
            ``BIJECTIVE`` op fails the round-trip gate.

    """
    discover_ops()
    cls = OP_REGISTRY.get(name)
    if cls is None:
        known = ', '.join(sorted(OP_REGISTRY)) or '(none)'
        raise ContractValidationError(
            f"Unknown op '{name}'. Registered ops: {known}"
        )
    op = cls(args, ctx)
    if cls.kind is Invertibility.BIJECTIVE:
        _verify_round_trip(op)
    return op


# =============================================================================
# Pipelines
# =============================================================================


def forward_pipeline(arr: np.ndarray, ops: 'list[Op] | tuple[Op, ...]') -> np.ndarray:
    """Run ops front-to-back (build / decode direction)."""
    for op in ops:
        arr = op.forward(arr)
    return arr


def inverse_pipeline(arr: np.ndarray, ops: 'list[Op] | tuple[Op, ...]') -> np.ndarray:
    """Run ops back-to-front via their inverses (serve / encode direction)."""
    for op in reversed(ops):
        arr = op.inverse(arr)
    return arr


# =============================================================================
# Built-in ops
# =============================================================================


@register_op('rad2deg', kind=Invertibility.BIJECTIVE)
class Rad2DegOp(Op):
    """Radians (ROS) -> degrees (dataset). Inverse: degrees -> radians."""

    def forward(self, arr: np.ndarray) -> np.ndarray:
        return np.rad2deg(arr)

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        return np.deg2rad(arr)


@register_op('resize', kind=Invertibility.FORWARD_ONLY)
class ResizeOp(Op):
    """
    Nearest-neighbor resize to ``[h, w]`` for HxW or HxWxC image arrays.

    FORWARD_ONLY: downsampling discards pixels, so resize is observation only.
    An action carrying ``resize`` is rejected at contract load.
    """

    def __init__(self, args: Any, ctx: OpContext) -> None:
        del ctx
        if not (isinstance(args, (list, tuple)) and len(args) == 2):
            raise ContractValidationError(
                f"resize op expects [h, w], got {args!r}"
            )
        self.height = int(args[0])
        self.width = int(args[1])
        if self.height <= 0 or self.width <= 0:
            raise ContractValidationError(
                f'resize op dimensions must be positive, got [{self.height}, {self.width}]'
            )

    def forward(self, arr: np.ndarray) -> np.ndarray:
        return _nearest_resize(arr, self.height, self.width)


@register_op('clamp', kind=Invertibility.BIDIRECTIONAL)
class ClampOp(Op):
    """
    Clip values element-wise to ``[lo, hi]``.

    BIDIRECTIONAL, not BIJECTIVE: it *runs in the serve direction* but does not
    round-trip. The bound matters most on serve -- ``inverse_pipeline`` runs on
    encode (policy command -> ROS), so the outgoing command is clipped before
    it reaches hardware. ``forward`` clips identically so the same bound holds
    if ``clamp`` is used on an observation. Clamp is lossy outside the range
    (many inputs map to one bound), so it has no exact inverse; ``clip`` both
    ways is the safe, idempotent behavior we want (``clip o clip == clip``).
    That is exactly why it is BIDIRECTIONAL and skips the round-trip gate.
    """

    def __init__(self, args: Any, ctx: OpContext) -> None:
        del ctx
        if not (isinstance(args, (list, tuple)) and len(args) == 2):
            raise ContractValidationError(
                f"clamp op expects [lo, hi], got {args!r}"
            )
        try:
            self.lo = float(args[0])
            self.hi = float(args[1])
        except (TypeError, ValueError) as exc:
            raise ContractValidationError(
                f"clamp op bounds must be numbers, got {args!r}"
            ) from exc
        if not (np.isfinite(self.lo) and np.isfinite(self.hi)):
            raise ContractValidationError(
                f'clamp op bounds must be finite, got [{self.lo}, {self.hi}]'
            )
        if self.lo > self.hi:
            raise ContractValidationError(
                f'clamp op requires lo <= hi, got [{self.lo}, {self.hi}]'
            )

    def forward(self, arr: np.ndarray) -> np.ndarray:
        return np.clip(arr, self.lo, self.hi)

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        return np.clip(arr, self.lo, self.hi)


# =============================================================================
# Helpers
# =============================================================================


def _nearest_resize(img: np.ndarray, rh: int, rw: int) -> np.ndarray:
    """Pure-numpy nearest-neighbor resize for HxW or HxWxC arrays."""
    h, w = img.shape[:2]
    if h == rh and w == rw:
        return img
    y = np.linspace(0, h - 1, rh).astype(np.int64)
    x = np.linspace(0, w - 1, rw).astype(np.int64)
    # Works for both 2D (HxW) and 3D (HxWxC) arrays
    return img[y][:, x]
