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
when it is invertible, an inverse direction (serve / action-encode). Actions
run the pipeline in reverse via the inverse: ``inverse_pipeline`` walks the
ops back-to-front through each op's ``inverse``.

Invertibility is a *declared* property of each op (the ``invertible`` flag
set at registration). ``rad2deg`` is invertible (degrees<->radians is an
exact bijection); ``resize`` is not (downsampling discards information), so
an action whose ``apply`` contains ``resize`` is rejected at contract load
(see ``_parse_apply(require_invertible=True)``).

To add a capability you register a new op here; the contract frame does not
change.

Usage:
    @register_op("rad2deg", invertible=True)
    class Rad2DegOp(Op):
        def forward(self, arr): return np.rad2deg(arr)
        def inverse(self, arr): return np.deg2rad(arr)

    ops = [build_op("rad2deg", None, ctx)]
    decoded = forward_pipeline(raw, ops)        # build direction
    action  = inverse_pipeline(vec, ops)        # serve direction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .contract import ContractValidationError


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

    Subclasses set ``name``/``invertible`` via the ``@register_op`` decorator
    and implement ``forward`` (and ``inverse`` when invertible). ``args`` is
    the op's own YAML payload (``None`` for bare-string ops like ``rad2deg``,
    a list for ``resize: [h, w]``). ``ctx`` is the static :class:`OpContext`.
    """

    name: str = ''
    invertible: bool = True

    def __init__(self, args: Any, ctx: OpContext) -> None:  # noqa: D107
        del args, ctx

    def forward(self, arr: np.ndarray) -> np.ndarray:
        """Apply the op in the build / decode direction."""
        raise NotImplementedError

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        """Apply the op in the serve / encode direction."""
        raise ContractValidationError(
            f"op '{self.name}' is not invertible and has no inverse"
        )


OP_REGISTRY: dict[str, type[Op]] = {}
"""Registry mapping op name -> op class. Open set; add via @register_op."""


def register_op(name: str, *, invertible: bool):
    """
    Register an op class under ``name``.

    Args
    ----
        name: YAML op name (the key used in an ``apply`` list).
        invertible: Whether the op can run in the serve direction. Set False
            for lossy ops (e.g. resize); using such an op on an action is a
            contract-load error.

    """

    def _wrap(cls: type[Op]) -> type[Op]:
        cls.name = name
        cls.invertible = invertible
        OP_REGISTRY[name] = cls
        return cls

    return _wrap


def build_op(name: str, args: Any, ctx: OpContext) -> Op:
    """
    Instantiate a registered op.

    Args
    ----
        name: Registered op name.
        args: The op's YAML payload (None / list / scalar).
        ctx: Static context for build-time configuration.

    Raises
    ------
        ContractValidationError: If ``name`` is not a registered op.

    """
    cls = OP_REGISTRY.get(name)
    if cls is None:
        known = ', '.join(sorted(OP_REGISTRY)) or '(none)'
        raise ContractValidationError(
            f"Unknown op '{name}'. Registered ops: {known}"
        )
    return cls(args, ctx)


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


@register_op('rad2deg', invertible=True)
class Rad2DegOp(Op):
    """Radians (ROS) -> degrees (dataset). Inverse: degrees -> radians."""

    def forward(self, arr: np.ndarray) -> np.ndarray:
        return np.rad2deg(arr)

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        return np.deg2rad(arr)


@register_op('resize', invertible=False)
class ResizeOp(Op):
    """
    Nearest-neighbor resize to ``[h, w]`` for HxW or HxWxC image arrays.

    Non-invertible: downsampling discards pixels, so resize is observation
    only. An action carrying ``resize`` is rejected at contract load.
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


@register_op('clamp', invertible=True)
class ClampOp(Op):
    """
    Clip values element-wise to ``[lo, hi]``.

    Invertible in the sense ops.py uses the word: it *can run in the serve
    direction*. The bound matters most there -- ``inverse_pipeline`` runs on
    encode (policy command -> ROS), so the outgoing command is clipped before
    it reaches hardware. ``forward`` clips identically so the same bound holds
    if ``clamp`` is used on an observation. Clamp has no true mathematical
    inverse (it is lossy outside the range), but ``clip`` both ways is the safe,
    idempotent behavior we want (``clip o clip == clip``).
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
