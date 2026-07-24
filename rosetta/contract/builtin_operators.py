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
Built-in operator plugins, registered into :mod:`.operators` on import.

These have no special status over a third-party plugin loaded via the
``rosetta.operators`` entry point (see :func:`.operators.discover_operators`)
-- they just ship in-tree, and everything spec resolution reads off them is
declared on the :class:`.operators.Operator` interface (``resize`` declares
its geometry via ``output_hw``; nothing is matched by name). Importing this
module is what registers them; :mod:`.schema` does so for side effect
(``# noqa: F401``) so a contract that references
``rad2deg``/``resize``/``clamp`` resolves them at load time.

To add a built-in operator, add it here; :mod:`.operators` (the framework:
registry, invertibility tiers, round-trip gate) does not change.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import ContractValidationError
from .operators import Invertibility, Operator, OperatorContext, register_operator

# =============================================================================
# Built-in operators
# =============================================================================


@register_operator("rad2deg", kind=Invertibility.BIJECTIVE)
class Rad2DegOperator(Operator):
    """Radians (ROS) -> degrees (dataset). Inverse: degrees -> radians."""

    def forward(self, arr: np.ndarray) -> np.ndarray:
        return np.rad2deg(arr)

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        return np.deg2rad(arr)


_MAX_RESIZE_DIM = 8192
"""Sanity ceiling for resize dimensions: beyond any real camera, so a
transposed or garbage value fails at contract load instead of attempting a
multi-GB allocation on the first message."""


@register_operator("resize", kind=Invertibility.FORWARD_ONLY)
class ResizeOperator(Operator):
    """
    Nearest-neighbor resize to ``[h, w]`` for HxW or HxWxC image arrays.

    FORWARD_ONLY: downsampling discards pixels, so resize is observation only.
    An action carrying ``resize`` is rejected at contract load. Only valid on
    image streams (``ctx.is_image``): on a state vector it would crash on
    every message at runtime, so it is rejected at load instead.
    """

    def __init__(self, args: Any, ctx: OperatorContext) -> None:
        if not ctx.is_image:
            raise ContractValidationError("resize operator is only valid on image streams (observation.images.*)")
        if not (isinstance(args, (list, tuple)) and len(args) == 2):
            raise ContractValidationError(f"resize operator expects [h, w], got {args!r}")
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in args):
            raise ContractValidationError(f"resize operator dimensions must be integers, got {args!r}")
        if not all(0 < v <= _MAX_RESIZE_DIM for v in args):
            raise ContractValidationError(f"resize operator dimensions must be in [1, {_MAX_RESIZE_DIM}], got {args!r}")
        self.output_hw = (args[0], args[1])  # declared geometry, read at spec resolution

    def forward(self, arr: np.ndarray) -> np.ndarray:
        return _nearest_resize(arr, *self.output_hw)


@register_operator("clamp", kind=Invertibility.BIDIRECTIONAL)
class ClampOperator(Operator):
    """
    Clip values element-wise to ``{min: lo, max: hi}``.

    BIDIRECTIONAL, not BIJECTIVE: it *runs in the serve direction* but does not
    round-trip. The bound matters most on serve -- ``inverse_pipeline`` runs on
    encode (policy command -> ROS), so the outgoing command is clipped before
    it reaches hardware. ``forward`` clips identically so the same bound holds
    if ``clamp`` is used on an observation. Clamp is lossy outside the range
    (many inputs map to one bound), so it has no exact inverse; ``clip`` both
    ways is the safe, idempotent behavior we want (``clip o clip == clip``).
    That is exactly why it is BIDIRECTIONAL and skips the round-trip gate.

    Clamp is a *range* bound, not a finiteness guard: ``np.clip`` propagates
    NaN unchanged. Non-finite commands are refused by the serve path's
    finiteness gate in :func:`rosetta.frames.codecs.encode_value`, never by
    clamp. The input dtype is preserved (no silent float64 promotion); on
    integer dtypes, fractional bounds are truncated by the cast back.
    """

    def __init__(self, args: Any, ctx: OperatorContext) -> None:
        del ctx
        if not (isinstance(args, dict) and set(args) == {"min", "max"}):
            raise ContractValidationError(f"clamp operator expects {{min, max}}, got {args!r}")
        bounds = (args["min"], args["max"])
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in bounds):
            raise ContractValidationError(f"clamp operator bounds must be numbers, got {args!r}")
        self.lo = float(bounds[0])
        self.hi = float(bounds[1])
        if not (np.isfinite(self.lo) and np.isfinite(self.hi)):
            raise ContractValidationError(f"clamp operator bounds must be finite, got [{self.lo}, {self.hi}]")
        if self.lo > self.hi:
            raise ContractValidationError(f"clamp operator requires lo <= hi, got [{self.lo}, {self.hi}]")

    def _clip(self, arr: np.ndarray) -> np.ndarray:
        # astype back: python-float bounds would promote integer inputs (uint8
        # images -> float64, 8x memory) under NEP 50; operators preserve dtype.
        return np.clip(arr, self.lo, self.hi).astype(arr.dtype, copy=False)

    def forward(self, arr: np.ndarray) -> np.ndarray:
        return self._clip(arr)

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        return self._clip(arr)


# =============================================================================
# Helpers
# =============================================================================


def _nearest_resize(img: np.ndarray, rh: int, rw: int) -> np.ndarray:
    """Pure-numpy nearest-neighbor resize for HxW or HxWxC arrays.

    Sampling convention: endpoint-aligned indices, symmetrically rounded
    (``round(linspace(0, n-1, rn))``), so up- and downsampling stay unbiased
    (2x2 -> 4x4 samples rows [0, 0, 1, 1], not the top-left-skewed [0, 0, 0, 1]
    of truncation). Record and serve share this exact code path, so train and
    inference pixels match by construction; resizing with an external tool
    (cv2/PIL use slightly different conventions) may differ by up to 1 source
    pixel.
    """
    h, w = img.shape[:2]
    if h == rh and w == rw:
        return img
    y = np.round(np.linspace(0, h - 1, rh)).astype(np.int64)
    x = np.round(np.linspace(0, w - 1, rw)).astype(np.int64)
    # Works for both 2D (HxW) and 3D (HxWxC) arrays
    return img[y][:, x]
