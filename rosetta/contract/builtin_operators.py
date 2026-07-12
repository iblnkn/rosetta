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
-- they just ship in-tree. Importing this module is what registers them;
:mod:`.schema` does so for side effect (``# noqa: F401``) so a contract that
references ``rad2deg``/``resize``/``clamp`` resolves them at load time.

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


@register_operator("resize", kind=Invertibility.FORWARD_ONLY)
class ResizeOperator(Operator):
    """
    Nearest-neighbor resize to ``[h, w]`` for HxW or HxWxC image arrays.

    FORWARD_ONLY: downsampling discards pixels, so resize is observation only.
    An action carrying ``resize`` is rejected at contract load.
    """

    def __init__(self, args: Any, ctx: OperatorContext) -> None:
        del ctx
        if not (isinstance(args, (list, tuple)) and len(args) == 2):
            raise ContractValidationError(f"resize operator expects [h, w], got {args!r}")
        self.height = int(args[0])
        self.width = int(args[1])
        if self.height <= 0 or self.width <= 0:
            raise ContractValidationError(
                f"resize operator dimensions must be positive, got [{self.height}, {self.width}]"
            )

    def forward(self, arr: np.ndarray) -> np.ndarray:
        return _nearest_resize(arr, self.height, self.width)


@register_operator("clamp", kind=Invertibility.BIDIRECTIONAL)
class ClampOperator(Operator):
    """
    Clip values element-wise to ``{min: lo, max: hi}`` (or ``[lo, hi]``).

    BIDIRECTIONAL, not BIJECTIVE: it *runs in the serve direction* but does not
    round-trip. The bound matters most on serve -- ``inverse_pipeline`` runs on
    encode (policy command -> ROS), so the outgoing command is clipped before
    it reaches hardware. ``forward`` clips identically so the same bound holds
    if ``clamp`` is used on an observation. Clamp is lossy outside the range
    (many inputs map to one bound), so it has no exact inverse; ``clip`` both
    ways is the safe, idempotent behavior we want (``clip o clip == clip``).
    That is exactly why it is BIDIRECTIONAL and skips the round-trip gate.
    """

    def __init__(self, args: Any, ctx: OperatorContext) -> None:
        del ctx
        if isinstance(args, dict):
            if set(args) != {"min", "max"}:
                raise ContractValidationError(f"clamp operator expects {{min, max}}, got {args!r}")
            bounds = (args["min"], args["max"])
        elif isinstance(args, (list, tuple)) and len(args) == 2:
            bounds = (args[0], args[1])
        else:
            raise ContractValidationError(f"clamp operator expects {{min, max}} or [lo, hi], got {args!r}")
        try:
            self.lo = float(bounds[0])
            self.hi = float(bounds[1])
        except (TypeError, ValueError) as exc:
            raise ContractValidationError(f"clamp operator bounds must be numbers, got {args!r}") from exc
        if not (np.isfinite(self.lo) and np.isfinite(self.hi)):
            raise ContractValidationError(f"clamp operator bounds must be finite, got [{self.lo}, {self.hi}]")
        if self.lo > self.hi:
            raise ContractValidationError(f"clamp operator requires lo <= hi, got [{self.lo}, {self.hi}]")

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
