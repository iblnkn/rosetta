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
Built-in operators, registered on import (:mod:`.schema` imports this module).

They use only the public :class:`.operators.Operator` interface, the same as
a third-party plugin.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import ContractValidationError
from .operators import Invertibility, Operator, OperatorContext, register_operator


@register_operator("rad2deg", kind=Invertibility.BIJECTIVE)
class Rad2DegOperator(Operator):
    """Radians (ROS) -> degrees (dataset). Inverse: degrees -> radians."""

    def forward(self, arr: np.ndarray) -> np.ndarray:
        return np.rad2deg(arr)

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        return np.deg2rad(arr)


_MAX_RESIZE_DIM = 8192
"""Above any real camera, so a garbage value fails at load instead of
allocating gigabytes on the first message."""


@register_operator("resize", kind=Invertibility.FORWARD_ONLY)
class ResizeOperator(Operator):
    """Nearest-neighbor resize to ``[h, w]`` for HxW or HxWxC images. Image streams only."""

    def __init__(self, args: Any, ctx: OperatorContext) -> None:
        if not ctx.is_image:
            raise ContractValidationError("resize operator is only valid on image streams (observation.images.*)")
        if not (isinstance(args, (list, tuple)) and len(args) == 2):
            raise ContractValidationError(f"resize operator expects [h, w], got {args!r}")
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in args):
            raise ContractValidationError(f"resize operator dimensions must be integers, got {args!r}")
        if not all(0 < v <= _MAX_RESIZE_DIM for v in args):
            raise ContractValidationError(f"resize operator dimensions must be in [1, {_MAX_RESIZE_DIM}], got {args!r}")
        self.output_hw = (args[0], args[1])

    def forward(self, arr: np.ndarray) -> np.ndarray:
        return _nearest_resize(arr, *self.output_hw)


@register_operator("clamp", kind=Invertibility.BIDIRECTIONAL)
class ClampOperator(Operator):
    """
    Clip values element-wise to ``{min: lo, max: hi}``, in both directions.

    On an action the inverse clips the outgoing command before it reaches
    hardware. NaN passes through; ``encode_value`` rejects non-finite
    commands. The input dtype is preserved, so integer inputs truncate
    fractional bounds.
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
        # Float bounds would promote integer inputs (uint8 images -> float64).
        return np.clip(arr, self.lo, self.hi).astype(arr.dtype, copy=False)

    def forward(self, arr: np.ndarray) -> np.ndarray:
        return self._clip(arr)

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        return self._clip(arr)


def _nearest_resize(img: np.ndarray, rh: int, rw: int) -> np.ndarray:
    """Pure-numpy nearest-neighbor resize for HxW or HxWxC arrays.

    Indices are ``round(linspace(0, n-1, rn))``, so sampling is unbiased
    (2x2 -> 4x4 takes rows [0, 0, 1, 1]). Recording and serving share this
    code, so train and inference pixels match; cv2/PIL may differ by a pixel.
    """
    h, w = img.shape[:2]
    if h == rh and w == rw:
        return img
    y = np.round(np.linspace(0, h - 1, rh)).astype(np.int64)
    x = np.round(np.linspace(0, w - 1, rw)).astype(np.int64)
    return img[y][:, x]
