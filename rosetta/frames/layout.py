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
FrameLayout: the canonical mapping between contract keys and per-spec slices.

Multiple contract specs may share one output key (e.g. two topics feeding
``observation.state``); their values concatenate into one flat vector. This
module is the single source of truth for that layout:

- key order is first-occurrence order of the spec list
- within a key, specs concatenate in declaration order
- each spec occupies a static slice sized ``max(len(spec.names), 1)``

Layout is purely positional (list/declaration order), by design. Full control
over where a source lands in the concatenated vector requires *some* explicit
signal, and order is the only one that doesn't add a second mechanism on top
of what YAML already gives for free: a list is already ordered, and
reordering two sources in the contract is a one-line diff. An explicit
ordering key (e.g. `position: 2`) would duplicate that ordering in a second
place, invite drift between the two, and still resolve ties by... list order.
Order is therefore not a limitation to work around but the entire mechanism.

Everything that assembles frames (live bridge, bag porter), splits action
frames back into per-stream slices, or derives per-key offsets for a backend
(LeRobot features, starVLA modality.json, vla_foundry field dims) must go
through a FrameLayout so the layouts agree by construction.

The contract's dtype vocabulary ('float32', 'video', ...) is
lerobot-compatible by choice; this module has no lerobot dependency. Backends
derive their own schema views from KeyLayout (starVLA/vla_foundry grouping);
lerobot_features() is the one lerobot-format emitter, kept here so the
declared feature shapes and the zero-fill shapes share one source of truth.

Pure numpy; no ROS dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..contract.errors import ContractValidationError
from ..contract.specs import (
    ActionStreamSpec,
    ObservationStreamSpec,
    StreamSpec,
)

# All supported image encodings are normalized to 3-channel uint8 RGB by the
# decoders (3ch color as-is, 4ch drops alpha, mono->rgb); depth/RGBD encodings
# are explicitly rejected at decode (see decoders.DepthEncodingNotSupported).
# The decoded output channel count is therefore always 3, regardless of the
# source encoding. This is the single source of truth for the feature shape
# and the zero-fill shape so they agree.
DECODED_IMAGE_CHANNELS = 3

# Contract dtype string (lerobot-compatible vocabulary) -> numpy dtype, for
# the numeric frame-assembly paths.
DTYPE_MAP: dict[str, Any] = {
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "int64": np.int64,
    "bool": bool,
}


def resolve_stream_dtype(spec: StreamSpec) -> str:
    """Contract dtype string (lerobot-compatible) for any resolved stream spec.

    Both ObservationStreamSpec and ActionStreamSpec carry a resolved ``dtype``
    (set at spec iteration: explicit > native codec dtype > float64 for custom
    decoders).
    """
    dtype = getattr(spec, "dtype", None)
    if not dtype:
        raise ValueError(f"Spec '{spec.key}' has no resolved dtype")
    return dtype


def get_namespaced_names(spec: StreamSpec) -> list[str]:
    """Get selector names with namespace prefix applied."""
    namespace = getattr(spec, "namespace", None)
    if namespace:
        return [f"{namespace}.{n}" for n in spec.names]
    return list(spec.names)


def build_feature(spec: ObservationStreamSpec | ActionStreamSpec) -> dict[str, Any]:
    """
    Build LeRobot feature dict from a single spec.

    Returns a dict with:
    - dtype: LeRobot dtype string
    - shape: tuple of dimensions
    - names: axis names (or None)

    For a key shared by several specs, use FrameLayout.lerobot_features()
    instead so the shapes/names cover the whole concatenated vector.
    """
    dtype = resolve_stream_dtype(spec)

    if dtype in ("video", "image"):
        if not spec.image_resize:
            raise ValueError(f"Image spec '{spec.key}' must have image_resize")
        h, w = spec.image_resize
        return {
            "dtype": "video",
            "shape": (h, w, DECODED_IMAGE_CHANNELS),
            "names": ["height", "width", "channels"],
        }

    if dtype == "string":
        return {"dtype": "string", "shape": (1,), "names": None}

    # Numeric types (float32, float64, int32, int64, bool)
    n = len(spec.names) if spec.names else 1
    names = list(spec.names) if spec.names else None
    return {"dtype": dtype, "shape": (n,), "names": names}


def zeros_for_spec(spec: ObservationStreamSpec) -> np.ndarray:
    """
    Create a zero-filled array for missing data.

    Returns
    -------
        Zero-filled numpy array:
        - Images: (H, W, DECODED_IMAGE_CHANNELS) uint8
        - Vectors: (N,) with dtype from spec, N = max(len(names), 1)

    """
    if spec.is_image:
        h, w = spec.image_resize
        # A zero-filled missing frame must match the shape of a real decoded
        # frame and the feature declared by build_feature.
        return np.zeros((h, w, DECODED_IMAGE_CHANNELS), dtype=np.uint8)

    np_dtype = DTYPE_MAP.get(spec.dtype, np.float32)
    return np.zeros(max(len(spec.names), 1), dtype=np_dtype)


def _spec_category(spec: StreamSpec) -> str:
    """'image' | 'string' | 'numeric' for layout purposes.

    Duck-typed (like build_feature/zeros_for_spec) so test stand-ins work:
    action specs simply have no is_image attribute and are numeric.
    """
    if getattr(spec, "is_image", False):
        return "image"
    if getattr(spec, "dtype", None) == "string":
        return "string"
    return "numeric"


@dataclass(frozen=True)
class SpecSlice:
    """One spec's static slice within its key's concatenated vector."""

    spec: StreamSpec
    start: int
    dim: int  # max(len(spec.names), 1)

    @property
    def end(self) -> int:
        return self.start + self.dim


@dataclass(frozen=True)
class KeyLayout:
    """Layout of one contract key: category, dtype, and ordered spec slices."""

    key: str
    category: str  # 'image' | 'string' | 'numeric'
    dtype: str  # resolved LeRobot dtype string
    slices: tuple[SpecSlice, ...]  # declaration order within the key

    @property
    def dim(self) -> int:
        return sum(s.dim for s in self.slices)

    @property
    def np_dtype(self) -> Any:
        return DTYPE_MAP[self.dtype]


class FrameLayout:
    """Canonical key -> (spec, offset, dim) layout for an ordered spec list.

    Built once from resolved stream specs (validation happens here, so it
    fires at bridge configure / writer open / runner setup). All assemble and
    split operations align positionally with ``self.specs`` — specs contain
    list fields and are unhashable, and (key, topic) is not unique (the same
    topic may appear in several specs), so position is the identity.
    """

    def __init__(self, specs: Sequence[StreamSpec]):
        self.specs: tuple[StreamSpec, ...] = tuple(specs)

        by_key: dict[str, list[StreamSpec]] = {}
        for spec in self.specs:
            by_key.setdefault(spec.key, []).append(spec)

        layouts: dict[str, KeyLayout] = {}
        for key, group in by_key.items():
            shared = len(group) > 1
            categories = {_spec_category(s) for s in group}
            if shared and categories != {"numeric"}:
                raise ContractValidationError(
                    f"Key '{key}' is shared by {len(group)} specs but is not "
                    f"all-numeric ({sorted(categories)}). Images and strings "
                    f"cannot be concatenated; give each its own key."
                )
            dtypes = {resolve_stream_dtype(s) for s in group}
            if len(dtypes) > 1:
                raise ContractValidationError(
                    f"Key '{key}' is shared by specs with different dtypes "
                    f"({sorted(dtypes)}). Specs sharing a key must resolve to "
                    f"one dtype; set dtype: explicitly to align them."
                )
            if shared:
                nameless = [s.source.channel.topic for s in group if not s.names]
                if nameless:
                    raise ContractValidationError(
                        f"Key '{key}' is shared by multiple specs, but "
                        f"{nameless} have no select. Concatenation needs "
                        f"static dims; add select: to every spec of a shared key."
                    )

            category = _spec_category(group[0])
            dtype = next(iter(dtypes))
            if category == "numeric" and dtype not in DTYPE_MAP:
                raise ContractValidationError(
                    f"Unsupported dtype '{dtype}' for key '{key}'. Supported: {', '.join(sorted(DTYPE_MAP))}."
                )

            slices = []
            start = 0
            for s in group:
                dim = max(len(s.names), 1)
                slices.append(SpecSlice(spec=s, start=start, dim=dim))
                start += dim
            layouts[key] = KeyLayout(key=key, category=category, dtype=dtype, slices=tuple(slices))

        self._layouts = layouts
        self.keys: tuple[str, ...] = tuple(layouts)  # first-occurrence order

    def __getitem__(self, key: str) -> KeyLayout:
        return self._layouts[key]

    def __contains__(self, key: str) -> bool:
        return key in self._layouts

    # -------------------- assemble / split --------------------

    def assemble(self, values: Sequence[Any]) -> dict[str, Any]:
        """Assemble a frame dict from per-spec values.

        ``values[i]`` pairs with ``self.specs[i]`` (the resampled output for
        that stream at one tick: a numpy array, a string, or None when the
        stream had no data). Numeric specs sharing a key concatenate in
        declaration order; missing values zero-fill at the spec's static dim.
        Returns {contract_key: np.ndarray | str}.
        """
        if len(values) != len(self.specs):
            raise ValueError(f"assemble() got {len(values)} values for {len(self.specs)} specs")

        # Regroup values by key, preserving spec order (positional pairing).
        by_key: dict[str, list[Any]] = {}
        for spec, value in zip(self.specs, values, strict=False):
            by_key.setdefault(spec.key, []).append(value)

        frame: dict[str, Any] = {}
        for key, key_values in by_key.items():
            layout = self._layouts[key]

            if layout.category == "image":
                val = key_values[0]
                frame[key] = (
                    np.asarray(val, dtype=np.uint8) if val is not None else zeros_for_spec(layout.slices[0].spec)
                )
            elif layout.category == "string":
                val = key_values[0]
                frame[key] = str(val) if val is not None else ""
            else:
                np_dtype = layout.np_dtype
                parts = []
                for sl, val in zip(layout.slices, key_values, strict=False):
                    if val is None:
                        parts.append(np.zeros(sl.dim, dtype=np_dtype))
                    else:
                        parts.append(np.asarray(val, dtype=np_dtype).flatten())
                frame[key] = np.concatenate(parts) if len(parts) > 1 else parts[0]

        return frame

    def split(self, frame: dict[str, Any]) -> list[np.ndarray]:
        """Split a per-key frame into per-spec arrays aligned with self.specs.

        Inverse of the numeric concatenation in :meth:`assemble`. Validates
        each key's vector length against the layout (skipped only for a
        select-less single-spec key, whose dim is not statically known).
        Raises KeyError for a missing key, ValueError for a length mismatch.
        """
        per_key_parts: dict[str, list[np.ndarray]] = {}
        for key, layout in self._layouts.items():
            if key not in frame:
                raise KeyError(f"Action frame missing key '{key}'")
            np_dtype = layout.np_dtype if layout.category == "numeric" else np.float64
            arr = np.asarray(frame[key], dtype=np_dtype).flatten()

            dynamic_dim = len(layout.slices) == 1 and not layout.slices[0].spec.names
            if not dynamic_dim and arr.size != layout.dim:
                raise ValueError(
                    f"Key '{key}' expects a vector of length {layout.dim} "
                    f"(sum of per-spec select dims), got {arr.size}."
                )
            if dynamic_dim:
                per_key_parts[key] = [arr]
            else:
                per_key_parts[key] = [arr[sl.start : sl.end] for sl in layout.slices]

        # Emit in spec order (positional contract).
        consumed: dict[str, int] = {}
        out: list[np.ndarray] = []
        for spec in self.specs:
            i = consumed.get(spec.key, 0)
            out.append(per_key_parts[spec.key][i])
            consumed[spec.key] = i + 1
        return out

    # -------------------- features --------------------

    def lerobot_features(self) -> dict[str, dict[str, Any]]:
        """Per-key LeRobot feature dicts covering the full concatenated vectors.

        Images/strings delegate to build_feature; numeric keys declare shape
        (key.dim,) with the namespaced selector names of every slice.

        LeRobot is the only backend consuming this format; starVLA and
        vla_foundry derive their schemas from KeyLayout directly (see their
        grouping modules). It lives here, not in lerobot_robot_rosetta, so the
        declared image shape and zeros_for_spec's zero-fill shape agree via
        DECODED_IMAGE_CHANNELS by construction.
        """
        feats: dict[str, dict[str, Any]] = {}
        for key, layout in self._layouts.items():
            if layout.category in ("image", "string"):
                feats[key] = build_feature(layout.slices[0].spec)
                continue
            names: list[str] = []
            for sl in layout.slices:
                names.extend(get_namespaced_names(sl.spec))
            feats[key] = {
                "dtype": layout.dtype,
                "shape": (layout.dim,),
                "names": names or None,
            }
        return feats
