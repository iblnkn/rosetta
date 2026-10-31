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

Every frame value except a string is a tensor with a static shape and dtype.
A stream's shape comes from the contract (``spec.shape``): an image is
``(h, w, 3)``, a stream with ``select`` is a vector of its names, and a stream
without one takes its declared ``shape`` or is a scalar. Multiple specs may
share one output key (e.g. two topics feeding ``observation.state``). Their
vectors concatenate into one. This module is the single source of truth for
that layout:

- key order is first-occurrence order of the spec list
- within a key, specs concatenate in declaration order
- each spec occupies a static slice sized ``spec.dim`` (= prod(spec.shape))

Layout is purely positional (list / declaration order), by design. Placing a
source at a chosen offset in the concatenated vector needs *some* explicit
signal, and order is the only one that adds no second mechanism on top of what
YAML already gives for free. A list is already ordered, and reordering two
sources in the contract is a one-line diff. An explicit ordering key (e.g.
`position: 2`) would duplicate that ordering in a second place, invite drift
between the two, and still break ties by list order. Order is the whole
mechanism, not a limit to work around.

Everything that assembles frames (live bridge, bag porter), splits action
frames back into per-stream slices, or derives per-key offsets for a backend
(LeRobot features, other adapters' schema views) goes through
a FrameLayout, so the layouts agree by construction.

The dtype vocabulary lives in ``frames.codecs`` (SUPPORTED_NUMERIC_DTYPES /
SPECIAL_DTYPES, lerobot-compatible by choice, and rejected at load, so this
module's own dtype check is only a backstop for hand-built specs). This module
has no lerobot dependency. Backends derive their own schema views from
KeyLayout. lerobot_features() is the one
lerobot-format emitter, kept here so the declared feature shapes and the
zero-fill shapes share one source of truth.

The assemble / split dtype asymmetry is deliberate. Do not "unify" it.
``assemble`` casts to the key's declared dtype. It targets dataset storage,
decoders already produce that dtype, and observations have no finiteness gate.
``split`` always emits float64. Its output feeds ``encode_value``, whose
NaN/Inf gate inspects float64 and whose encoders do the wire-type cast. Casting
to an int dtype here would launder a policy's NaN into INT_MIN, finite garbage
that sails through the gate onto hardware.

A value the transport delivered in device memory stays a device array (see
:mod:`rosetta.arrays`) when it alone fills its key. Its decoder produced the key's
dtype, which assembly checks rather than casts. A key that concatenates
several specs is numpy, as are missing values. No ROS dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..arrays import is_device_array, to_host
from ..codecs import SUPPORTED_NUMERIC_DTYPES, width_mismatch_message
from ..contract.specs import (
    ActionStreamSpec,
    ObservationStreamSpec,
    StreamSpec,
    spec_category,
    validate_key_groups,
)

__all__ = [
    "FrameLayout",
    "KeyLayout",
    "SpecSlice",
    "build_feature",
]


def build_feature(spec: ObservationStreamSpec) -> dict[str, Any]:
    """
    Build the LeRobot feature dict for an image or string spec.

    Numeric features come from FrameLayout.lerobot_features() only. A numeric
    key's shape and names must cover the whole concatenated value (with
    namespacing), which a single spec cannot know.

    The ObservationStreamSpec signature is deliberate. Image and string streams
    are observation-shaped by construction (``is_image`` exists only under
    observation.images.*, and action-shaped specs are forced numeric at spec
    resolution), and ``image_resize`` exists only on ObservationStreamSpec.
    """
    if spec.is_image:
        if not spec.image_resize:
            raise ValueError(f"Image spec '{spec.key}' must have image_resize")
        return {"dtype": "video", "shape": spec.shape, "names": ["height", "width", "channels"]}

    if spec.dtype == "string":
        return {"dtype": "string", "shape": (1,), "names": None}

    raise ValueError(
        f"build_feature() covers image/string specs only; numeric key "
        f"'{spec.key}' gets its feature from FrameLayout.lerobot_features()."
    )


def _same_dtype(val: Any, np_dtype: np.dtype) -> bool:
    """True if ``val``'s dtype is ``np_dtype``. Array libraries spell it alike after their prefix (torch.uint8)."""
    return str(val.dtype).rsplit(".", 1)[-1] == np_dtype.name


def _tensor_part(key: str, sl: "SpecSlice", val: Any, np_dtype: np.dtype) -> Any:
    """One spec's value at its static shape.

    Zeros if missing, a device array as is, otherwise numpy cast to the key's dtype.
    """
    shape = sl.spec.shape
    if val is None:
        return np.zeros(shape, dtype=np_dtype)
    if is_device_array(val):
        # Casting would copy it to the host, so the decoder must have produced the key's dtype.
        if not _same_dtype(val, np_dtype):
            raise ValueError(f"Stream '{key}' produced device dtype {val.dtype}, not {np_dtype.name}")
        if tuple(val.shape) != shape:
            raise ValueError(f"Stream '{key}' produced device shape {tuple(val.shape)}, not {shape}")
        return val
    part = np.asarray(val, dtype=np_dtype)
    if part.size != sl.dim:
        raise ValueError(
            width_mismatch_message("Stream value", key, sl.spec.source.channel.topic, part.size, sl.dim, sl.spec.names)
        )
    return part.reshape(shape)


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
    def names(self) -> tuple[str, ...]:
        """Namespaced selector names of every slice, in order. Empty for a slice without select."""
        return tuple(name for sl in self.slices for name in sl.spec.namespaced_names)

    @property
    def shape(self) -> tuple[int, ...]:
        """The key's value shape: its one spec's shape, or the concatenated vector of several."""
        if len(self.slices) == 1:
            return self.slices[0].spec.shape
        return (self.dim,)

    @property
    def is_action(self) -> bool:
        """True when this key is fed by action-shaped specs (policy output).

        Role comes from the spec's type, i.e. which contract section declared
        it, never from the key's spelling. Valid on any slice because
        FrameLayout rejects mixed-role keys at construction.
        """
        return isinstance(self.slices[0].spec, ActionStreamSpec)

    @property
    def np_dtype(self) -> Any:
        """Numpy dtype of the key's values: uint8 for an image. String keys have none."""
        if self.category == "image":
            return np.dtype(np.uint8)
        if self.category != "numeric":
            raise ValueError(
                f"Key '{self.key}' is category '{self.category}' ('{self.dtype}'); "
                f"np_dtype applies to image and numeric keys only."
            )
        return np.dtype(SUPPORTED_NUMERIC_DTYPES[self.dtype])


class FrameLayout:
    """Canonical key -> (spec, offset, dim) layout for an ordered spec list.

    Built once from resolved stream specs. Construction runs
    :func:`~rosetta.contract.specs.validate_key_groups` again, the same check
    contract loading runs, as a backstop for hand-built specs. All assemble and split
    operations align positionally with ``self.specs``. Specs are unhashable
    (``source`` holds a dict-valued ``qos``), and (key, topic) is not unique
    (one topic may appear in several specs), so position is the identity.
    """

    def __init__(self, specs: Sequence[StreamSpec]):
        self.specs: tuple[StreamSpec, ...] = tuple(specs)

        layouts: dict[str, KeyLayout] = {}
        for key, group in validate_key_groups(self.specs).items():
            category = spec_category(group[0])
            dtype = group[0].dtype
            slices = []
            start = 0
            for s in group:
                slices.append(SpecSlice(spec=s, start=start, dim=s.dim))
                start += s.dim
            layouts[key] = KeyLayout(key=key, category=category, dtype=dtype, slices=tuple(slices))

        self._layouts = layouts
        self.keys: tuple[str, ...] = tuple(layouts)  # first-occurrence order

    def __getitem__(self, key: str) -> KeyLayout:
        return self._layouts[key]

    @property
    def nameless_specs(self) -> tuple[StreamSpec, ...]:
        """Numeric specs without select names: scalars, and streams with a declared shape.

        A consumer that addresses values by name (LeRobot's per-name floats)
        can't carry them.
        """
        return tuple(
            sl.spec
            for layout in self._layouts.values()
            if layout.category == "numeric"
            for sl in layout.slices
            if not sl.spec.names
        )

    # -------------------- assemble / split --------------------

    def assemble(self, values: Sequence[Any], *, fill_missing: bool = True) -> dict[str, Any]:
        """Assemble a frame dict from per-spec values.

        ``values[i]`` pairs with ``self.specs[i]`` (the resampled output for
        that stream at one tick: a numpy array, a string, or None when the
        stream had no data). Each value takes its spec's static shape, cast to
        the key's dtype. Numeric specs sharing a key concatenate in declaration
        order. Missing values zero-fill; with ``fill_missing=False`` a key with
        any missing value is left out. Returns {contract_key: array | str}:
        numpy, or a device array for a single-spec key whose value is one (see
        :mod:`rosetta.arrays`).
        """
        if len(values) != len(self.specs):
            raise ValueError(f"assemble() got {len(values)} values for {len(self.specs)} specs")

        # Regroup values by key, preserving spec order (positional pairing).
        by_key: dict[str, list[Any]] = {}
        for spec, value in zip(self.specs, values, strict=True):
            by_key.setdefault(spec.key, []).append(value)

        frame: dict[str, Any] = {}
        for key, key_values in by_key.items():
            if not fill_missing and any(val is None for val in key_values):
                continue
            layout = self._layouts[key]

            if layout.category == "string":
                val = key_values[0]
                if val is not None and not isinstance(val, str):
                    raise ValueError(
                        f"String stream '{key}' produced {type(val).__name__}; its decoder must return str"
                    )
                frame[key] = val if val is not None else ""
                continue
            np_dtype = layout.np_dtype
            parts = [_tensor_part(key, sl, val, np_dtype) for sl, val in zip(layout.slices, key_values, strict=True)]
            # Concatenation runs on the host: rosetta has no device concatenate,
            # and keys several specs share are small vectors.
            frame[key] = parts[0] if len(parts) == 1 else np.concatenate([to_host(p) for p in parts])

        return frame

    def split(self, frame: dict[str, Any]) -> list[np.ndarray]:
        """Split a per-key frame into per-spec float64 arrays aligned with self.specs.

        Inverse of the numeric concatenation in :meth:`assemble`. Always emits
        float64, never the key's declared dtype, so a policy's NaN/Inf reaches
        encode_value's finiteness gate intact instead of being laundered into a
        finite integer by the cast (encoders do the wire-type cast, after the
        gate). Validates every key's vector length against the layout's static
        dims. Raises KeyError for a missing key, ValueError for an unknown key
        or a length mismatch.
        """
        per_key_parts: dict[str, list[np.ndarray]] = {}
        for key, layout in self._layouts.items():
            if key not in frame:
                raise KeyError(f"Action frame missing key '{key}'")
            arr = np.asarray(frame[key], dtype=np.float64).flatten()
            if arr.size != layout.dim:
                raise ValueError(
                    f"Key '{key}' expects a vector of length {layout.dim} "
                    f"(sum of per-spec select dims), got {arr.size}."
                )
            per_key_parts[key] = [arr[sl.start : sl.end] for sl in layout.slices]

        extra = set(frame) - set(self._layouts)
        if extra:
            raise ValueError(f"Action frame has unknown key(s) {sorted(extra)}; this layout has {list(self.keys)}.")

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
        """Per-key LeRobot feature dicts covering the full concatenated values.

        Images and strings delegate to build_feature. Numeric keys declare the
        key's shape, with the namespaced selector names of every slice when
        they have them.

        LeRobot is the only backend consuming this format. Other adapters
        derive their schemas from KeyLayout directly. It lives here, not in
        lerobot_robot_rosetta, so the declared shapes and the zero-fill shapes
        both come from ``spec.shape`` by construction.
        """
        feats: dict[str, dict[str, Any]] = {}
        for key, layout in self._layouts.items():
            if layout.category in ("image", "string"):
                feats[key] = build_feature(layout.slices[0].spec)
                continue
            feats[key] = {
                "dtype": layout.dtype,
                "shape": layout.shape,
                "names": list(layout.names) or None,
            }
        return feats
