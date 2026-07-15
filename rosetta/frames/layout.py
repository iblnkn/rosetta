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
``observation.state``). Their values concatenate into one flat vector. This
module is the single source of truth for that layout:

- key order is first-occurrence order of the spec list
- within a key, specs concatenate in declaration order
- each spec occupies a static slice sized ``spec.dim`` (= max(len(names), 1),
  where a select-less numeric stream is a scalar, enforced at assembly)

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
(LeRobot features, starVLA modality.json, vla_foundry field dims) goes through
a FrameLayout, so the layouts agree by construction.

The dtype vocabulary lives in ``frames.codecs`` (SUPPORTED_NUMERIC_DTYPES /
SPECIAL_DTYPES, lerobot-compatible by choice, and rejected at load, so this
module's own dtype check is only a backstop for hand-built specs). This module
has no lerobot dependency. Backends derive their own schema views from
KeyLayout (starVLA / vla_foundry grouping). lerobot_features() is the one
lerobot-format emitter, kept here so the declared feature shapes and the
zero-fill shapes share one source of truth.

The assemble / split dtype asymmetry is deliberate. Do not "unify" it.
``assemble`` casts to the key's declared dtype. It targets dataset storage,
decoders already produce that dtype, and observations have no finiteness gate.
``split`` always emits float64. Its output feeds ``encode_value``, whose
NaN/Inf gate inspects float64 and whose encoders do the wire-type cast. Casting
to an int dtype here would launder a policy's NaN into INT_MIN, finite garbage
that sails through the gate onto hardware.

Pure numpy. No ROS dependency.
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
from .codecs import SUPPORTED_NUMERIC_DTYPES, width_mismatch_message

# All supported image encodings are normalized to 3-channel uint8 RGB by the
# decoders (3ch color as-is, 4ch drops alpha, mono->rgb). Depth/RGBD encodings
# are explicitly rejected at decode (see decoders.DepthEncodingNotSupported).
# The decoded output channel count is therefore always 3, regardless of the
# source encoding. This is the single source of truth for the feature shape
# and the zero-fill shape so they agree.
DECODED_IMAGE_CHANNELS = 3


def build_feature(spec: ObservationStreamSpec) -> dict[str, Any]:
    """
    Build the LeRobot feature dict for an image or string spec.

    Numeric features come from FrameLayout.lerobot_features() only. A numeric
    key's shape and names must cover the whole concatenated vector (with
    namespacing), which a single spec cannot know.

    The ObservationStreamSpec signature is deliberate. Image and string streams
    are observation-shaped by construction (``is_image`` exists only under
    observation.images.*, and action-shaped specs are forced numeric at spec
    resolution), and ``image_resize`` exists only on ObservationStreamSpec.
    """
    if spec.is_image:
        if not spec.image_resize:
            raise ValueError(f"Image spec '{spec.key}' must have image_resize")
        h, w = spec.image_resize
        return {
            "dtype": "video",
            "shape": (h, w, DECODED_IMAGE_CHANNELS),
            "names": ["height", "width", "channels"],
        }

    if spec.dtype == "string":
        return {"dtype": "string", "shape": (1,), "names": None}

    raise ValueError(
        f"build_feature() covers image/string specs only; numeric key "
        f"'{spec.key}' gets its feature from FrameLayout.lerobot_features()."
    )


def zero_image_for_spec(spec: ObservationStreamSpec) -> np.ndarray:
    """(H, W, DECODED_IMAGE_CHANNELS) uint8 zeros for a missing image frame.

    Must match both the shape of a real decoded frame and the feature
    declared by build_feature. Numeric zero-fill lives inline in
    FrameLayout.assemble (it is per-slice, not per-spec-alone).
    """
    if spec.image_resize is None:
        raise ValueError(f"Image spec '{spec.key}' must have image_resize")
    h, w = spec.image_resize
    return np.zeros((h, w, DECODED_IMAGE_CHANNELS), dtype=np.uint8)


def _spec_category(spec: StreamSpec) -> str:
    """'image' | 'string' | 'numeric' for layout purposes.

    ``is_image`` and ``dtype`` are base-class fields, so action specs answer
    ``is_image=False`` honestly rather than by attribute absence.
    """
    if spec.is_image:
        return "image"
    if spec.dtype == "string":
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
    def is_action(self) -> bool:
        """True when this key is fed by action-shaped specs (policy output).

        Role comes from the spec's type, i.e. which contract section declared
        it, never from the key's spelling. Valid on any slice because
        FrameLayout rejects mixed-role keys at construction.
        """
        return isinstance(self.slices[0].spec, ActionStreamSpec)

    @property
    def np_dtype(self) -> Any:
        """Numpy dtype for a numeric key. Image/string keys have none."""
        if self.category != "numeric":
            raise ValueError(
                f"Key '{self.key}' is category '{self.category}' ('{self.dtype}'); "
                f"np_dtype applies to numeric keys only."
            )
        return SUPPORTED_NUMERIC_DTYPES[self.dtype]


class FrameLayout:
    """Canonical key -> (spec, offset, dim) layout for an ordered spec list.

    Built once from resolved stream specs (validation happens here, so it fires
    at bridge configure / writer open / runner setup). All assemble and split
    operations align positionally with ``self.specs``. Specs are unhashable
    (``source`` holds a dict-valued ``qos``), and (key, topic) is not unique
    (one topic may appear in several specs), so position is the identity.
    """

    def __init__(self, specs: Sequence[StreamSpec]):
        self.specs: tuple[StreamSpec, ...] = tuple(specs)

        by_key: dict[str, list[StreamSpec]] = {}
        for spec in self.specs:
            by_key.setdefault(spec.key, []).append(spec)

        layouts: dict[str, KeyLayout] = {}
        for key, group in by_key.items():
            shared = len(group) > 1
            # A key must belong to exactly one role: mixing action-shaped and
            # observation-shaped specs would make the key's routing (state
            # column vs action column) depend on which slice a consumer
            # inspects. Base StreamSpec counts as observation-shaped.
            action_topics = [s.source.channel.topic for s in group if isinstance(s, ActionStreamSpec)]
            if action_topics and len(action_topics) < len(group):
                obs_topics = [s.source.channel.topic for s in group if not isinstance(s, ActionStreamSpec)]
                raise ContractValidationError(
                    f"Key '{key}' is shared by observation-shaped specs "
                    f"({obs_topics}) and action-shaped specs ({action_topics}); "
                    f"a frame key must belong to exactly one role."
                )
            categories = {_spec_category(s) for s in group}
            if shared and categories != {"numeric"}:
                raise ContractValidationError(
                    f"Key '{key}' is shared by {len(group)} specs but is not "
                    f"all-numeric ({sorted(categories)}). Images and strings "
                    f"cannot be concatenated; give each its own key."
                )
            dtypes = {s.dtype for s in group}
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

            # Rendered feature names must be unique per key: two sources can
            # legally flatten to the same name via different (namespace, name)
            # pairs (e.g. namespace None + "b.data" vs namespace "b" + "data"),
            # which would silently collide in lerobot_features().
            seen_names: dict[str, str] = {}
            for s in group:
                for name in s.namespaced_names:
                    other = seen_names.get(name)
                    if other is not None:
                        raise ContractValidationError(
                            f"Key '{key}': rendered feature name '{name}' from topic "
                            f"{s.source.channel.topic} collides with the same name from "
                            f"topic {other}. Rendered names must be unique within a key."
                        )
                    seen_names[name] = s.source.channel.topic

            category = _spec_category(group[0])
            dtype = next(iter(dtypes))
            # Backstop for hand-built specs. Contract-resolved dtypes are
            # already vocabulary-checked at load (schema._validate_dtype).
            if category == "numeric" and dtype not in SUPPORTED_NUMERIC_DTYPES:
                raise ContractValidationError(
                    f"Unsupported dtype '{dtype}' for key '{key}'. "
                    f"Supported: {', '.join(sorted(SUPPORTED_NUMERIC_DTYPES))}."
                )

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

    # -------------------- assemble / split --------------------

    def assemble(self, values: Sequence[Any]) -> dict[str, Any]:
        """Assemble a frame dict from per-spec values.

        ``values[i]`` pairs with ``self.specs[i]`` (the resampled output for
        that stream at one tick: a numpy array, a string, or None when the
        stream had no data). Numeric specs sharing a key concatenate in
        declaration order. Missing values zero-fill at the spec's static dim.
        Returns {contract_key: np.ndarray | str}.
        """
        if len(values) != len(self.specs):
            raise ValueError(f"assemble() got {len(values)} values for {len(self.specs)} specs")

        # Regroup values by key, preserving spec order (positional pairing).
        by_key: dict[str, list[Any]] = {}
        for spec, value in zip(self.specs, values, strict=True):
            by_key.setdefault(spec.key, []).append(value)

        frame: dict[str, Any] = {}
        for key, key_values in by_key.items():
            layout = self._layouts[key]

            if layout.category == "image":
                val = key_values[0]
                frame[key] = (
                    np.asarray(val, dtype=np.uint8) if val is not None else zero_image_for_spec(layout.slices[0].spec)
                )
            elif layout.category == "string":
                val = key_values[0]
                if val is not None and not isinstance(val, str):
                    raise ValueError(
                        f"String stream '{key}' produced {type(val).__name__}; its decoder must return str"
                    )
                frame[key] = val if val is not None else ""
            else:
                np_dtype = layout.np_dtype
                parts = []
                for sl, val in zip(layout.slices, key_values, strict=True):
                    if val is None:
                        parts.append(np.zeros(sl.dim, dtype=np_dtype))
                        continue
                    part = np.asarray(val, dtype=np_dtype).flatten()
                    if part.size != sl.dim:
                        raise ValueError(
                            width_mismatch_message(
                                "Stream value",
                                key,
                                sl.spec.source.channel.topic,
                                part.size,
                                sl.dim,
                                sl.spec.names,
                            )
                        )
                    parts.append(part)
                frame[key] = np.concatenate(parts) if len(parts) > 1 else parts[0]

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
        """Per-key LeRobot feature dicts covering the full concatenated vectors.

        Images and strings delegate to build_feature. Numeric keys declare
        shape (key.dim,) with the namespaced selector names of every slice.

        LeRobot is the only backend consuming this format. starVLA and
        vla_foundry derive their schemas from KeyLayout directly (see their
        grouping modules). It lives here, not in lerobot_robot_rosetta, so the
        declared image shape and zero_image_for_spec's zero-fill shape agree
        via DECODED_IMAGE_CHANNELS by construction.
        """
        feats: dict[str, dict[str, Any]] = {}
        for key, layout in self._layouts.items():
            if layout.category in ("image", "string"):
                feats[key] = build_feature(layout.slices[0].spec)
                continue
            names: list[str] = []
            for sl in layout.slices:
                names.extend(sl.spec.namespaced_names)
            feats[key] = {
                "dtype": layout.dtype,
                "shape": (layout.dim,),
                "names": names or None,
            }
        return feats
