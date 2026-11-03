# Copyright 2026 Isaac Blankenau
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

"""Exact normalization statistics over contract frames.

A dataset backend computes stats on the frames the contract produced, after
its transforms, so training normalizes the values the policy sees live.
:func:`sufficient_statistics` reduces frames to what exact stats need, and
:func:`frame_stats` turns that into, per key:

- a numeric key: ``count``, ``mean``, ``std`` (population), ``min``, ``max``
  and the quantiles in :data:`QUANTILES` (numpy's linear interpolation), per
  element, in the contract's units;
- an image key: the same per channel, in pixel values 0..255;
- a string key: nothing.

``count`` is the number of frames. Sufficient statistics from several episodes
merge exactly: numeric values by concatenation, histograms by summing, which
:func:`merge_stats` does.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ..frames.layout import FrameLayout

__all__ = [
    "LEVELS",
    "QUANTILES",
    "frame_stats",
    "histogram_stats",
    "merge_stats",
    "quantile_name",
    "sufficient_statistics",
    "value_stats",
]

QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
"""The quantiles reported, named ``q01`` ... ``q99`` (LeRobot's defaults and names)."""

LEVELS = 256
"""Pixel values an image histogram counts (uint8)."""


def value_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    """Stats of ``(count, *shape)`` values, per element."""
    stats = {
        "count": np.array([len(values)]),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "min": values.min(axis=0),
        "max": values.max(axis=0),
    }
    for q, value in zip(QUANTILES, np.quantile(values, QUANTILES, axis=0), strict=True):
        stats[quantile_name(q)] = value
    return stats


def histogram_stats(histograms: np.ndarray, frames: int) -> dict[str, np.ndarray]:
    """Per-channel stats of uint8 pixels from ``(channels, 256)`` histograms, as if computed on every pixel.

    Quantiles follow numpy's linear interpolation between the order statistics
    the histogram holds.
    """
    levels = np.arange(LEVELS)
    pixels = histograms.sum(axis=1)
    mean = histograms @ levels / pixels
    stats = {
        "count": np.array([frames]),
        "mean": mean,
        "std": np.sqrt((histograms * (levels - mean[:, None]) ** 2).sum(axis=1) / pixels),
        "min": np.array([np.flatnonzero(h)[0] for h in histograms], dtype=np.float64),
        "max": np.array([np.flatnonzero(h)[-1] for h in histograms], dtype=np.float64),
    }
    cumulative = histograms.cumsum(axis=1)
    for q in QUANTILES:
        value = []
        for c in range(len(histograms)):
            position = q * (pixels[c] - 1)
            lo = int(np.floor(position))
            hi = min(lo + 1, pixels[c] - 1)
            at = np.searchsorted(cumulative[c], [lo, hi], side="right")
            value.append(at[0] + (position - lo) * (at[1] - at[0]))
        stats[quantile_name(q)] = np.array(value, dtype=np.float64)
    return stats


def quantile_name(q: float) -> str:
    return f"q{round(q * 100):02d}"


def sufficient_statistics(frames: Iterable[dict[str, Any]], layout: FrameLayout) -> tuple[int, dict[str, np.ndarray]]:
    """The frame count and, per key, what exact stats need: numeric values stacked, image histograms.

    A numeric key gives its values, ``(frames, *shape)``. An image key gives
    per-channel histograms of its uint8 pixels, ``(channels, 256)``. String
    keys give nothing.

    Raises:
        RuntimeError: If ``frames`` is empty.

    """
    values: dict[str, list[np.ndarray]] = {}
    histograms: dict[str, np.ndarray] = {}
    count = 0
    for frame in frames:
        count += 1
        for key in layout.keys:
            category = layout[key].category
            if category == "numeric":
                values.setdefault(key, []).append(np.asarray(frame[key], dtype=np.float64))
            elif category == "image":
                image = frame[key]
                channels = image.shape[-1]
                hist = histograms.setdefault(key, np.zeros((channels, LEVELS), dtype=np.int64))
                for c in range(channels):
                    hist[c] += np.bincount(image[..., c].ravel(), minlength=LEVELS)
    if not count:
        raise RuntimeError("Episode has no frames to compute stats from")
    return count, {**{key: np.stack(rows) for key, rows in values.items()}, **histograms}


def frame_stats(frames: Iterable[dict[str, Any]], layout: FrameLayout) -> dict[str, dict[str, np.ndarray]]:
    """Exact stats, per key of ``layout``, over every frame in ``frames``."""
    return merge_stats([sufficient_statistics(frames, layout)], layout)


def merge_stats(
    parts: Iterable[tuple[int, dict[str, np.ndarray]]], layout: FrameLayout
) -> dict[str, dict[str, np.ndarray]]:
    """Exact stats, per key of ``layout``, over the frames behind several :func:`sufficient_statistics` results.

    Raises:
        ValueError: If ``parts`` is empty.

    """
    parts = list(parts)
    if not parts:
        raise ValueError("No sufficient statistics to merge")
    count = sum(n for n, _ in parts)
    stats: dict[str, dict[str, np.ndarray]] = {}
    for key in layout.keys:
        category = layout[key].category
        if category == "image":
            stats[key] = histogram_stats(sum(payload[key] for _, payload in parts), count)
        elif category == "numeric":
            stats[key] = value_stats(np.concatenate([payload[key] for _, payload in parts]))
    return stats
