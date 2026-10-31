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

"""Exact normalization statistics, stored as a Rerun layer beside the recordings.

Stats depend on the recordings and on the contract, which picks and transforms
the values. So each contract gets its own layer: one ``.rrd`` per segment,
registered under ``rosetta_stats_<hash of the contract file>``. Its property of
the same name holds that episode's sufficient statistics, one field per
contract key, and a ``<layer>_meta`` property holds the frame count and the
contract text:

- a numeric key: the episode's frame values themselves (frames x width);
- an image key: per-channel histograms of the decoded uint8 pixels
  (channels x 256), after the contract's ``resize``;
- a string key: nothing.

Loading merges those across segments, values by concatenation and histograms by
summing, so every statistic is exact over every frame: count, mean, std
(population), min, max, and the quantiles in :data:`QUANTILES` (numpy's linear
interpolation). The recordings themselves are never touched, and new episodes
only need their own layer files.

Frames are each episode's porter frames: unwindowed, every tick after warmup.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import rerun as rr
from rerun.experimental.dataloader import NoShuffle

from rosetta.contract.schema import load_contract
from rosetta.contract.specs import iter_policy_specs
from rosetta.frames.layout import FrameLayout

from .dataset import ContractFrames

if TYPE_CHECKING:
    from rerun.catalog import DatasetEntry

    from .layout import RecordingLayout

#: The quantiles reported, named ``q01`` ... ``q99`` (LeRobot's defaults and names).
QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)

LEVELS = 256  # uint8 pixel values


def stats_layer_name(contract_text: str) -> str:
    """The layer holding stats for the contract whose file text is ``contract_text``."""
    return f"rosetta_stats_{hashlib.sha256(contract_text.encode()).hexdigest()[:12]}"


def write_stats_layer(
    dataset_entry: DatasetEntry,
    contract_path: str | Path,
    out_dir: str | Path,
    *,
    token: str | None = None,
    layout: RecordingLayout | None = None,
) -> list[str]:
    """Compute and register the stats layer for every segment that lacks it. Returns those segments.

    Each segment's layer file is written to ``out_dir/<layer>/<segment>.rrd``
    and registered from there, so a ``file://`` catalog needs ``out_dir`` on
    the server's filesystem. ``token`` is the catalog's, for reading the frames.
    """
    text = Path(contract_path).read_text()
    contract = load_contract(contract_path)
    name = stats_layer_name(text)
    frame_layout = FrameLayout(list(iter_policy_specs(contract)))
    todo = sorted(segment for segment, layers in _segment_layers(dataset_entry).items() if name not in layers)

    layer_dir = Path(out_dir) / name
    layer_dir.mkdir(parents=True, exist_ok=True)
    uris = []
    for segment in todo:
        frames = ContractFrames(
            dataset_entry.catalog.url,
            dataset_entry.name,
            contract,
            token=token,
            segments=[segment],
            shuffle_strategy=NoShuffle(),
            layout=layout,
        )
        count, payload = _sufficient_statistics(frames, frame_layout)
        path = layer_dir / f"{segment}.rrd"
        with rr.RecordingStream("rosetta_stats", recording_id=segment, send_properties=False) as rec:
            rec.save(path)
            rec.send_property(name, rr.AnyValues(**{key: value.ravel() for key, value in payload.items()}))
            rec.send_property(f"{name}_meta", rr.AnyValues(frames=count, contract=text))
        uris.append(path.resolve().as_uri())
    if uris:
        dataset_entry.register(uris, layer_name=name).wait()
    return todo


def load_stats(dataset_entry: DatasetEntry, contract_path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Exact stats over every segment, per contract key, from the contract's stats layer.

    Numeric keys keep the contract's units, shape ``(width,)``. Image keys are
    per channel in pixel values 0..255, shape ``(channels,)``. ``count`` is the
    number of frames.

    Raises:
        ValueError: If a segment has no stats layer for this contract.

    """
    text = Path(contract_path).read_text()
    name = stats_layer_name(text)
    frame_layout = FrameLayout(list(iter_policy_specs(load_contract(contract_path))))
    missing = sorted(segment for segment, layers in _segment_layers(dataset_entry).items() if name not in layers)
    if missing:
        raise ValueError(f"Segments without the stats layer {name}: {missing}. Run write_stats_layer first.")

    # The dataframe reader keeps property field names as logged. The segment
    # table rewrites dots in them, and contract keys have dots.
    table = (
        dataset_entry.filter_contents([f"/__properties/{name}", f"/__properties/{name}_meta"])
        .reader(index=None)
        .to_arrow_table()
    )
    column = lambda prop, field: table.column(f"property:{prop}:{field}").to_pylist()  # noqa: E731

    frames = sum(int(row[0]) for row in column(f"{name}_meta", "frames"))
    stats: dict[str, dict[str, np.ndarray]] = {}
    for key in frame_layout.keys:
        category = frame_layout[key].category
        if category == "string":
            continue
        rows = [np.asarray(row) for row in column(name, key)]
        if category == "image":
            stats[key] = histogram_stats(sum(row.reshape(-1, LEVELS) for row in rows), frames)
        else:
            stats[key] = value_stats(np.concatenate([row.reshape(-1, *frame_layout[key].shape) for row in rows]))
    return stats


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


def _sufficient_statistics(frames: Iterable[dict[str, Any]], layout: FrameLayout) -> tuple[int, dict[str, np.ndarray]]:
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


def _segment_layers(dataset_entry: DatasetEntry) -> dict[str, list[str]]:
    table = dataset_entry.segment_table().to_arrow_table()
    segments = table.column("rerun_segment_id").to_pylist()
    return dict(zip(segments, table.column("rerun_layer_names").to_pylist(), strict=True))
