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
summing, so every statistic is exact over every frame (see
:mod:`rosetta.data.stats`). The recordings themselves are never touched, and new episodes
only need their own layer files.

Frames are each episode's porter frames: unwindowed, every tick after warmup.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import rerun as rr
from rerun.experimental.dataloader import NoShuffle

from rosetta.contract.schema import load_contract
from rosetta.contract.specs import iter_policy_specs
from rosetta.data.stats import LEVELS, histogram_stats, sufficient_statistics, value_stats
from rosetta.frames.layout import FrameLayout

from .dataset import ContractFrames, NoSamplesError
from .layout import load_layout, observation_starts

if TYPE_CHECKING:
    from rerun.catalog import DatasetEntry

    from .layout import RecordingLayout


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
    layout = layout or load_layout(contract.robot_interface)  # once, not per segment
    usable = set(observation_starts(dataset_entry, contract))
    todo = sorted(
        segment
        for segment, layers in _segment_layers(dataset_entry).items()
        if name not in layers and segment in usable
    )

    layer_dir = Path(out_dir) / name
    layer_dir.mkdir(parents=True, exist_ok=True)
    uris = []
    for segment in todo:
        try:
            frames = ContractFrames(
                dataset_entry.catalog.url,
                dataset_entry.name,
                contract,
                token=token,
                segments=[segment],
                shuffle_strategy=NoShuffle(),
                layout=layout,
            )
        except NoSamplesError:
            # Training skips a segment it can't sample, so its stats count 0
            # frames rather than stopping every other segment's layer.
            logging.warning("%s: no tick has data for every observation stream; its stats count 0 frames", segment)
            count, payload = 0, {}
        else:
            count, payload = sufficient_statistics(frames, frame_layout)
        path = layer_dir / f"{segment}.rrd"
        _write_layer(path, segment, name, count, payload, text)
        uris.append(path.resolve().as_uri())
    if uris:
        dataset_entry.register(uris, layer_name=name).wait()
    return todo


def _write_layer(path: Path, segment: str, name: str, count: int, payload: dict[str, np.ndarray], text: str) -> None:
    """Write one segment's stats layer: the sufficient statistics, and the frame count and contract."""
    with rr.RecordingStream("rosetta_stats", recording_id=segment, send_properties=False) as rec:
        rec.save(path)
        if payload:
            rec.send_property(name, rr.AnyValues(**{key: value.ravel() for key, value in payload.items()}))
        rec.send_property(f"{name}_meta", rr.AnyValues(frames=count, contract=text))


def load_stats(
    dataset_entry: DatasetEntry, contract_path: str | Path, *, segments: list[str] | None = None
) -> dict[str, dict[str, np.ndarray]]:
    """Exact stats over ``segments``, per contract key, from the contract's stats layer.

    ``segments`` defaults to every segment with data for each observation
    stream, the ones a reader trains on (see :func:`.layout.observation_starts`).

    Numeric keys keep the contract's units, shape ``(width,)``. Image keys are
    per channel in pixel values 0..255, shape ``(channels,)``. ``count`` is the
    number of frames.

    Raises:
        ValueError: If a segment has no stats layer for this contract.

    """
    text = Path(contract_path).read_text()
    name = stats_layer_name(text)
    contract = load_contract(contract_path)
    frame_layout = FrameLayout(list(iter_policy_specs(contract)))
    wanted = set(segments if segments is not None else observation_starts(dataset_entry, contract))
    layers = _segment_layers(dataset_entry)
    missing = sorted(segment for segment in wanted if name not in layers.get(segment, []))
    if missing:
        raise ValueError(f"Segments without the stats layer {name}: {missing}. Run write_stats_layer first.")

    # The dataframe reader keeps property field names as logged. The segment
    # table rewrites dots in them, and contract keys have dots.
    table = (
        dataset_entry.filter_contents([f"/__properties/{name}", f"/__properties/{name}_meta"])
        .reader(index=None)
        .to_arrow_table()
    )
    keep = [segment in wanted for segment in table.column("rerun_segment_id").to_pylist()]

    def column(prop: str, field: str) -> list:
        values = table.column(f"property:{prop}:{field}").to_pylist()
        return [value for value, kept in zip(values, keep, strict=True) if kept]

    frames = [int(row[0]) for row in column(f"{name}_meta", "frames")]
    rows = {key: column(name, key) for key in frame_layout.keys if frame_layout[key].category != "string"}
    return _stats_from_layers(frames, rows, frame_layout)


def _stats_from_layers(
    frames: list[int], rows: dict[str, list[Any]], frame_layout: FrameLayout
) -> dict[str, dict[str, np.ndarray]]:
    """Merge per-segment stats layers into stats over every frame.

    A segment with no samplable tick has a layer with 0 frames and no values
    (its rows are None), and adds nothing.
    """
    stats: dict[str, dict[str, np.ndarray]] = {}
    for key, key_rows in rows.items():
        present = [np.asarray(row) for row in key_rows if row is not None]
        if frame_layout[key].category == "image":
            stats[key] = histogram_stats(sum(row.reshape(-1, LEVELS) for row in present), sum(frames))
        else:
            stats[key] = value_stats(np.concatenate([row.reshape(-1, *frame_layout[key].shape) for row in present]))
    return stats


def _segment_layers(dataset_entry: DatasetEntry) -> dict[str, list[str]]:
    table = dataset_entry.segment_table().to_arrow_table()
    segments = table.column("rerun_segment_id").to_pylist()
    return dict(zip(segments, table.column("rerun_layer_names").to_pylist(), strict=True))
