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

"""Contract-shaped torch datasets over a Rerun catalog.

``ContractFrames`` is the lazy sibling of ``iter_bag_frames``: same contract,
same ``FrameLayout.assemble`` output, but frames materialize from catalog
queries instead of a sequential bag read.

Sampling, windowing, shuffling and worker/rank sharding are the dataloader's.
A ``Manifest`` built with the observation fields as ``required_fields`` drops
every tick before all observation streams have a sample, which is the porter's
warmup gate. Where each stream sits in a recording comes from the contract's
robot interface's :class:`~.layout.RecordingLayout`.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from multiprocessing.sharedctypes import RawValue
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch.utils.data
from rerun.catalog import CatalogClient
from rerun.experimental.dataloader import (
    BlockShuffle,
    DataSource,
    Manifest,
    RerunIterableDataset,
    SampleShuffle,
    ShuffleStrategy,
)

from rosetta.frames.layout import FrameLayout

from .layout import load_layout
from .query import CompiledQuery, compile_contract, field_key

if TYPE_CHECKING:
    from rerun.catalog import DatasetEntry

    from rosetta.contract.schema import Contract

    from .layout import RecordingLayout


def contract_dataset(
    catalog_url: str, dataset_name: str, contract: Contract | str | Path, **kwargs: Any
) -> ContractFrames:
    """Open ``dataset_name`` on ``catalog_url`` as contract-shaped frames. Same arguments as :class:`ContractFrames`."""
    return ContractFrames(catalog_url, dataset_name, contract, **kwargs)


class ContractFrames(torch.utils.data.IterableDataset):
    """Iterable dataset yielding porter-identical frame dicts.

    Each sample is ``{contract_key: np.ndarray | str}``, what ``iter_bag_frames``
    yields per tick, minus the episode markers and ``task`` string. A key in
    ``windows`` holds a ``[window_len, ...]`` stack, one frame value per offset.

    ``contract`` is a loaded ``Contract`` or a path to its YAML. ``token``
    authenticates against a hub; a local ``rerun server`` needs none. DataLoader
    workers reconnect with the same URL, name and token.

    ``shuffle_strategy`` is Rerun's: ``SampleShuffle()`` (the default) puts every
    sample at an independent random position. ``BlockShuffle(buffer_size=...)``
    fetches contiguous blocks and mixes decoded samples through a buffer before
    emitting them. It fetches faster, but every DataLoader worker holds
    ``buffer_size`` decoded samples, and batches stay correlated by about
    ``fetch_block_size / buffer_size``. ``BlockShuffle`` without a buffer emits
    runs of consecutive ticks and is rejected. ``NoShuffle()`` gives natural
    order, segment by segment along the timeline.

    Each epoch replays the manifest in a new order: call :meth:`set_epoch` before
    iterating a ``DataLoader`` for that epoch. Workers read the epoch when an
    epoch starts, so ``persistent_workers=True`` works. The manifest is frozen
    for ``num_workers``, so iterate only through a ``DataLoader`` with that many
    workers (``0``: iterate it directly), and with
    ``multiprocessing_context="spawn"`` when there are any. ``layout`` defaults
    to the one registered for the contract's robot interface.

    Building the manifest scans the whole dataset. With ``manifest`` set to a
    path, the first run writes the manifest there and later runs load it
    instead. A saved manifest that doesn't match this dataset, contract,
    ``windows`` or ``num_workers`` raises. It records which samples existed
    when it was built, so delete it after registering new recordings or
    changing ``segments``.
    """

    def __init__(
        self,
        catalog_url: str,
        dataset_name: str,
        contract: Contract | str | Path,
        *,
        token: str | None = None,
        segments: list[str] | None = None,
        windows: dict[str, tuple[int, int]] | None = None,
        shuffle_strategy: ShuffleStrategy | None = None,
        seed: int = 0,
        num_workers: int = 0,
        layout: RecordingLayout | None = None,
        manifest: str | Path | None = None,
    ) -> None:
        super().__init__()
        if isinstance(contract, (str, Path)):
            from rosetta.contract.schema import load_contract

            contract = load_contract(contract)
        self._shuffle_strategy = _training_safe(shuffle_strategy)
        layout = layout or load_layout(contract.robot_interface)
        dataset_entry = CatalogClient(catalog_url, token=token).get_dataset(dataset_name)
        self._connection = (catalog_url, dataset_name, token, segments)
        self._query = compile_contract(contract, dataset_entry.schema().column_names(), layout, windows=windows)
        self._layout = FrameLayout(self._query.specs)
        self._source = DataSource(dataset_entry, segments)
        num_workers_per_rank = max(1, num_workers)
        # A parquet-backed manifest pickles as its path, so DataLoader workers
        # don't each receive every row (which also starts them one at a time).
        self._manifest_dir = None
        if manifest is None and num_workers:
            self._manifest_dir = tempfile.TemporaryDirectory(prefix="rerun_rosetta_")
            manifest = Path(self._manifest_dir.name) / "manifest.parquet"
        if manifest is not None and Path(manifest).exists():
            self._manifest = Manifest.from_parquet(manifest)
            _check_manifest(self._manifest, dataset_entry, self._query, num_workers_per_rank, manifest)
        else:
            self._manifest = Manifest.generate(
                self._source,
                self._query.index,
                self._query.fields,
                timeline_sampling=self._query.sampling,
                num_workers_per_rank=num_workers_per_rank,
                required_fields=list(self._query.required),
            )
            if manifest is not None:
                self._manifest.write_parquet(manifest)
                self._manifest = Manifest.from_parquet(manifest)
        if not self._manifest.num_rows:
            raise RuntimeError("No sample has data for every observation stream; nothing to train on.")
        self._seed = seed
        # Shared memory, inherited by DataLoader workers when they spawn, so
        # persistent workers see every set_epoch.
        self._epoch = RawValue("q", 0)

        # A windowed key stacks one assembled value per offset. Assembling it
        # through a layout of just its specs keeps zero-fill and concatenation
        # FrameLayout's, without assembling the whole frame per offset.
        windowed = {
            key: [(field_key(i), spec) for i, spec in enumerate(self._query.specs) if spec.key == key]
            for key in self._query.windows
        }
        self._windowed = {
            key: (FrameLayout([spec for _, spec in group]), [fk for fk, _ in group]) for key, group in windowed.items()
        }

    @property
    def features(self) -> dict[str, Any]:
        """LeRobot feature schema for the assembled frame keys (unwindowed shapes)."""
        return self._layout.lerobot_features()

    def set_epoch(self, epoch: int) -> None:
        """Replay the manifest in epoch ``epoch``'s order from the next iteration on."""
        self._epoch.value = epoch

    def __len__(self) -> int:
        return self._manifest.num_rows

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self._source is None:  # a DataLoader worker: the live entry doesn't pickle
            catalog_url, dataset_name, token, segments = self._connection
            self._source = DataSource(CatalogClient(catalog_url, token=token).get_dataset(dataset_name), segments)
        manifest = self._manifest.shuffle(self._shuffle_strategy, seed=self._seed + self._epoch.value)
        for sample in RerunIterableDataset.from_manifest(manifest, self._source, self._query.fields):
            yield self._assemble(sample)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_source"] = None
        state["_manifest_dir"] = None  # only the parent removes the temporary manifest
        return state

    def _assemble(self, sample: dict[str, Any]) -> dict[str, Any]:
        frame = self._layout.assemble(
            [None if spec.key in self._windowed else sample.get(field_key(i)) for i, spec in enumerate(self._query.specs)]
        )
        for key, (layout, keys) in self._windowed.items():
            start, end = self._query.windows[key]
            per_offset = []
            for offset in range(end - start + 1):
                values = [None if sample.get(k) is None else sample[k][offset] for k in keys]
                per_offset.append(layout.assemble(values)[key])
            frame[key] = np.stack(per_offset)
        return frame


def _check_manifest(
    manifest: Manifest, entry: DatasetEntry, query: CompiledQuery, num_workers_per_rank: int, path: str | Path
) -> None:
    """Raise unless ``manifest`` is the one ``Manifest.generate`` would build for ``query``."""
    meta = manifest.metadata
    expected = {
        "dataset": (meta.dataset_id, str(entry.id)),
        "timeline": (meta.index_name, query.index),
        # Rerun's own rate-to-step conversion (FixedRateSampling): the contract's fps.
        "fps": (meta.ns_per_sample, round(1e9 / query.sampling.rate_hz)),
        "streams and windows": (meta.recipe, {key: field.to_recipe() for key, field in query.fields.items()}),
        "required streams": (meta.required_fields, sorted(query.required)),
        "num_workers": (meta.num_workers_per_rank, num_workers_per_rank),
    }
    differ = [name for name, (saved, now) in expected.items() if saved != now]
    if differ:
        raise ValueError(
            f"Manifest {path} was built for a different {', '.join(differ)}. "
            f"Delete it to rebuild, or pass another path."
        )


def _training_safe(strategy: ShuffleStrategy | None) -> ShuffleStrategy:
    """``strategy``, defaulting to ``SampleShuffle``. Rejects ``BlockShuffle`` without an emission buffer."""
    if strategy is None:
        return SampleShuffle()
    if isinstance(strategy, BlockShuffle) and strategy.buffer_size is None:
        raise ValueError(
            "BlockShuffle without buffer_size emits runs of consecutive ticks, so every batch comes from one "
            "stretch of one episode. Pass BlockShuffle(buffer_size=N), with N a few times "
            "fetch_block_size x batch_size (each DataLoader worker holds N decoded samples), or use SampleShuffle()."
        )
    return strategy
