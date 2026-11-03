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

"""Train from a Rerun catalog without porting: a :class:`~rosetta.data.FrameReader` over ``.rrd`` recordings.

Each catalog segment with at least one sample start is an episode, so a
recording shorter than the windows after warmup isn't one. Frames come from :class:`~.dataset.ContractFrames`,
so decoding is Rosetta's and windows and the sample set are Rerun's
dataloader's. :meth:`RerunReader.read` reads samples by index, a batch per
catalog query (Rerun's map-style dataset), so the trainer's sampler picks the
order. :meth:`RerunReader.stream` streams them in the order of ``shuffle``
instead (Rerun's iterable dataset), which with ``"block"`` fetches far
faster. Sample ``i`` is the ``i``-th sample start in episode, then tick,
order, so ``index`` counts only the starts Rerun keeps: where it trims, a
ported dataset's ``index`` runs ahead.

Rerun's dataloader trims instead of padding: a tick whose window would leave
its episode is never a sample start. So the last ``max(offset)`` ticks of
each episode, and with an observation history the first ``-min(offset)``, are
not sampled, and ``<key>_is_pad`` is always False (``info().pads`` is
False). A ported LeRobot dataset samples them, padded.

Each episode's task is the operator prompt the recorder stored in the bag's
custom_data, which the ``.rrd`` keeps in its ``rosbag2`` property. A
contract with a ``tasks`` section is refused: per-frame task topics aren't
read from the catalog yet. Stats come from the contract's stats layer (see
:func:`~.stats.write_stats_layer`).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import yaml
from rerun.catalog import CatalogClient

from rosetta.contract.schema import load_contract
from rosetta.contract.specs import iter_policy_specs
from rosetta.data import ReaderInfo
from rosetta.frames.layout import FrameLayout

from .dataset import ContractFrames
from .layout import observation_starts
from .query import POSITION_FIELD
from .stats import load_stats

if TYPE_CHECKING:
    from rerun.experimental.dataloader import ShuffleStrategy

__all__ = ["RerunReader"]

PROMPT_KEY = "lerobot.operator_prompt"
_METADATA_COLUMN = "property:rosbag2:TextDocument:text"


class RerunReader:
    """Segments of a Rerun catalog dataset served as contract frames, one segment per episode.

    ``contract`` is the contract's path (its stats layer is named after the
    file's text). For :meth:`stream`, ``num_workers`` must match the
    DataLoader's: Rerun builds its manifest for that many workers.
    :meth:`read` works in any number of workers. ``manifest`` caches the
    manifest at a path, as in :class:`~.dataset.ContractFrames`. ``segments``
    limits the reader to those segment ids, a list or a comma-separated string.

    ``shuffle`` sets :meth:`stream`'s order and is ``"sample"`` (Rerun's ``SampleShuffle``, the default: every
    sample at a random position, one catalog query each), ``"block"``
    (``BlockShuffle(buffer_size=shuffle_buffer)``: contiguous fetch blocks
    mixed through a buffer, far less work for the server, batches correlated
    by about 128 / ``shuffle_buffer``) or ``"none"``. A
    ``ShuffleStrategy`` instance is taken as is. Strings and string numbers are
    accepted so the options can come from a command line.
    """

    def __init__(
        self,
        *,
        catalog_url: str,
        dataset: str,
        contract: str | Path,
        token: str | None = None,
        segments: list[str] | str | None = None,
        num_workers: int = 0,
        manifest: str | Path | None = None,
        shuffle: str | ShuffleStrategy = "sample",
        shuffle_buffer: int | str = 256,
    ) -> None:
        self._catalog_url = catalog_url
        self._dataset_name = dataset
        self._contract_path = Path(contract)
        self._contract = load_contract(contract)
        if self._contract.tasks:
            raise ValueError(
                "The Rerun reader doesn't read per-frame task topics yet; this contract has a tasks section. "
                "Port the recordings with rosetta_port instead."
            )
        self._token = token
        if isinstance(segments, str):  # a command line's "seg_a,seg_b"
            segments = [s for s in segments.split(",") if s]
        self._segments = segments
        self._num_workers = int(num_workers)
        self._manifest = manifest
        self._shuffle_strategy = _shuffle_strategy(shuffle, int(shuffle_buffer))
        self._layout = FrameLayout(list(iter_policy_specs(self._contract)))
        self._frames: ContractFrames | None = None
        self._windows: dict[str, tuple[int, int]] | None = None
        self._episodes: dict[str, _Episode] = {}

    # -------------------- FrameReader --------------------

    def features(self) -> dict[str, dict[str, Any]]:
        return self._layout.features()

    def prepare(self, windows: Mapping[str, Sequence[int]]) -> None:
        """Build the manifest for ``windows`` and read each segment's range and prompt."""
        self._windows = _contiguous(windows)
        entry = self._entry()
        starts = observation_starts(entry, self._contract, self._segments)
        self._kept = list(starts)
        self._frames = ContractFrames(
            self._catalog_url,
            self._dataset_name,
            self._contract,
            token=self._token,
            segments=self._kept,
            windows=self._windows,
            shuffle_strategy=self._shuffle_strategy,
            num_workers=self._num_workers,
            manifest=self._manifest,
            positions=True,
        )
        # Sample i is the i-th manifest row in episode, then tick, order, as a
        # ported dataset numbers its frames. A segment too short for the windows
        # has no rows, and isn't an episode.
        positions = self._frames.positions()
        sampled = {segment for segment, _ in positions}
        self._episodes = _read_episodes(
            entry, self._frames.index_name, self._contract.fps, {s: v for s, v in starts.items() if s in sampled}
        )
        self._rows = sorted(
            range(len(positions)), key=lambda r: (self._episodes[positions[r][0]].index, positions[r][1])
        )
        self._index_of = {positions[row]: i for i, row in enumerate(self._rows)}
        lengths = [0] * len(self._episodes)
        for segment, _ in positions:
            lengths[self._episodes[segment].index] += 1
        self._lengths = tuple(lengths)
        self._task_index = {task: i for i, task in enumerate(self.info().tasks)}

    def info(self) -> ReaderInfo:
        self._prepared()
        episodes = sorted(self._episodes.values(), key=lambda e: e.index)
        return ReaderInfo(
            fps=self._contract.fps,
            robot_type=self._contract.robot_type,
            episode_lengths=self._lengths,
            episode_tasks=tuple(e.task for e in episodes),
            tasks=tuple(dict.fromkeys(e.task for e in episodes)),
            pads=False,
        )

    def stats(self) -> dict[str, dict[str, np.ndarray]]:
        self._prepared()
        return load_stats(self._entry(), self._contract_path, segments=self._kept)

    def read(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        """The samples at ``indices``, fetched in one catalog query."""
        frames = self._prepared()
        return [self._numbered(frame) for frame in frames.read([self._rows[i] for i in indices])]

    def stream(self, *, seed: int, epoch: int, worker: tuple[int, int]) -> Iterator[dict[str, Any]]:
        """Every sample once, in the order of this reader's ``shuffle``, the share of DataLoader worker ``worker``."""
        frames = self._prepared()
        workers = max(1, self._num_workers)
        if worker[1] != workers:
            raise ValueError(
                f"The manifest was built for {workers} DataLoader worker(s), but this is worker {worker[0]} "
                f"of {worker[1]}. Pass num_workers={worker[1]} to the reader."
            )
        frames.reseed(seed)
        frames.set_epoch(epoch)
        for frame in frames:
            yield self._numbered(frame)

    # -------------------- internals --------------------

    def _entry(self):
        return CatalogClient(self._catalog_url, token=self._token).get_dataset(self._dataset_name)

    def _prepared(self) -> ContractFrames:
        if self._frames is None:
            raise RuntimeError("Call prepare(windows) before info(), read() or stream()")
        return self._frames

    def _numbered(self, frame: dict[str, Any]) -> dict[str, Any]:
        """``frame`` with its task, padding flags and numbers, from its position."""
        segment, index_ns = frame.pop(POSITION_FIELD)
        episode = self._episodes[segment]
        frame_index = (index_ns - episode.start_ns) // round(1e9 / self._contract.fps)
        for key, (start, end) in self._windows.items():
            frame[f"{key}_is_pad"] = np.zeros(end - start + 1, dtype=bool)
        frame["task"] = episode.task
        frame["task_index"] = self._task_index[episode.task]
        frame["episode_index"] = episode.index
        frame["frame_index"] = int(frame_index)
        frame["index"] = self._index_of[(segment, index_ns)]
        frame["timestamp"] = frame_index / self._contract.fps
        return frame


class _Episode:
    __slots__ = ("index", "start_ns", "task")

    def __init__(self, index: int, start_ns: int, task: str) -> None:
        self.index, self.start_ns, self.task = index, start_ns, task


def _shuffle_strategy(shuffle: str | ShuffleStrategy, buffer: int) -> ShuffleStrategy:
    from rerun.experimental.dataloader import BlockShuffle, NoShuffle, SampleShuffle

    if not isinstance(shuffle, str):
        return shuffle
    strategies = {"sample": SampleShuffle, "block": lambda: BlockShuffle(buffer_size=buffer), "none": NoShuffle}
    if shuffle not in strategies:
        raise ValueError(f"shuffle must be one of {sorted(strategies)}, got {shuffle!r}")
    return strategies[shuffle]()


def _contiguous(windows: Mapping[str, Sequence[int]]) -> dict[str, tuple[int, int]]:
    """Windows as inclusive ``(start, end)`` ranges. Rerun's dataloader takes only contiguous offsets."""
    out = {}
    for key, given in windows.items():
        offsets = list(given)
        if not offsets or offsets != list(range(offsets[0], offsets[-1] + 1)):
            raise ValueError(f"Window for '{key}' must be consecutive tick offsets, got {offsets}")
        out[key] = (offsets[0], offsets[-1])
    return out


def _read_episodes(entry, index: str, fps: int, observed: dict[str, int | None]) -> dict[str, _Episode]:
    """Each segment's episode number, first frame's index value and prompt, by segment id.

    ``observed`` holds the segments to number, with the index value at which
    every observation stream has data. The first frame is the first tick at or
    after it, as the porter's warmup ends, so frames are numbered as the
    porter numbers them.
    """
    table = entry.segment_table().to_arrow_table()
    ids = table.column("rerun_segment_id").to_pylist()
    starts = table.column(f"{index}:start").cast(pa.int64()).to_pylist()
    texts = table.column(_METADATA_COLUMN).to_pylist() if _METADATA_COLUMN in table.column_names else [None] * len(ids)
    step = round(1e9 / fps)
    episodes: dict[str, _Episode] = {}
    for segment, start, text in sorted(zip(ids, starts, texts, strict=True)):
        if segment not in observed or start is None:
            continue  # no data on the index timeline: no samples either
        warm = observed[segment]
        first = start if warm is None else start - (start - warm) // step * step
        episodes[segment] = _Episode(len(episodes), first, _prompt(text))
    return episodes


def _prompt(text: Any) -> str:
    """The operator prompt in an rrd's ``rosbag2`` metadata property, or ""."""
    if isinstance(text, list):
        text = text[0] if text else None
    if not text:
        return ""
    custom = (yaml.safe_load(text) or {}).get("custom_data") or {}
    return str(custom.get(PROMPT_KEY, "")) if isinstance(custom, dict) else ""
