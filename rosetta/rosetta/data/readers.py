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

"""The seam a dataset reader implements: recordings served as frames, at training time.

A dataset writer applies the contract once, at port time, and writes a
dataset a framework's trainer reads. A :class:`FrameReader` applies it at
training time instead: it reads recordings and serves contract frames
straight to the trainer, so changing the contract needs no new port. A
framework's view (such as ``lerobot_dataset_rosetta.view.LeRobotView``)
turns the reader's samples into what that framework's trainer takes.

A reader serves samples by index, as a ported dataset does: sample ``i`` of
``num_frames``, episode by episode. Which samples a training step reads, in
which order and in which worker, is the trainer's sampler's choice. A reader
whose recordings read faster in an order of its own can also stream them
(:class:`StreamingFrameReader`).

Each sample is a dict:

- ``{contract_key: value}`` for every policy stream key. A key in the
  prepared windows holds a ``[len(offsets), ...]`` stack, with
  ``<key>_is_pad`` (a bool array) beside it.
- ``task`` (str), ``task_index``, ``episode_index``, ``frame_index``,
  ``index`` (ints) and ``timestamp`` (float seconds). ``frame_index`` counts
  ticks from the episode's first frame after warmup, as the port numbers a
  ported dataset's frames. ``timestamp`` is ``frame_index / fps``. ``index``
  is the sample's own index.

Values are in the contract's units: numeric keys as numpy arrays, images as
``(height, width, channels)`` uint8.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = ["FrameReader", "ReaderInfo", "StreamingFrameReader", "Windows"]

Windows = Mapping[str, Sequence[int]]
"""Contract key -> tick offsets from the sampled tick, e.g. ``{"action": range(50)}``."""


@dataclass(frozen=True)
class ReaderInfo:
    """What a trainer needs to know about a reader's data before it reads any."""

    fps: int
    """The contract's frame rate."""
    robot_type: str
    """The contract's ``robot_type``."""
    episode_lengths: tuple[int, ...]
    """Sample starts in each episode, in episode order. Episode ``e``'s samples
    are the indices from ``sum(episode_lengths[:e])`` on."""
    episode_tasks: tuple[str, ...]
    """Each episode's first task, in episode order."""
    tasks: tuple[str, ...]
    """Distinct task strings; a sample's ``task_index`` indexes this tuple."""
    pads: bool
    """True when windows pad at episode edges. False when the reader trims instead:
    a tick whose window would leave its episode is never a sample start, and
    ``<key>_is_pad`` is always False. ``episode_lengths`` then count the sample
    starts left for the prepared windows."""

    @property
    def num_episodes(self) -> int:
        return len(self.episode_lengths)

    @property
    def num_frames(self) -> int:
        """Sample starts across every episode."""
        return sum(self.episode_lengths)


@runtime_checkable
class FrameReader(Protocol):
    """Recordings served as contract frames, for training without porting.

    Registered under ``rosetta.frame_readers`` and constructed with keyword
    options (see :func:`rosetta.data.registry.load_frame_reader`). A trainer
    calls :meth:`prepare` once, in its main process, before anything else;
    :meth:`read` may then run in DataLoader worker processes.
    """

    def features(self) -> dict[str, dict[str, Any]]:
        """:meth:`~rosetta.frames.layout.FrameLayout.features` of the policy streams, unwindowed.

        Available before :meth:`prepare`: a trainer reads it to choose the windows.
        """
        ...

    def prepare(self, windows: Windows) -> None:
        """Do the one-time work for these windows: index the recordings, count samples, gather stats.

        Called once before :meth:`info`, :meth:`stats` and :meth:`read`, with
        the windows every sample holds. A reader that trims at episode edges
        counts its samples here.
        """
        ...

    def info(self) -> ReaderInfo:
        """The data's shape and size, for the prepared windows."""
        ...

    def stats(self) -> dict[str, dict[str, np.ndarray]]:
        """Exact stats per key, as :func:`rosetta.data.stats.frame_stats` gives them."""
        ...

    def read(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        """The samples at ``indices``, in that order. A trainer reads a batch in one call."""
        ...


@runtime_checkable
class StreamingFrameReader(FrameReader, Protocol):
    """A reader that also streams its samples in an order of its own, faster than reading them by index."""

    def stream(self, *, seed: int, epoch: int, worker: tuple[int, int]) -> Iterator[dict[str, Any]]:
        """Every sample once, in an order fixed by ``seed`` and ``epoch``.

        ``worker`` is ``(index, count)``: this call yields only that worker's
        share, and the shares of one epoch partition the samples.
        """
        ...
