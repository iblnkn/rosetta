#!/usr/bin/env python
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

"""Any Rosetta frame reader as a dataset LeRobot's trainer takes.

:class:`LeRobotView` wraps a :class:`~rosetta.data.FrameReader` (a Rerun
catalog, ...) and serves it the way a ported ``LeRobotDataset`` serves
its frames, by index, to LeRobot's own sampler. :class:`LeRobotStreamView`
streams a reader that reads faster in an order of its own instead, the way
``StreamingLeRobotDataset`` does. Either way, each sample holds the windows
the policy's config asks for
(``observation_delta_indices``, ``action_delta_indices``,
``reward_delta_indices``), ``<key>_is_pad`` beside each, ``task`` and the
index columns, and ``meta`` describes the data with features and stats.
So any LeRobot policy trains on recordings without porting them.

Images are served CHW uint8, which ``lerobot-train`` scales to [0, 1].
Floating values are float32. Stats are converted from Rosetta's units (image
stats per channel in 0..255) to LeRobot's, and with ``use_imagenet_stats``
camera stats are replaced by ImageNet's, as ``lerobot-train`` does for its
own datasets.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from lerobot.datasets.dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import create_empty_dataset_info
from lerobot.utils.constants import ACTION, DEFAULT_FEATURES, IMAGENET_STATS, OBS_PREFIX, REWARD

from rosetta.data import FrameReader, ReaderInfo, StreamingFrameReader, Windows

__all__ = ["LeRobotStreamView", "LeRobotView", "policy_windows", "split_episodes"]

_INDEX_KEYS = ("task_index", "episode_index", "frame_index", "index")


def policy_windows(policy_cfg: Any, features: dict[str, dict[str, Any]]) -> dict[str, list[int]]:
    """The windows a LeRobot policy config asks for, per frame key.

    The same rule as LeRobot's ``resolve_delta_timestamps``: ``action`` takes
    ``action_delta_indices``, ``next.reward`` takes ``reward_delta_indices``,
    and every ``observation.*`` key takes ``observation_delta_indices``.
    String keys have no window.
    """
    windows: dict[str, list[int]] = {}
    for key, feature in features.items():
        if feature["dtype"] == "string":
            continue
        if key == ACTION:
            indices = getattr(policy_cfg, "action_delta_indices", None)
        elif key == REWARD:
            indices = getattr(policy_cfg, "reward_delta_indices", None)
        elif key.startswith(OBS_PREFIX):
            indices = getattr(policy_cfg, "observation_delta_indices", None)
        else:
            indices = None
        if indices is not None:
            windows[key] = [int(i) for i in indices]
    return windows


class _Prepared:
    """A reader prepared for one policy's windows, with the LeRobot metadata that describes it."""

    def __init__(self, reader: FrameReader, policy_cfg: Any, *, repo_id: str, use_imagenet_stats: bool) -> None:
        self.reader = reader
        features = reader.features()
        self.windows: Windows = policy_windows(policy_cfg, features)
        reader.prepare(self.windows)
        self.info = reader.info()
        self.camera_keys = [k for k, f in features.items() if f["dtype"] in ("video", "image")]
        self.meta = _metadata(self.info, features, reader.stats(), repo_id, use_imagenet_stats)
        # Episode e's samples are indices offsets[e] .. offsets[e + 1] - 1.
        self.offsets = np.concatenate([[0], np.cumsum(self.info.episode_lengths)]).astype(np.int64)


class _ViewProperties:
    _prepared: _Prepared

    @property
    def meta(self) -> LeRobotDatasetMetadata:
        return self._prepared.meta

    @property
    def windows(self) -> Windows:
        """The windows served, from the policy's delta indices."""
        return self._prepared.windows

    @property
    def info(self) -> ReaderInfo:
        """The reader's :class:`~rosetta.data.ReaderInfo`, every episode included."""
        return self._prepared.info

    @property
    def num_frames(self) -> int:
        return self._prepared.info.num_frames

    @property
    def num_episodes(self) -> int:
        return self._prepared.info.num_episodes

    @property
    def fps(self) -> int:
        return self._prepared.info.fps


class LeRobotView(_ViewProperties, torch.utils.data.Dataset):
    """``reader``'s recordings as a LeRobot training dataset for the policy ``policy_cfg`` describes.

    Map-style, as a ported ``LeRobotDataset`` is: ``lerobot-train`` draws
    samples with its own sampler (``EpisodeAwareSampler``), which shuffles,
    shards across ranks and resumes exactly, and reads a batch with one
    reader call (``__getitems__``). ``meta.episodes`` holds each episode's
    index range, as the sampler reads it.

    ``repo_id`` only names the data in LeRobot's logs and metadata.
    ``image_transforms`` (LeRobot's) runs on each camera image, as LeRobot's
    datasets run it. :meth:`select` gives a view over some of the episodes,
    as LeRobot's ``episodes=`` does, for a train and eval split.
    """

    def __init__(
        self,
        reader: FrameReader,
        policy_cfg: Any,
        *,
        repo_id: str = "rosetta/view",
        image_transforms: Callable[[torch.Tensor], torch.Tensor] | None = None,
        use_imagenet_stats: bool = True,
    ) -> None:
        super().__init__()
        self._prepared = _Prepared(reader, policy_cfg, repo_id=repo_id, use_imagenet_stats=use_imagenet_stats)
        self._image_transforms = image_transforms
        self.episodes: list[int] | None = None
        self.absolute_to_relative_idx: dict[int, int] | None = None
        self._absolute: np.ndarray | None = None

    def select(
        self,
        episodes: Sequence[int],
        *,
        image_transforms: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> LeRobotView:
        """A view over ``episodes`` only, sharing this view's prepared reader.

        Its positions run over the selected episodes' samples, and
        ``absolute_to_relative_idx`` maps each sample's index to its position,
        as LeRobot's episode-filtered datasets do.
        """
        view = LeRobotView.__new__(LeRobotView)
        torch.utils.data.Dataset.__init__(view)
        view._prepared = self._prepared
        view._image_transforms = image_transforms
        view.episodes = sorted(int(e) for e in episodes)
        offsets = self._prepared.offsets
        view._absolute = np.concatenate(
            [np.arange(offsets[e], offsets[e + 1]) for e in view.episodes] or [np.zeros(0, dtype=np.int64)]
        )
        view.absolute_to_relative_idx = {int(a): r for r, a in enumerate(view._absolute)}
        return view

    @property
    def num_frames(self) -> int:
        """Samples in this view: the selected episodes' only, as ``LeRobotDataset(episodes=...)`` counts them."""
        return self._prepared.info.num_frames if self._absolute is None else len(self._absolute)

    @property
    def num_episodes(self) -> int:
        """Episodes in this view: the selected ones only, as ``LeRobotDataset(episodes=...)`` counts them."""
        return self._prepared.info.num_episodes if self.episodes is None else len(self.episodes)

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, position: int) -> dict[str, Any]:
        return self.__getitems__([position])[0]

    def __getitems__(self, positions: Sequence[int]) -> list[dict[str, Any]]:
        indices = [int(p) for p in positions] if self._absolute is None else [int(self._absolute[p]) for p in positions]
        samples = self._prepared.reader.read(indices)
        return [_to_torch(sample, self._prepared.camera_keys, self._image_transforms) for sample in samples]


class LeRobotStreamView(_ViewProperties, torch.utils.data.IterableDataset):
    """``reader``'s recordings streamed in the reader's own order, for a reader that reads faster that way.

    For ``lerobot-train --dataset.streaming=true``: the reader shuffles and
    gives each DataLoader worker its share, and each pass is a new epoch with
    a new order. Streaming has no eval split and resumes at an epoch, not a
    sample, as LeRobot's own streaming does. ``reader`` must be a
    :class:`~rosetta.data.StreamingFrameReader`.
    """

    def __init__(
        self,
        reader: StreamingFrameReader,
        policy_cfg: Any,
        *,
        repo_id: str = "rosetta/view",
        image_transforms: Callable[[torch.Tensor], torch.Tensor] | None = None,
        use_imagenet_stats: bool = True,
        seed: int = 0,
    ) -> None:
        if not isinstance(reader, StreamingFrameReader):
            raise TypeError(f"{type(reader).__name__} can't stream; read it with LeRobotView instead")
        super().__init__()
        self._prepared = _Prepared(reader, policy_cfg, repo_id=repo_id, use_imagenet_stats=use_imagenet_stats)
        self._image_transforms = image_transforms
        self._seed = seed
        self._epoch = 0

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = torch.utils.data.get_worker_info()
        if worker is None:
            share, epoch = (0, 1), self._epoch
            self._epoch += 1
        else:
            # Restarted workers get a new base seed (seed - id) each pass, the
            # same in every worker of that pass. Persistent workers, which
            # lerobot-train uses by default, keep theirs, but each counts its
            # own passes, and all of them count the same. Together they make
            # every pass's epoch new and shared by its workers.
            share = (worker.id, worker.num_workers)
            epoch = (worker.seed - worker.id + self._epoch) % 2**31
            self._epoch += 1
        for sample in self._prepared.reader.stream(seed=self._seed, epoch=epoch, worker=share):
            yield _to_torch(sample, self._prepared.camera_keys, self._image_transforms)


def split_episodes(
    info: ReaderInfo, eval_split: float, *, episodes: Sequence[int] | None = None
) -> tuple[list[int], list[int]]:
    """Train and eval episodes, split as ``lerobot-train --dataset.eval_split`` splits a dataset.

    For each task (an episode's first), the last ``ceil(n * eval_split)`` of
    its ``n`` episodes are held out for eval. With ``episodes`` (LeRobot's
    ``--dataset.episodes``) only those are split.

    Raises:
        ValueError: If no episode is left to train on.

    """
    base = range(info.num_episodes) if episodes is None else [int(e) for e in episodes]
    by_task: dict[str, list[int]] = {}
    for episode in base:
        by_task.setdefault(info.episode_tasks[episode], []).append(episode)
    train: list[int] = []
    held_out: list[int] = []
    for task_episodes in by_task.values():
        n_eval = math.ceil(len(task_episodes) * eval_split)
        train.extend(task_episodes[: len(task_episodes) - n_eval])
        held_out.extend(task_episodes[len(task_episodes) - n_eval :])
    if not train:
        raise ValueError(f"eval_split={eval_split} leaves 0 training episodes from {len(base)} total.")
    return sorted(train), sorted(held_out)


def train_eval_views(
    view: LeRobotView,
    *,
    episodes: Sequence[int] | None,
    eval_split: float,
    image_transforms: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[LeRobotView, LeRobotView | None]:
    """The train and eval views ``lerobot-train`` asks for, as it makes them from a ported dataset.

    ``episodes`` (``--dataset.episodes``) restricts both to those episodes,
    and ``eval_split`` holds out the last of each task's among them. The eval
    view gets no image transforms, as LeRobot's doesn't.
    """
    if eval_split == 0:
        return (view, None) if episodes is None else (view.select(episodes, image_transforms=image_transforms), None)
    train, held_out = split_episodes(view.info, eval_split, episodes=episodes)
    return view.select(train, image_transforms=image_transforms), view.select(held_out)


def _to_torch(
    sample: dict[str, Any],
    camera_keys: Sequence[str],
    image_transforms: Callable[[torch.Tensor], torch.Tensor] | None,
) -> dict[str, Any]:
    item: dict[str, Any] = {}
    for key, value in sample.items():
        if key == "task":
            item[key] = value
        elif key in _INDEX_KEYS:
            item[key] = torch.tensor(value, dtype=torch.int64)
        elif key == "timestamp":
            item[key] = torch.tensor(value, dtype=torch.float32)
        elif key.endswith("_is_pad"):
            item[key] = torch.from_numpy(np.asarray(value, dtype=bool))
        elif isinstance(value, str):
            continue  # string streams aren't policy inputs
        elif key in camera_keys:
            item[key] = _image(np.asarray(value), image_transforms)
        else:
            array = np.asarray(value)
            tensor = torch.from_numpy(np.ascontiguousarray(array))
            item[key] = tensor.float() if np.issubdtype(array.dtype, np.floating) else tensor
    return item


def _image(value: np.ndarray, transforms: Callable[[torch.Tensor], torch.Tensor] | None) -> torch.Tensor:
    """HWC (or THWC) uint8 to CHW (or TCHW), then LeRobot's image transforms."""
    tensor = torch.from_numpy(np.ascontiguousarray(value))
    tensor = tensor.permute(0, 3, 1, 2) if tensor.ndim == 4 else tensor.permute(2, 0, 1)
    return transforms(tensor) if transforms is not None else tensor


def _metadata(
    info: ReaderInfo,
    reader_features: dict[str, dict[str, Any]],
    stats: dict[str, dict[str, np.ndarray]],
    repo_id: str,
    use_imagenet_stats: bool,
) -> LeRobotDatasetMetadata:
    """A LeRobotDatasetMetadata describing the reader's data, with no files behind it."""
    features = {key: dict(f) for key, f in reader_features.items() if f["dtype"] != "string"}
    meta = LeRobotDatasetMetadata.__new__(LeRobotDatasetMetadata)
    meta.repo_id = repo_id
    meta._requested_root = None
    meta.root = Path("/nonexistent/rosetta") / repo_id  # nothing is ever read from it
    meta.revision = None
    meta._pq_writer = None
    meta.latest_episode = None
    meta._metadata_buffer = []
    meta._metadata_buffer_size = 10
    meta._finalized = True  # nothing to flush
    meta.info = create_empty_dataset_info(
        CODEBASE_VERSION, info.fps, {**features, **DEFAULT_FEATURES}, True, info.robot_type
    )
    meta.info.total_episodes = info.num_episodes
    meta.info.total_frames = info.num_frames
    meta.info.total_tasks = len(info.tasks)
    meta.tasks = pd.DataFrame({"task_index": range(len(info.tasks))}, index=pd.Index(list(info.tasks), name="task"))
    # Each episode's index range, the part of LeRobot's episode table its sampler reads.
    ends = np.cumsum(info.episode_lengths).tolist()
    meta.episodes = {"dataset_from_index": [0, *ends[:-1]], "dataset_to_index": ends}
    meta.stats = _lerobot_stats(stats, features)
    if use_imagenet_stats:
        for key in meta.camera_keys:
            if key in meta.depth_keys:
                continue
            for name, value in IMAGENET_STATS.items():
                meta.stats.setdefault(key, {})[name] = torch.tensor(value, dtype=torch.float32)
    return meta


def _lerobot_stats(stats: dict[str, dict[str, np.ndarray]], features: dict[str, dict[str, Any]]) -> dict:
    """Rosetta stats in LeRobot's form: tensors, images per channel (C, 1, 1) in [0, 1]."""
    out: dict[str, dict[str, torch.Tensor]] = {}
    for key, key_stats in stats.items():
        if key not in features:
            continue
        image = features[key]["dtype"] in ("video", "image")
        out[key] = {}
        for name, value in key_stats.items():
            array = np.asarray(value, dtype=np.float64)
            if name == "count":
                out[key][name] = torch.from_numpy(array)
                continue
            if image:
                array = array.reshape(-1, 1, 1) / 255.0
            out[key][name] = torch.from_numpy(array.astype(np.float32))
    return out
