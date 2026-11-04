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

"""In-memory frame readers for tests: frames held in a list, numbered and padded at episode edges."""

from pathlib import Path

import numpy as np
from test_dataset_writer import CONTRACT_YAML

from rosetta.contract.schema import load_contract, parse_contract
from rosetta.contract.specs import iter_observation_specs, iter_policy_specs, iter_specs
from rosetta.data import ReaderInfo
from rosetta.data.stats import frame_stats
from rosetta.frames.layout import FrameLayout

# Episode-stream markers the porter emits that a dataset doesn't carry.
_MARKERS = ("is_first", "is_last", "is_terminal")


class MemoryReader:
    """A FrameReader over ``episodes``, lists of frame dicts, with every window padded at the episode's edges."""

    def __init__(self, episodes, contract=None):
        self.episodes = episodes
        self.contract = contract if contract is not None else parse_contract(CONTRACT_YAML)
        self.layout = FrameLayout(list(iter_policy_specs(self.contract)))
        self.prepared = None

    def features(self):
        return self.layout.features()

    def prepare(self, windows):
        self.prepared = dict(windows)

    def info(self):
        return ReaderInfo(
            fps=self.contract.fps,
            robot_type=self.contract.robot_type,
            episode_lengths=tuple(map(len, self.episodes)),
            episode_tasks=tuple(e[0]["task"] for e in self.episodes),
            tasks=tuple(dict.fromkeys(f["task"] for e in self.episodes for f in e)),
            pads=True,
        )

    def stats(self):
        return frame_stats([f for e in self.episodes for f in e], self.layout)

    def read(self, indices):
        return [self._sample(int(i)) for i in indices]

    def _sample(self, index):
        tasks = list(self.info().tasks)
        episode, t = index, index
        for episode, frames in enumerate(self.episodes):  # noqa: B007
            if t < len(frames):
                break
            t -= len(frames)
        sample = _window(self.episodes[episode], self.prepared, t)
        sample.update(
            task_index=tasks.index(sample["task"]),
            episode_index=episode,
            frame_index=t,
            index=index,
            timestamp=t / self.contract.fps,
        )
        return sample


class PortedFramesReader(MemoryReader):
    """A MemoryReader over the porter's frames for each recording in ``raw_dir``, in name order.

    Built as a reader is from ``--rosetta.*`` flags; options it doesn't use are ignored.
    """

    def __init__(self, contract, raw_dir, **_options):
        from rosetta_ros.offline.bag_frames import find_bag_dirs, iter_bag_frames

        parsed = load_contract(contract)
        specs = list(iter_specs(parsed))
        warmup_keys = {s.key for s in iter_observation_specs(parsed)}
        episodes = []
        for bag_dir in find_bag_dirs(Path(raw_dir)):
            frames = list(iter_bag_frames(bag_dir, specs, warmup_keys=warmup_keys))
            for frame in frames:
                for marker in _MARKERS:
                    del frame[marker]
            episodes.append(frames)
        super().__init__(episodes, parsed)


def _window(frames, windows, t):
    """Frame ``t`` with each windowed key stacked over its offsets, padded at the episode's edges as LeRobot pads.

    An offset outside the episode repeats the edge frame's value, and
    ``<key>_is_pad`` is True there.
    """
    sample = dict(frames[t])
    last = len(frames) - 1
    for key, offsets in windows.items():
        ticks = t + np.asarray(list(offsets), dtype=np.int64)
        sample[key] = np.stack([np.asarray(frames[i][key]) for i in np.clip(ticks, 0, last)])
        sample[f"{key}_is_pad"] = (ticks < 0) | (ticks > last)
    return sample
