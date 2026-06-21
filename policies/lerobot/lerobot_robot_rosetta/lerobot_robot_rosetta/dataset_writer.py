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
# limitations under the License.

"""
LeRobot DatasetWriter.

Consumes backend-neutral frame dicts and writes a LeRobotDataset (Parquet +
MP4). The frame dicts from rosetta.ros2.bag_frames.iter_bag_frames already match
what LeRobotDataset.add_frame expects, so this writer just handles the dataset
lifecycle and builds the LeRobot feature schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from rosetta.core.contract import Contract, StreamSpec
from rosetta.core.contract_utils import build_feature, get_namespaced_names
from rosetta.core.converters import DTYPES


def build_lerobot_features(specs: list[StreamSpec]) -> dict[str, dict[str, Any]]:
    """
    Build LeRobot feature definitions from contract specs.

    Specs sharing a key are aggregated (names concatenated for vectors).
    Adds the is_first/is_last/is_terminal boundary markers.
    """
    by_key: dict[str, list[StreamSpec]] = {}
    for spec in specs:
        by_key.setdefault(spec.key, []).append(spec)

    features: dict[str, dict[str, Any]] = {}
    for key, key_specs in by_key.items():
        first = key_specs[0]
        dtype = DTYPES[first.msg_type]

        if dtype in ('video', 'image', 'string'):
            # Images / strings: no aggregation
            features[key] = build_feature(first)
        else:
            # Numeric: aggregate names from all specs sharing the key
            all_names: list[str] = []
            for spec in key_specs:
                all_names.extend(get_namespaced_names(spec))
            n = len(all_names) or 1
            features[key] = {
                'dtype': dtype,
                'shape': (n,),
                'names': all_names if all_names else None,
            }

    features['is_first'] = {'dtype': 'bool', 'shape': (1,), 'names': None}
    features['is_last'] = {'dtype': 'bool', 'shape': (1,), 'names': None}
    features['is_terminal'] = {'dtype': 'bool', 'shape': (1,), 'names': None}
    return features


class LeRobotDatasetWriter:
    """DatasetWriter backed by LeRobotDataset."""

    def __init__(self) -> None:
        self._ds: LeRobotDataset | None = None
        self._push_to_hub = False
        self._tags: list[str] = ['rosetta', 'rosbag']

    def open(
        self,
        *,
        contract: Contract,
        specs: list[StreamSpec],
        repo_id: str,
        root: Path | None = None,
        vcodec: str = 'libsvtav1',
        push_to_hub: bool = False,
        **_opts: Any,
    ) -> None:
        """Create the LeRobotDataset with features derived from contract specs."""
        # LeRobot uses root directly as the dataset path, so append repo_id.
        dataset_root = Path(root) / repo_id if root else None
        self._push_to_hub = push_to_hub
        self._ds = LeRobotDataset.create(
            repo_id=repo_id,
            root=dataset_root,
            robot_type=contract.robot_type,
            fps=contract.fps,
            features=build_lerobot_features(specs),
            vcodec=vcodec,
        )

    def add_frame(self, frame: dict[str, Any]) -> None:
        assert self._ds is not None, 'open() must be called before add_frame()'
        self._ds.add_frame(frame)

    def save_episode(self) -> None:
        assert self._ds is not None, 'open() must be called before save_episode()'
        self._ds.save_episode()

    def finalize(self) -> None:
        assert self._ds is not None, 'open() must be called before finalize()'
        self._ds.finalize()
        if self._push_to_hub:
            self._ds.push_to_hub(tags=self._tags, private=False)
