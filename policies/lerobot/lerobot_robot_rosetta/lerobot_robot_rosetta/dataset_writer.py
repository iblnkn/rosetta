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

Consumes framework-neutral frame dicts and writes a LeRobotDataset (Parquet +
MP4). The frame dicts from rosetta.robots.ros2.bag_frames.iter_bag_frames already match
what LeRobotDataset.add_frame expects, so this writer just handles the dataset
lifecycle and builds the LeRobot feature schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lerobot.configs import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from rosetta.contract.schema import Contract
from rosetta.contract.specs import StreamSpec
from rosetta.frames.layout import FrameLayout


def build_lerobot_features(specs: list[StreamSpec]) -> dict[str, dict[str, Any]]:
    """
    Build LeRobot feature definitions from contract specs.

    Delegates key aggregation to the core FrameLayout (specs sharing a key
    concatenate; the feature dtype is the spec's resolved dtype, so explicit
    overrides and custom-decoder streams work). Adds the
    is_first/is_last/is_terminal boundary markers.
    """
    features = FrameLayout(specs).lerobot_features()
    features["is_first"] = {"dtype": "bool", "shape": (1,), "names": None}
    features["is_last"] = {"dtype": "bool", "shape": (1,), "names": None}
    features["is_terminal"] = {"dtype": "bool", "shape": (1,), "names": None}
    return features


class LeRobotDatasetWriter:
    """DatasetWriter backed by LeRobotDataset."""

    def __init__(self) -> None:
        self._ds: LeRobotDataset | None = None
        self._push_to_hub = False
        self._hub_private = True  # publishing must be an explicit choice
        self._tags: list[str] = ["rosetta", "rosbag"]

    def open(
        self,
        *,
        contract: Contract,
        specs: list[StreamSpec],
        repo_id: str,
        root: Path | None = None,
        vcodec: str = "libsvtav1",
        push_to_hub: bool = False,
        hub_private: bool = True,
        hub_tags: list[str] | None = None,
        **_opts: Any,
    ) -> None:
        """Create the LeRobotDataset with features derived from contract specs."""
        # LeRobot uses root directly as the dataset path, so append repo_id.
        dataset_root = Path(root) / repo_id if root else None
        self._push_to_hub = push_to_hub
        self._hub_private = hub_private
        if hub_tags is not None:
            self._tags = list(hub_tags)
        self._ds = LeRobotDataset.create(
            repo_id=repo_id,
            root=dataset_root,
            robot_type=contract.robot_type,
            fps=contract.fps,
            features=build_lerobot_features(specs),
            rgb_encoder=RGBEncoderConfig(vcodec=vcodec),
        )

    def add_frame(self, frame: dict[str, Any]) -> None:
        assert self._ds is not None, "open() must be called before add_frame()"
        self._ds.add_frame(frame)

    def save_episode(self) -> None:
        assert self._ds is not None, "open() must be called before save_episode()"
        self._ds.save_episode()

    def discard_episode(self) -> None:
        """Drop the partially buffered episode (safe when nothing is buffered)."""
        assert self._ds is not None, "open() must be called before discard_episode()"
        self._ds.clear_episode_buffer()

    def finalize(self) -> None:
        assert self._ds is not None, "open() must be called before finalize()"
        self._ds.finalize()
        if self._push_to_hub:
            self._ds.push_to_hub(tags=self._tags, private=self._hub_private)
