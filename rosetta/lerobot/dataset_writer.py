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
LeRobot dataset writer: the LeRobot consumer adapter for the porting pipeline.

Builds the LeRobot feature schema from contract specs and wraps
``LeRobotDataset`` behind a small writer interface that consumes the frame
stream produced by :mod:`rosetta.ros2.bag_reader`. This isolates the lerobot
dependency; a different consumer (e.g. another VLA dataset format) would
provide its own writer implementing the same surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from rosetta.core.contract import StreamSpec
from rosetta.core.contract_utils import build_feature, get_namespaced_names
from rosetta.core.converters import DTYPES


def build_features(specs: list[StreamSpec]) -> dict[str, dict[str, Any]]:
    """
    Build LeRobot feature definitions from contract specs.

    Specs sharing the same key are aggregated (names concatenated for vectors).
    Adds the episode boundary marker features (is_first/is_last/is_terminal).
    """
    # Group specs by output key
    by_key: dict[str, list[StreamSpec]] = {}
    for spec in specs:
        by_key.setdefault(spec.key, []).append(spec)

    features = {}
    for key, key_specs in by_key.items():
        first = key_specs[0]
        dtype = DTYPES[first.msg_type]

        if dtype in ('video', 'image'):
            # Images: no aggregation
            features[key] = build_feature(first)
        elif dtype == 'string':
            # Strings: no aggregation
            features[key] = build_feature(first)
        else:
            # Numeric: aggregate names from all specs
            all_names = []
            for spec in key_specs:
                all_names.extend(get_namespaced_names(spec))
            n = len(all_names) or 1
            features[key] = {
                'dtype': dtype,
                'shape': (n,),
                'names': all_names if all_names else None,
            }

    # Frame boundary markers
    features['is_first'] = {'dtype': 'bool', 'shape': (1,), 'names': None}
    features['is_last'] = {'dtype': 'bool', 'shape': (1,), 'names': None}
    features['is_terminal'] = {'dtype': 'bool', 'shape': (1,), 'names': None}

    return features


def create_dataset(
    repo_id: str,
    root: Path | None,
    robot_type: str,
    fps: int,
    features: dict[str, dict[str, Any]],
    vcodec: str = 'libsvtav1',
) -> LeRobotDataset:
    """
    Create a ``LeRobotDataset`` to write contract-derived frames into.

    Returned object exposes ``add_frame``/``save_episode``/``finalize``/
    ``push_to_hub``. This function is the single place the LeRobot dataset API
    is touched; a different consumer (other VLA dataset format) provides its own
    ``build_features`` + ``create_*`` returning an object with the same methods.
    """
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        robot_type=robot_type,
        fps=fps,
        features=features,
        vcodec=vcodec,
    )
