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

"""LeRobot leaf feature-schema unit test (runs in the `default` env)."""

from pathlib import Path

import rosetta.ros2.bag_frames  # noqa: F401  side-effect: register decoders/codecs
from rosetta.core.contract import load_contract
from rosetta.core.contract_utils import iter_action_specs, iter_observation_specs

from lerobot_robot_rosetta.dataset_writer import build_lerobot_features

# src/action/lerobot_robot_rosetta/test/ -> parents[2] == src/action
CONTRACTS = Path(__file__).resolve().parents[2] / 'rosetta' / 'contracts'


def _specs(name):
    c = load_contract(CONTRACTS / name)
    return c, list(iter_observation_specs(c)) + list(iter_action_specs(c))


def test_so101_features_schema():
    c, specs = _specs('so_101.yaml')
    feats = build_lerobot_features(specs)

    # state + action numeric vectors, three cameras, boundary markers.
    assert feats['observation.state']['dtype'] in ('float32', 'float64')
    assert feats['observation.state']['shape'] == (6,)
    assert feats['action']['shape'] == (6,)
    for cam in ('observation.images.right', 'observation.images.top', 'observation.images.wrist.right'):
        assert feats[cam]['dtype'] in ('video', 'image')
    for marker in ('is_first', 'is_last', 'is_terminal'):
        assert feats[marker] == {'dtype': 'bool', 'shape': (1,), 'names': None}


def test_same_key_aggregates_names():
    # turtlebot3 has multiple observation.state specs -> one aggregated feature.
    c, specs = _specs('turtlebot3.yaml')
    feats = build_lerobot_features(specs)
    assert feats['observation.state']['shape'][0] == len(feats['observation.state']['names'])
    assert feats['observation.state']['shape'][0] == 16
