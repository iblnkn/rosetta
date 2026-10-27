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

"""build_lerobot_features adds the episode-boundary markers LeRobot datasets carry on top of the frame layout."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from lerobot_rosetta.dataset_writer import build_lerobot_features
from rosetta.contract.schema import load_contract
from rosetta.contract.specs import iter_action_specs, iter_observation_specs
from rosetta.frames.codecs import discover_codecs

discover_codecs()  # register the built-in codecs

CONTRACT = Path(get_package_share_directory("rosetta_ros")) / "contracts" / "so_101.yaml"


def test_boundary_markers_are_declared():
    contract = load_contract(CONTRACT)
    feats = build_lerobot_features(list(iter_observation_specs(contract)) + list(iter_action_specs(contract)))
    for marker in ("is_first", "is_last", "is_terminal"):
        assert feats[marker] == {"dtype": "bool", "shape": (1,), "names": None}
