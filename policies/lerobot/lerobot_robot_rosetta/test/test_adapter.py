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

"""Round-trip tests for the Rosetta(Robot) flatten/unflatten adapter."""

from pathlib import Path

import numpy as np
import pytest
import rosetta.robots.ros2.offline.bag_frames  # noqa: F401  register decoders
from ament_index_python.packages import get_package_share_directory
from lerobot_robot_rosetta.config_rosetta import RosettaConfig
from lerobot_robot_rosetta.rosetta import Rosetta

CONTRACTS = Path(get_package_share_directory("rosetta")) / "contracts"


def _robot(contract_name):
    cfg = RosettaConfig(config_path=str(CONTRACTS / contract_name))
    return Rosetta(cfg)


@pytest.mark.parametrize("contract", ["so_101.yaml", "turtlebot3.yaml"])
def test_action_unflatten_roundtrip(contract):
    robot = _robot(contract)
    # Every action key's namespaced names, in order.
    for key, names in robot._act_vector_names.items():
        vec = np.arange(len(names), dtype=np.float32) + 0.5
        flat = {name: float(vec[i]) for i, name in enumerate(names)}
        out = robot._unflatten_action(flat)
        assert key in out
        assert np.allclose(out[key], vec), (key, out[key], vec)


def test_observation_flatten_state_and_images():
    robot = _robot("so_101.yaml")
    names = robot._obs_vector_names["observation.state"]
    vec = np.arange(len(names), dtype=np.float32) + 1.0

    frame = {"observation.state": vec}
    for full_key in robot._obs_image_keys:
        frame[full_key] = np.zeros((4, 4, 3), dtype=np.uint8)

    obs = robot._flatten_observation(frame)

    # state floats land under their namespaced names, in order
    for i, name in enumerate(names):
        assert obs[name] == float(vec[i])
    # images pass through under their short keys
    for full_key, short in robot._obs_image_keys.items():
        assert short in obs
        assert obs[short] is frame[full_key]


@pytest.mark.parametrize("contract", ["so_101.yaml", "turtlebot3.yaml"])
def test_adapter_names_agree_with_frame_layout(contract):
    # The adapter's per-key name order must match the canonical FrameLayout
    # (key-major, declaration order within key) that assembles the frames it
    # flattens — otherwise live features and frame data disagree.
    from rosetta.contract.schema import load_contract
    from rosetta.contract.specs import iter_action_specs, iter_observation_specs
    from rosetta.frames.layout import FrameLayout

    robot = _robot(contract)
    c = load_contract(CONTRACTS / contract)
    specs = list(iter_observation_specs(c)) + list(iter_action_specs(c))
    feats = FrameLayout(specs).lerobot_features()
    for key, names in robot._obs_vector_names.items():
        assert feats[key]["names"] == names, key
    for key, names in robot._act_vector_names.items():
        assert feats[key]["names"] == names, key


def test_namespaced_multi_spec_state(contract="turtlebot3.yaml"):
    # turtlebot3 aggregates 3 specs under observation.state -> namespaced names.
    robot = _robot(contract)
    names = robot._obs_vector_names["observation.state"]
    # names carry a namespace prefix (derived from the 3 topics)
    assert any("." in n for n in names)
    vec = np.arange(len(names), dtype=np.float32)
    frame = {"observation.state": vec}
    for full_key in robot._obs_image_keys:
        frame[full_key] = np.zeros((4, 4, 3), dtype=np.uint8)
    obs = robot._flatten_observation(frame)
    reconstructed = np.array([obs[n] for n in names], dtype=np.float32)
    assert np.allclose(reconstructed, vec)
