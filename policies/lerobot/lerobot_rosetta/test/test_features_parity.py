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

"""Live vs offline LeRobot feature-layout parity (the standing guarantee).

lerobot's hw_to_dataset_features collapses every numeric leaf into one
observation.state (and every action leaf into one action). rosetta's offline
writer keeps one feature per contract key — the layout policies actually
train on. For EVERY bundled contract (glob — new contracts are auto-covered)
the live layout must be EQUIVALENT to the offline dataset layout (same keys,
same shapes, names in the same order), unless the contract is listed in
LIVE_REFUSED: then the connect/setup gate (ensure_live_observation_layout)
must reject it, naming the keys the live stack would lump. A contract that
gates without being listed fails here, as does any contract or lerobot upgrade
that re-introduces silent lumping.

lerobot_robot_rosetta/test/test_live_layout_gate.py pins what
ensure_live_observation_layout rejects.
"""

from pathlib import Path

import pytest
from ament_index_python.packages import get_package_share_directory
from lerobot.utils.feature_utils import hw_to_dataset_features

from lerobot_robot_rosetta.config_rosetta import RosettaConfig
from lerobot_robot_rosetta.rosetta import Rosetta, ensure_live_observation_layout
from lerobot_rosetta.dataset_writer import build_lerobot_features
from rosetta.contract.schema import load_contract
from rosetta.contract.specs import iter_action_specs, iter_observation_specs
from rosetta.frames.codecs import discover_codecs

discover_codecs()  # register the built-in codecs

CONTRACTS = sorted((Path(get_package_share_directory("rosetta_ros")) / "contracts").glob("*.yaml"))
BOUNDARY_MARKERS = {"is_first", "is_last", "is_terminal"}
# Bundled contracts that keep per-key layouts the live stack would lump -> the keys the gate must name.
LIVE_REFUSED = {"stone.yaml": ["observation.imu", "observation.battery", "action.base"]}


def _dtype_family(dtype: str) -> str:
    """Live declares float32/video; offline may resolve float64/image. Compare
    families: policies normalize numerics to float32 downstream either way."""
    if dtype in ("video", "image"):
        return "visual"
    if dtype == "string":
        return "string"
    return "numeric"


@pytest.mark.parametrize("contract_path", CONTRACTS, ids=lambda p: p.name)
def test_live_features_match_offline_or_gate_fires(contract_path):
    contract = load_contract(contract_path)
    obs_specs = list(iter_observation_specs(contract))
    act_specs = list(iter_action_specs(contract))

    if contract_path.name in LIVE_REFUSED:
        with pytest.raises(ValueError) as refused:
            ensure_live_observation_layout(obs_specs, act_specs)
        assert [key for key in LIVE_REFUSED[contract_path.name] if key not in str(refused.value)] == []
        return
    ensure_live_observation_layout(obs_specs, act_specs)
    robot = Rosetta(RosettaConfig(id="parity", config_path=str(contract_path)))
    live = hw_to_dataset_features(robot.observation_features, "observation")
    live.update(hw_to_dataset_features(robot.action_features, "action"))
    offline = {k: v for k, v in build_lerobot_features(obs_specs + act_specs).items() if k not in BOUNDARY_MARKERS}

    assert set(live) == set(offline), (
        f"live/offline feature key sets diverge:\n"
        f"  live only: {set(live) - set(offline)}\n"
        f"  offline only: {set(offline) - set(live)}"
    )
    for key, off_f in offline.items():
        live_f = live[key]
        assert _dtype_family(live_f["dtype"]) == _dtype_family(off_f["dtype"]), key
        assert tuple(live_f["shape"]) == tuple(off_f["shape"]), (
            f"{key}: live shape {live_f['shape']} != offline {off_f['shape']}"
        )
        if _dtype_family(off_f["dtype"]) == "numeric":
            # Order-sensitive: catches silent channel misordering.
            assert list(live_f["names"]) == list(off_f["names"]), (
                f"{key}: live names order {live_f['names']} != offline {off_f['names']}"
            )
