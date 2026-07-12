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
this test asserts one of:

  1. the live layout is EQUIVALENT to the offline dataset layout
     (same keys, same shapes, names in the same order), or
  2. the connect/setup gate (ensure_live_observation_layout) rejects the
     contract, so the mismatch can never silently reach a robot.

Any future contract or lerobot upgrade that re-introduces silent lumping
fails here.
"""

from pathlib import Path

import pytest
import rosetta.robots.ros2.offline.bag_frames  # noqa: F401  register codecs
from lerobot.utils.feature_utils import hw_to_dataset_features
from lerobot_robot_rosetta.config_rosetta import RosettaConfig
from lerobot_robot_rosetta.dataset_writer import build_lerobot_features
from lerobot_robot_rosetta.rosetta import Rosetta, ensure_live_observation_layout
from rosetta.contract.schema import Align, Channel, Source, load_contract
from rosetta.contract.specs import ActionStreamSpec, ObservationStreamSpec, iter_action_specs, iter_observation_specs

CONTRACTS = sorted((Path(__file__).resolve().parents[2] / "rosetta" / "contracts").glob("*.yaml"))
BOUNDARY_MARKERS = {"is_first", "is_last", "is_terminal"}


def _dtype_family(dtype: str) -> str:
    """Live declares float32/video; offline may resolve float64/image. Compare
    families: policies normalize numerics to float32 downstream either way."""
    if dtype in ("video", "image"):
        return "visual"
    if dtype == "string":
        return "string"
    return "numeric"


SELECTLESS_CONTRACT = """
robot_type: test
robot_interface: ros2
fps: 30
observations:
  observation.state:
    channel: {topic: /joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
actions:
  action:
    channel: {topic: /cmd, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.j1]
"""


def test_selectless_numeric_stream_rejected(tmp_path):
    """The live stack derives motor names from selector names, so a numeric
    stream without select: is silently dropped live while the offline dataset
    keeps it — the gate must fire instead."""
    p = tmp_path / "c.yaml"
    p.write_text(SELECTLESS_CONTRACT)
    contract = load_contract(p)
    with pytest.raises(ValueError, match="without select"):
        ensure_live_observation_layout(list(iter_observation_specs(contract)), list(iter_action_specs(contract)))


@pytest.mark.parametrize("contract_path", CONTRACTS, ids=lambda p: p.name)
def test_live_features_match_offline_or_gate_fires(contract_path):
    contract = load_contract(contract_path)
    obs_specs = list(iter_observation_specs(contract))
    act_specs = list(iter_action_specs(contract))

    try:
        ensure_live_observation_layout(obs_specs, act_specs)
    except ValueError as e:
        # Gate fires in lieu of parity: the error must name the keys so the
        # user can act on it.
        assert "observation" in str(e) or "action" in str(e)
        return

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


# The negative cases below build specs directly (no YAML needed).
def _obs(key, names, topic="/t"):
    return ObservationStreamSpec(
        key=key,
        names=list(names),
        fps=30,
        source=Source(
            channel=Channel(topic=topic, type="sensor_msgs/msg/JointState"),
            align=Align("hold", "receive"),
        ),
        is_image=False,
        image_resize=None,
        dtype="float64",
    )


def _act(key, names, topic="/cmd"):
    return ActionStreamSpec(
        key=key,
        names=list(names),
        fps=30,
        source=Source(
            channel=Channel(topic=topic, type="sensor_msgs/msg/JointState"),
            align=Align("hold", "receive"),
        ),
        dtype="float64",
    )


def test_gate_fires_on_second_numeric_observation_key():
    specs = [
        _obs("observation.state", ["a", "b"]),
        _obs("observation.environment_state", ["e"], topic="/env"),
    ]
    with pytest.raises(ValueError, match="observation.environment_state"):
        ensure_live_observation_layout(specs, [_act("action", ["x"])])


def test_gate_fires_on_second_action_key():
    with pytest.raises(ValueError, match="action.gripper"):
        ensure_live_observation_layout(
            [_obs("observation.state", ["a"])],
            [_act("action", ["x"]), _act("action.gripper", ["g"], topic="/grip")],
        )


def test_gate_passes_multi_topic_single_key():
    # Multi-topic aggregation under ONE key is the supported live pattern.
    specs = [
        _obs("observation.state", ["a", "b"], topic="/arm"),
        _obs("observation.state", ["g"], topic="/gripper"),
    ]
    ensure_live_observation_layout(specs, [_act("action", ["x"])])
