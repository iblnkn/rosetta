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

import rosetta.robots.ros2.offline.bag_frames  # noqa: F401  side-effect: register decoders/codecs
from ament_index_python.packages import get_package_share_directory
from rosetta.contract.schema import load_contract
from rosetta.contract.specs import iter_action_specs, iter_observation_specs

from lerobot_robot_rosetta.dataset_writer import build_lerobot_features

CONTRACTS = Path(get_package_share_directory("rosetta")) / "contracts"


def _specs(name):
    c = load_contract(CONTRACTS / name)
    return c, list(iter_observation_specs(c)) + list(iter_action_specs(c))


def test_so101_features_schema():
    _c, specs = _specs("so_101.yaml")
    feats = build_lerobot_features(specs)

    # state + action numeric vectors, three cameras, boundary markers.
    assert feats["observation.state"]["dtype"] in ("float32", "float64")
    assert feats["observation.state"]["shape"] == (6,)
    assert feats["action"]["shape"] == (6,)
    for cam in ("observation.images.right", "observation.images.top", "observation.images.wrist.right"):
        assert feats[cam]["dtype"] in ("video", "image")
    for marker in ("is_first", "is_last", "is_terminal"):
        assert feats[marker] == {"dtype": "bool", "shape": (1,), "names": None}


def test_same_key_aggregates_names():
    # turtlebot3 has multiple observation.state specs -> one aggregated feature.
    _c, specs = _specs("turtlebot3.yaml")
    feats = build_lerobot_features(specs)
    assert feats["observation.state"]["shape"][0] == len(feats["observation.state"]["names"])
    assert feats["observation.state"]["shape"][0] == 16


def test_live_robot_declares_decoded_image_channels(tmp_path):
    """Regression: the live Robot's observation_features used
    spec.image_channels (the SOURCE encoding count: 1 for mono8, 4 for rgba8)
    while decoders always deliver (h, w, 3) — a shape mismatch for any
    non-3-channel source. Live features must declare the decoded shape."""
    yaml = """
robot_type: test
robot_interface: ros2
fps: 30
observations:
  observation.images.cam:
    channel: {topic: /cam, type: sensor_msgs/msg/Image}
    align: {strategy: hold, timeline: receive}
    apply: [{resize: [48, 64]}]
  observation.state:
    channel: {topic: /js, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.j1]
actions:
  action:
    channel: {topic: /cmd, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.j1]
"""
    p = tmp_path / "mono.yaml"
    p.write_text(yaml)

    from lerobot_robot_rosetta.config_rosetta import RosettaConfig
    from lerobot_robot_rosetta.rosetta import Rosetta

    robot = Rosetta(RosettaConfig(config_path=str(p)))
    assert robot.observation_features["cam"] == (48, 64, 3)  # not (48, 64, 1)

    # And it equals what the offline writer declares for the same contract.
    c, _ = _specs("so_101.yaml")
    del c
    contract = load_contract(p)
    offline = build_lerobot_features(list(iter_observation_specs(contract)) + list(iter_action_specs(contract)))
    assert tuple(offline["observation.images.cam"]["shape"]) == robot.observation_features["cam"]
