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
rosetta: ROS 2 to LeRobot bridge.

This package provides common utilities (contract parsing, encoders/decoders)
and ROS2 nodes (episode_recorder, rosetta_client).

For LeRobot integration, the functionality is split into separate packages:
- lerobot_robot_rosetta: Robot plugin (auto-discovered by LeRobot)
- lerobot_teleoperator_rosetta: Teleoperator plugin (auto-discovered by LeRobot)
- rosetta_rl: RL training components (RosettaRobotEnv, actor/learner)

Usage (Robot - via plugin package):
    from lerobot_robot_rosetta import Rosetta, RosettaConfig

    config = RosettaConfig(config_path="contract.yaml")
    robot = Rosetta(config)
    robot.connect()

Usage (Teleoperator - via plugin package):
    from lerobot_teleoperator_rosetta import RosettaTeleop, RosettaTeleopConfig

    config = RosettaTeleopConfig(config_path="contract.yaml")
    teleop = RosettaTeleop(config)
    teleop.connect()

Usage (RL Training - via rl package):
    from rosetta_rl import setup_rosetta_training

    env, teleop, env_proc, act_proc = setup_rosetta_training(
        "contract.yaml", device="cuda", use_gripper=True
    )

Usage (Common utilities):
    from rosetta.common import load_contract, Contract
"""

# Re-export common utilities
from .common.contract import (
    ActionStreamSpec,
    Contract,
    ObservationStreamSpec,
    ResetMode,
    ResetSpec,
    load_contract,
)
from .common.contract_utils import iter_action_specs, iter_observation_specs

__all__ = [
    # Contract utilities
    "Contract",
    "load_contract",
    "ObservationStreamSpec",
    "ActionStreamSpec",
    "ResetSpec",
    "ResetMode",
    "iter_observation_specs",
    "iter_action_specs",
]
