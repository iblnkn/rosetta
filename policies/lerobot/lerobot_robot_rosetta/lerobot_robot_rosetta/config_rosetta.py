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
RosettaConfig: LeRobot RobotConfig that loads from contract YAML.

Example contract (single topic per key):
    robot_type: my_robot
    fps: 30

    observations:
      - key: observation.state
        topic: /joint_states
        type: sensor_msgs/msg/JointState
        selector:
          names: [position.joint1, position.joint2]

    actions:
      - key: action
        publish:
          topic: /cmd_joint
          type: sensor_msgs/msg/JointState
        selector:
          names: [position.joint1, position.joint2]

Example contract (multiple topics aggregated to same key):
    robot_type: mobile_manipulator
    fps: 30

    observations:
      # Multiple topics with same key -> concatenated into single tensor
      # Namespace is auto-derived from topic structure (arm, base)
      - key: observation.state
        topic: /arm/joint_states
        type: sensor_msgs/msg/JointState
        selector:
          names: [position.j1, position.j2]

      - key: observation.state
        topic: /base/odom
        type: nav_msgs/msg/Odometry
        selector:
          names: [twist.twist.linear.x, twist.twist.angular.z]

    actions:
      - key: action
        publish:
          topic: /arm/command
          type: sensor_msgs/msg/JointState
        selector:
          names: [position.j1, position.j2]

      - key: action
        publish:
          topic: /base/cmd_vel
          type: geometry_msgs/msg/Twist
        selector:
          names: [linear.x, angular.z]

    # Result: observation.state has names:
    #   ["arm.position.j1", "arm.position.j2", "base.twist.twist.linear.x", "base.twist.twist.angular.z"]
    # Result: action has names:
    #   ["arm.position.j1", "arm.position.j2", "base.linear.x", "base.angular.z"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.robots.config import RobotConfig

from rosetta.common.contract import (
    Contract,
    ObservationStreamSpec,
    ActionStreamSpec,
    load_contract,
)
from rosetta.common.contract_utils import iter_observation_specs, iter_action_specs


@RobotConfig.register_subclass("rosetta")
@dataclass
class RosettaConfig(RobotConfig):
    """LeRobot RobotConfig backed by a contract YAML file."""

    config_path: str = ""
    fps: int | None = None

    _contract: Contract | None = field(default=None, init=False, repr=False)
    _observation_specs: list[ObservationStreamSpec] | None = field(default=None, init=False, repr=False)
    _action_specs: list[ActionStreamSpec] | None = field(default=None, init=False, repr=False)
    _external_bridge: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()

        if not self.config_path:
            return

        self._contract = load_contract(self.config_path)

        if self.fps is None:
            self.fps = self._contract.fps

        if self.id is None:
            self.id = self._contract.robot_type

        self._observation_specs = list(iter_observation_specs(self._contract))
        self._action_specs = list(iter_action_specs(self._contract))

    @property
    def contract(self) -> Contract:
        if self._contract is None:
            raise ValueError("No contract loaded")
        return self._contract

    @property
    def observation_specs(self) -> list[ObservationStreamSpec]:
        if self._observation_specs is None:
            raise ValueError("No contract loaded")
        return self._observation_specs

    @property
    def action_specs(self) -> list[ActionStreamSpec]:
        if self._action_specs is None:
            raise ValueError("No contract loaded")
        return self._action_specs


def load_rosetta_config(path: str | Path, fps: int | None = None) -> RosettaConfig:
    """Load a RosettaConfig from YAML."""
    return RosettaConfig(config_path=str(path), fps=fps)
