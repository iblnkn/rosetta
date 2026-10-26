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
LeRobot Teleoperator plugin for Rosetta.

This package provides the RosettaTeleop teleoperator implementation that bridges
ROS2 topics to LeRobot's Teleoperator interface. It is auto-discovered by LeRobot
via the `register_third_party_plugins()` mechanism when the package name starts
with `lerobot_teleoperator_`.

Usage:
    # Auto-discovered when using LeRobot - just specify type in config:
    teleop:
        type: rosetta_teleop
        config_path: contracts/my_robot.yaml

    # Or import directly:
    from lerobot_teleoperator_rosetta import RosettaTeleop, RosettaTeleopConfig
"""

from .config_rosetta_teleop import RosettaTeleopConfig
from .rosetta_teleop import RosettaTeleop

__all__ = [
    "RosettaTeleop",
    "RosettaTeleopConfig",
]
