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
Rosetta LeRobot consumer layer.

LeRobot-specific code, built on the framework-agnostic :mod:`rosetta.core` +
:mod:`rosetta.ros2`:
- ``port_bags`` -- ROS2 bag -> LeRobotDataset converter (single-use, LeRobot-tied).
- ``classifier_server`` -- reward classifier gRPC server.

LeRobot policy *inference* is driven from the ROS2 client node
(``rosetta.ros2.nodes.rosetta_client_node``), the ROS<->LeRobot composition root
that owns the ``RobotClient`` lifecycle directly. Imports here may pull in lerobot.
"""
