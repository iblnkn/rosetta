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

Adapts the framework-agnostic :mod:`rosetta.core` outputs to LeRobot: the
episodic dataset writer (``LeRobotDataset``) and the reward classifier gRPC
server. This is one consumer adapter; a future consumer (e.g. another VLA
stack) would add a sibling package with its own writer consuming the same
neutral core frames -- no shared base class is required (duck-typed). Imports
here may pull in lerobot.

Note: LeRobot policy *inference* is driven from the ROS2 client node
(``rosetta.ros2.nodes.rosetta_client_node``), which is the ROS↔LeRobot
composition root and owns the ``RobotClient`` lifecycle directly.
"""
