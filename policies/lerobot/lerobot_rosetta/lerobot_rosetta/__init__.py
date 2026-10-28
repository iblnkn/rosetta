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

"""Rosetta's LeRobot backend adapter.

Dataset writer, policy runner, and gRPC inference servers, registered with
rosetta via entry points (rosetta.dataset_writers / rosetta.policy_runners,
name 'lerobot'). Deliberately NOT named with a 'lerobot_robot_*'/
'lerobot_teleoperator_*' prefix so LeRobot's register_third_party_plugins()
does not import it; the LeRobot-discovered Robot plugin lives in
lerobot_robot_rosetta.

Import from the submodules. This module imports nothing, so the inference
servers (``python -m lerobot_rosetta.policy_server``) load without ROS and can
run on a GPU host that has none.
"""
