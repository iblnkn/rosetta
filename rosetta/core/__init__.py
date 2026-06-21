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
Rosetta core: framework-agnostic contract, ops, codecs, and resampling.

This package has no ROS or policy-framework dependencies. It defines the
contract schema, the op pipeline, the encode/decode registry + dispatch, the
stream resampler, and pure field-access helpers. ROS2 (``rosetta.ros2``) and the
framework leaves (``lerobot_robot_rosetta``, ``vla_foundry_rosetta``) are adapter
layers built on top of this core; backends are resolved via ``rosetta.backends``.
"""
