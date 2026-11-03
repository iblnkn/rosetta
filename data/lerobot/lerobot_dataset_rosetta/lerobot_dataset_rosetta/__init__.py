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

"""Rosetta's LeRobot dataset writer.

Registers :class:`~lerobot_dataset_rosetta.dataset_writer.LeRobotDatasetWriter`
on the ``rosetta.dataset_writers`` entry-point group under the name
``lerobot``, which ``rosetta_port`` loads. The policy side,
``lerobot_rosetta``, is a separate package.

Import from the submodules. This module imports nothing.
"""
