# Copyright 2026 Isaac Blankenau
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

"""The episode rules import nothing from ROS, so they test and reuse without it."""

import subprocess
import sys

PROBE = """
import sys
import rosetta_ros_hil.hil_episode
leaked = sorted(m for m in sys.modules if m.split('.')[0] in {'rclpy', 'rosidl_runtime_py', 'rosetta_ros'}
                or m.split('.')[0].endswith('_interfaces') or m.split('.')[0].endswith('_msgs'))
assert not leaked, f'ROS modules imported by hil_episode: {leaked}'
"""


def test_hil_episode_imports_without_ros():
    # A subprocess gives a clean sys.modules regardless of what this session imported.
    subprocess.run([sys.executable, "-c", PROBE], check=True)
