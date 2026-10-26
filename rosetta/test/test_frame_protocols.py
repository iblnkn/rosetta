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

"""Tests for the FrameIO seam and core's purity.

Core (``rosetta.contract``, ``rosetta.frames``, ``rosetta.policies``,
``rosetta.robots``) imports without any ROS dependency, so framework adapters
and robot interfaces other than ROS 2 are usable and testable without ROS.
"""

import subprocess
import sys

from rosetta.frames.protocols import FrameIO

NO_RCLPY_PROBE = """
import sys
import rosetta
import rosetta.contract.schema
import rosetta.contract.specs
import rosetta.frames.codecs
import rosetta.frames.layout
import rosetta.frames.naming
import rosetta.frames.resample
import rosetta.frames.protocols
import rosetta.policies
import rosetta.robots.protocols
import rosetta.robots.registry
assert 'rclpy' not in sys.modules, 'rclpy leaked into the core import graph'
assert 'rosidl_runtime_py' not in sys.modules, 'rosidl leaked into the core import graph'
"""


def test_core_imports_without_rclpy():
    # A subprocess gives a clean sys.modules regardless of what this test
    # session has already imported.
    subprocess.run([sys.executable, "-c", NO_RCLPY_PROBE], check=True)


def test_minimal_double_satisfies_source_and_sink():
    # The surface framework-adapter test doubles rely on: sample/publish plus
    # warmup/safety/reset. Keeps the protocol honest about its minimum.
    class Double:
        warmed_up = True

        def sample_frame(self):
            return {}

        def publish_frame(self, action_frame):
            return action_frame

        def send_safety_action(self):
            pass

        def reset_state(self):
            pass

    assert isinstance(Double(), FrameIO)
