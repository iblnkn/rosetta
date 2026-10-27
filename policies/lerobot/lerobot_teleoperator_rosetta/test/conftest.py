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

"""Shared rclpy context and domain for the teleoperator tests.

The teleop hosts its own private context; the session context here is for
the tests' fixture publishers.
"""

import os

import pytest
import rclpy

# See rosetta_ros/test/conftest.py: one domain per package's tests.
os.environ["ROS_DOMAIN_ID"] = "85"


@pytest.fixture(scope="session")
def ros():
    rclpy.init()
    yield
    rclpy.try_shutdown()
