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

"""Shared rclpy context for the whole test session.

rclpy.init() is global-context and process-wide — calling it a second time
before a prior call's rclpy.shutdown() raises, so test modules must not
initialize their own contexts in the shared pytest process. autouse=True so
it applies session-wide without every test file (pytest-native or
unittest.TestCase) needing to request it explicitly; unittest-style tests
can't take fixture parameters.
"""

import os

import pytest
import rclpy

# This package's tests run on their own domain, so a robot or sim on this
# machine, or another package's tests run in parallel by colcon, never shares
# their topics. Set at import, before any test creates a context.
os.environ["ROS_DOMAIN_ID"] = "82"


@pytest.fixture(scope="session", autouse=True)
def rclpy_ctx():
    rclpy.init()
    yield
    rclpy.try_shutdown()
