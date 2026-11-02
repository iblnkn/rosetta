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

"""The adapter imports no robot middleware: the robot side reaches it only through frames."""

import subprocess
import sys

import pytest

BLOCKED = ("rclpy", "rosidl_runtime_py", "rosetta_ros")


@pytest.mark.parametrize("module", ["lerobot_rosetta.policy_runner", "lerobot_rosetta.dataset_writer"])
def test_imports_with_ros_unavailable(module):
    # A None entry in sys.modules makes any import of that name raise ImportError.
    block = "; ".join(f"sys.modules[{name!r}] = None" for name in BLOCKED)
    code = f"import sys; {block}; import {module}"
    proc = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
