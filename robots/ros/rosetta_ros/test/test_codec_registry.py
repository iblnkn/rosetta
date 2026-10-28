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

"""The ros2 interface registers its built-in codecs when codecs are discovered."""

import subprocess
import sys

# Runs in a fresh interpreter: in this one, earlier test modules have already
# imported rosetta_ros.{decoders,encoders}, which registers the codecs whether
# or not entry-point discovery works.
_PROBE = """
from rosetta.frames.codecs import has_decoder, has_encoder
assert has_decoder("sensor_msgs/msg/JointState")
assert has_encoder("geometry_msgs/msg/Twist")
assert has_decoder("std_msgs/msg/Bool")  # required by stone.yaml's extended sections
"""


def test_discovery_registers_builtin_codecs():
    """Registry checks must not depend on whether rosetta_ros.{decoders,encoders}
    happened to be imported. The accessors are the API spec resolution uses."""
    proc = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=60, check=False)
    assert proc.returncode == 0, proc.stderr
