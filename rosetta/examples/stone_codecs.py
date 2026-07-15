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

"""Example custom codecs for the showcase contract (contracts/stone.yaml).

A contract channel can name an inline decoder/encoder as
``"module.path:function"`` — normally in your own package. These are the
worked examples stone.yaml points at, bundled so the showcase contract
load-validates anywhere Rosetta is installed. They double as the signature
reference for custom codecs:

- decoder: ``fn(msg, spec) -> np.ndarray | str`` — reduce the message to a
  flat numeric array (or string); the operator pipeline runs after.
- encoder: ``fn(action_vec, spec, stamp_ns) -> msg`` — scatter an
  already-operator-inverted value slice into a ROS message.
"""

from __future__ import annotations

import numpy as np


def decode_battery(msg, spec):
    """std_msgs/Float64MultiArray -> the pack cells we care about (first 4).

    The entry's ``select`` labels these four values and declares the stream's
    width; a decoder must return exactly ``len(spec.names)`` values (checked
    by decode_value).
    """
    del spec
    return np.asarray(msg.data[:4], dtype=np.float32)


def encode_gripper(action_vec, spec, stamp_ns):
    """One action dim -> std_msgs/Float64 gripper command."""
    from std_msgs.msg import Float64

    del spec, stamp_ns
    msg = Float64()
    msg.data = float(action_vec[0])
    return msg
