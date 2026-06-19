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

"""Full encode->decode round-trip tests.

DEVCONTAINER-ONLY: encode_value builds real ROS message instances via
rosidl_runtime_py.get_message, so this needs the ROS message packages
installed. The module is skipped when they are unavailable.
"""

import numpy as np
import pytest

# Skip the whole module if real ROS message classes aren't importable.
get_message = pytest.importorskip(
    'rosidl_runtime_py.utilities'
).get_message
try:
    get_message('sensor_msgs/msg/JointState')
except Exception:  # noqa: BLE001 - any failure means ROS msgs unavailable
    pytest.skip('ROS message packages unavailable', allow_module_level=True)

# Import for @register_decoder / @register_encoder side effects.
import rosetta.common.decoders  # noqa: E402,F401
import rosetta.common.encoders  # noqa: E402,F401
from rosetta.common.contract import load_contract  # noqa: E402
from rosetta.common.contract_utils import (  # noqa: E402
    iter_action_specs,
    iter_observation_specs,
)
from rosetta.common.converters import decode_value, encode_value  # noqa: E402


def _contract(tmp_path, apply_block):
    yaml = f"""
robot_type: test
fps: 30
observations:
  - key: observation.state
    topic: /joint_states
    type: sensor_msgs/msg/JointState
    select: [position.j1, position.j2]
    apply: {apply_block}
actions:
  - key: action
    topic: /joint_states
    type: sensor_msgs/msg/JointState
    select: [position.j1, position.j2]
    apply: {apply_block}
"""
    p = tmp_path / 'c.yaml'
    p.write_text(yaml)
    return load_contract(p)


def test_roundtrip_rad2deg(tmp_path):
    contract = _contract(tmp_path, '[rad2deg]')
    action_spec = next(iter(iter_action_specs(contract)))
    obs_spec = next(iter(iter_observation_specs(contract)))

    # Dataset-space values (degrees). encode runs inverse (deg->rad) into the
    # ROS message; decode runs forward (rad->deg) back out.
    values = [90.0, 45.0]
    msg = encode_value(action_spec, values)
    out = decode_value(msg, obs_spec)
    assert np.allclose(out, values)
