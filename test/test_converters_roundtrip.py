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

"""Full encode->decode round-trip tests for every codec pair.

DEVCONTAINER-ONLY: encode_value builds real ROS message instances via
rosidl_runtime_py.get_message, so this needs the ROS message packages
installed. The module is skipped when they are unavailable; individual pairs
whose message package is missing (e.g. control_msgs) skip per-parameter.

Coverage is enforced: ``test_every_pair_has_a_sample`` fails if a registered
encoder/decoder pair has no sample here, so a new pair cannot be added without
a round-trip case -- closing the historical 1-of-N gap for good.
"""

import numpy as np
import pytest

# Skip the whole module if real ROS message classes aren't importable.
get_message = pytest.importorskip('rosidl_runtime_py.utilities').get_message
try:
    get_message('sensor_msgs/msg/JointState')
except Exception:  # noqa: BLE001 - any failure means ROS msgs unavailable
    pytest.skip('ROS message packages unavailable', allow_module_level=True)

# Import for @register_decoder / @register_encoder side effects.
import rosetta.ros2.decoders  # noqa: E402,F401
import rosetta.ros2.encoders  # noqa: E402,F401
from rosetta.core.contract import load_contract  # noqa: E402
from rosetta.core.contract_utils import (  # noqa: E402
    iter_action_specs,
    iter_observation_specs,
)
from rosetta.core.converters import (  # noqa: E402
    DECODERS,
    ENCODER_ROUNDTRIP,
    ENCODERS,
    decode_value,
    encode_value,
)

# Every type that has BOTH an encoder and a decoder is round-trippable and must
# have a sample below. type -> (select, sample_vector, atol).
# atol > 0 for float32-backed codecs (storage precision); 0 means bit-exact.
SAMPLES: dict[str, tuple[list[str], list[float], float]] = {
    'sensor_msgs/msg/JointState': (['position.j1', 'position.j2'], [0.5, -0.3], 0.0),
    'geometry_msgs/msg/Twist': (
        ['linear.x', 'linear.y', 'linear.z', 'angular.x', 'angular.y', 'angular.z'],
        [0.1, 0.2, 0.3, -0.1, -0.2, -0.3],
        0.0,
    ),
    'geometry_msgs/msg/TwistStamped': (['linear.x', 'angular.z'], [0.7, -0.4], 0.0),
    'std_msgs/msg/Float64': (['v'], [1.5], 0.0),
    'std_msgs/msg/Float32': (['v'], [1.25], 1e-6),
    'std_msgs/msg/Float64MultiArray': (['a', 'b', 'c'], [0.1, 0.2, 0.3], 0.0),
    'std_msgs/msg/Float32MultiArray': (['a', 'b', 'c'], [0.25, 0.5, -0.75], 1e-6),
    'std_msgs/msg/Int32MultiArray': (['a', 'b', 'c'], [1, 2, 3], 0.0),
    'sensor_msgs/msg/Joy': (['axes.0', 'axes.1', 'buttons.0'], [0.5, -0.25, 1.0], 1e-6),
    'trajectory_msgs/msg/JointTrajectory': (
        ['position.j1', 'position.j2'],
        [0.5, -0.3],
        0.0,
    ),
    'control_msgs/msg/MultiDOFCommand': (['values.d1', 'values.d2'], [0.5, -0.3], 0.0),
}


def _pairs() -> list[str]:
    """Message types that have both an encoder and a decoder."""
    return sorted(set(DECODERS) & set(ENCODERS))


def _specs(tmp_path, msg_type: str, select: list[str], apply_block: str = '[]'):
    """Build (action_spec, observation_spec) for a type from a tiny contract."""
    sel = '[' + ', '.join(select) + ']'
    yaml = f"""
robot_type: test
fps: 30
observations:
  - {{key: observation.x, topic: /t, type: {msg_type}, select: {sel}, apply: {apply_block}}}
actions:
  - {{key: action, topic: /t, type: {msg_type}, select: {sel}, apply: {apply_block}}}
"""
    p = tmp_path / 'c.yaml'
    p.write_text(yaml)
    contract = load_contract(p)
    return (
        next(iter(iter_action_specs(contract))),
        next(iter(iter_observation_specs(contract))),
    )


def test_every_pair_has_a_sample():
    """Coverage guard: no registered codec pair may lack a round-trip sample."""
    missing = sorted(set(_pairs()) - set(SAMPLES))
    assert not missing, (
        f'Codec pairs without a round-trip sample in SAMPLES: {missing}. '
        f'Add a sample so the pair is round-trip tested.'
    )


@pytest.mark.parametrize('msg_type', _pairs())
def test_codec_pair_round_trips(msg_type, tmp_path):
    """decode(encode(v)) == v for every round-trippable pair; weak check if lossy."""
    try:
        get_message(msg_type)
    except Exception:  # noqa: BLE001 - message package not installed in this env
        pytest.skip(f'{msg_type} message package unavailable')

    assert msg_type in SAMPLES, f'No sample for {msg_type}'
    select, vec, atol = SAMPLES[msg_type]
    action_spec, obs_spec = _specs(tmp_path, msg_type, select)
    v = np.asarray(vec, dtype=np.float64)

    out1 = np.asarray(decode_value(encode_value(action_spec, vec), obs_spec), dtype=np.float64)

    if ENCODER_ROUNDTRIP.get(msg_type, True):
        # Strong invariant: the value survives the round trip.
        assert np.allclose(out1, v, rtol=0, atol=atol), f'{msg_type}: {v} -> {out1}'
    else:
        # Weak invariant for a declared-lossy encoder: stable after first pass.
        out2 = np.asarray(
            decode_value(encode_value(action_spec, out1.tolist()), obs_spec), dtype=np.float64
        )
        assert np.allclose(out1, out2, rtol=0, atol=atol), (
            f'{msg_type} not stable: {out1} -> {out2}'
        )


def test_roundtrip_rad2deg(tmp_path):
    """Round trip through the op pipeline (deg<->rad), not just the raw codec."""
    action_spec, obs_spec = _specs(
        tmp_path, 'sensor_msgs/msg/JointState', ['position.j1', 'position.j2'],
        apply_block='[rad2deg]',
    )
    values = [90.0, 45.0]
    out = decode_value(encode_value(action_spec, values), obs_spec)
    assert np.allclose(out, values)
