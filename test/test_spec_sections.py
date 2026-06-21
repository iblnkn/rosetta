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

"""Guard: the VLA writers consume observations and actions only, never the
extended sections (reward, signal, info, complementary_data) that the porter's
iter_specs also yields. This keeps RL and record-only columns out of
observation.state."""

import rosetta.ros2.bag_frames  # noqa: F401  register codecs
from rosetta.core.contract import ActionSpec, Contract, ObservationSpec
from rosetta.core.contract_utils import (
    iter_action_specs,
    iter_observation_specs,
    iter_specs,
)


def _contract_with_reward():
    obs = ObservationSpec(
        key='observation.state', topic='/joints', type='sensor_msgs/msg/JointState',
        select=['position.j1', 'position.j2'],
    )
    act = ActionSpec(
        key='action', topic='/cmd', type='sensor_msgs/msg/JointState',
        select=['position.j1', 'position.j2'],
    )
    # A reward whose key classifies as state ('next.reward' is not image or action).
    rew = ObservationSpec(
        key='next.reward', topic='/reward', type='std_msgs/msg/Float32',
        select=['data'], dtype='float32',
    )
    return Contract(
        robot_type='r', fps=30, observations=[obs], actions=[act], tasks=[],
        recording={}, adjunct=[], rewards=[rew], signals=[], info=[],
        complementary_data=[],
    )


def test_reward_leaks_into_iter_specs_but_not_obs_action():
    c = _contract_with_reward()
    all_keys = {s.key for s in iter_specs(c)}
    oa_keys = {s.key for s in iter_observation_specs(c)} | {s.key for s in iter_action_specs(c)}

    assert 'next.reward' in all_keys          # the porter's iter_specs carries it
    assert 'next.reward' not in oa_keys        # but the VLA writers (obs+action) don't
    assert oa_keys == {'observation.state', 'action'}
