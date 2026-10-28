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

"""contract_output_remaps moves every topic a policy runner can publish, once each."""

from pathlib import Path

import pytest

from rosetta_ros.launch_utils import contract_output_remaps
from rosetta_ros.rclpy_utils import prefixed_topic

STONE = str(Path(__file__).resolve().parent.parent / "contracts" / "stone.yaml")

# stone.yaml: `action` spans the arm and gripper topics, `action.base` is /cmd_vel,
# and the single reward is /reward.
STONE_OUTPUTS = ["/arm/joint_commands", "/gripper/command", "/cmd_vel", "/reward"]


@pytest.mark.parametrize("prefix", ["/hil/policy", "/hil/policy/"])
def test_every_action_and_reward_topic_moves_under_the_prefix(prefix):
    assert contract_output_remaps(STONE, prefix) == [(t, f"/hil/policy{t}") for t in STONE_OUTPUTS]


def test_no_prefix_means_no_remaps():
    assert contract_output_remaps(STONE, "") == []


def test_a_prefix_without_a_contract_path_names_the_problem():
    with pytest.raises(ValueError, match="explicit contract_path"):
        contract_output_remaps("", "/hil/policy")


@pytest.mark.parametrize("prefix", ["/hil/policy", "/hil/policy/"])
@pytest.mark.parametrize("topic", ["/cmd_vel", "cmd_vel"])
def test_prefixed_topic_joins_with_exactly_one_slash(prefix, topic):
    """The runner's remap, the forwarder and the reward mux all name the moved topic this way."""
    assert prefixed_topic(prefix, topic) == "/hil/policy/cmd_vel"
