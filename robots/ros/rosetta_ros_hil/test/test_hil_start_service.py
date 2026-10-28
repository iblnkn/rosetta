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

"""~/start_episode runs an episode with the prompt, time limit and reward threshold it was given."""

from std_srvs.srv import SetBool

from rosetta_interfaces.srv import StartHILEpisode
from rosetta_ros.node_utils import wait_until
from rosetta_ros_hil.hil_episode import TERMINATION_REWARD_THRESHOLD, TERMINATION_TIMEOUT


def _start(node, **request):
    node._accepting_work = True
    return node._on_start_episode(StartHILEpisode.Request(**request), StartHILEpisode.Response())


def _result(node):
    """The ~/result the episode thread publishes, once it has released the work slot."""
    assert wait_until(lambda: not node.busy, timeout=5.0, poll=0.01)
    (result,) = node._result_pub.published
    return result


def test_start_runs_the_episode_with_the_requested_prompt_and_time_limit(node, child_prompts):
    response = _start(node, prompt="open the drawer", max_duration_s=0.05)
    result = _result(node)
    assert (response.accepted, set(child_prompts), result.termination_reason) == (
        True,
        {"open the drawer"},
        TERMINATION_TIMEOUT,
    )


def test_start_ends_the_episode_at_the_requested_reward_threshold(node, child_prompts):
    _start(node, prompt="t", success_reward_threshold=0.5)
    # Children start after the episode resets its label, so a label from here on sticks.
    assert wait_until(lambda: child_prompts, timeout=5.0, poll=0.01)
    node._on_set_reward_override(SetBool.Request(data=True), SetBool.Response())  # reward 1.0
    assert _result(node).termination_reason == TERMINATION_REWARD_THRESHOLD
