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

"""Each action must declare every reason its own server can produce.

A reason a client cannot name is a reason it cannot branch on. The .action
files are the single source of these values -- ``node_utils`` re-exports them
rather than restating them -- so what is left to check is coverage: that
``RecordEpisode`` declares everything the recorder can report, and so on.

Also pins the shape of the vocabulary: no core result carries ``outcome``
(apps such as rosetta_ros_hil add it where a task verdict is meaningful), and no
result carries a ``success`` flag.
"""

import pytest

from rosetta_interfaces.action import RecordEpisode, RunPolicy
from rosetta_ros import node_utils

# What each action's server can produce, as the node code decides it.
PRODUCED = {
    RecordEpisode.Result: {
        node_utils.TERMINATION_STOPPED,
        node_utils.TERMINATION_TIMEOUT,
        node_utils.TERMINATION_CANCELLED,
        node_utils.TERMINATION_NODE_DEACTIVATED,
        node_utils.TERMINATION_ERROR,
    },
    RunPolicy.Result: {
        node_utils.TERMINATION_COMPLETED,
        node_utils.TERMINATION_TIMEOUT,
        node_utils.TERMINATION_CANCELLED,
        node_utils.TERMINATION_NODE_DEACTIVATED,
        node_utils.TERMINATION_ERROR,
    },
}


def _declared(result_type, prefix):
    return {getattr(result_type, name) for name in dir(result_type) if name.startswith(prefix)}


@pytest.mark.parametrize("result_type", list(PRODUCED))
def test_a_shared_reason_has_the_same_value_in_every_action(result_type):
    # Each .action declares its own string literals, so `timeout` could easily
    # become `timed_out` in one of them. node_utils sources each name from one
    # action; this is what stops the others drifting from it.
    for name in dir(result_type):
        if not name.startswith(("TERMINATION_", "OUTCOME_")):
            continue
        assert getattr(result_type, name) == getattr(node_utils, name), (
            f"{result_type.__name__}.{name} disagrees with node_utils.{name}"
        )


@pytest.mark.parametrize(("result_type", "produced"), list(PRODUCED.items()))
def test_every_reason_a_server_produces_is_declared(result_type, produced):
    assert produced <= _declared(result_type, "TERMINATION_")


def test_recorder_and_policy_do_not_declare_an_outcome():
    # `outcome` answers "did the robot do the task". A recording has no reward
    # concept and a policy run has no task verdict, so offering the field there
    # would invite a caller to read a verdict nobody ever set.
    assert _declared(RecordEpisode.Result, "OUTCOME_") == set()
    assert _declared(RunPolicy.Result, "OUTCOME_") == set()
    assert not hasattr(RecordEpisode.Result(), "outcome")
    assert not hasattr(RunPolicy.Result(), "outcome")


def test_no_feedback_carries_a_status_field():
    """`status` was a hardcoded constant at all three publish sites, advertising
    states ("starting", "stopping") that were never emitted. Feedback arriving
    at all is the "running" signal; GoalStatus and termination_reason cover the
    end."""
    for action in (RecordEpisode, RunPolicy):
        assert not hasattr(action.Feedback(), "status")


def test_no_result_carries_a_success_flag():
    """No result carries a `success` flag, which would conflate mechanics,
    health, and task outcome. GoalStatus is the mechanics, termination_reason
    the detail, and outcome the verdict."""
    for result_type in PRODUCED:
        assert not hasattr(result_type(), "success")
