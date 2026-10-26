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

"""ManageEpisode declares every reason and outcome the HIL manager produces.

The shared reasons come from Rosetta's node toolkit; ManageEpisode must spell
them the same way, and add its own on top.
"""

from rosetta_interfaces.action import ManageEpisode
from rosetta_interfaces.msg import ManageEpisodeResult, ManageEpisodeStatus
from rosetta_ros import node_utils
from rosetta_ros_hil import hil_episode

PRODUCED = {
    node_utils.TERMINATION_STOPPED,
    node_utils.TERMINATION_TIMEOUT,
    node_utils.TERMINATION_CANCELLED,
    node_utils.TERMINATION_NODE_DEACTIVATED,
    node_utils.TERMINATION_ERROR,
    hil_episode.TERMINATION_TIMEOUT,
    hil_episode.TERMINATION_REWARD_THRESHOLD,
}


def _declared(prefix):
    return {getattr(ManageEpisode.Result, n) for n in dir(ManageEpisode.Result) if n.startswith(prefix)}


def test_every_reason_the_manager_produces_is_declared():
    assert PRODUCED <= _declared("TERMINATION_")


def test_shared_reasons_match_rosetta():
    for name in dir(ManageEpisode.Result):
        if name.startswith("TERMINATION_") and hasattr(node_utils, name):
            assert getattr(ManageEpisode.Result, name) == getattr(node_utils, name), name


def test_outcome_vocabulary():
    assert _declared("OUTCOME_") == {
        hil_episode.OUTCOME_SUCCESS,
        hil_episode.OUTCOME_FAILURE,
        hil_episode.OUTCOME_UNLABELED,
    }


def test_topic_messages_match_the_action():
    """~/status and ~/result are what clients without the action see; they must not drift."""
    assert ManageEpisodeStatus.get_fields_and_field_types() == ManageEpisode.Feedback.get_fields_and_field_types()
    assert ManageEpisodeResult.get_fields_and_field_types() == ManageEpisode.Result.get_fields_and_field_types()


def test_episode_rules_spell_the_shared_timeout_like_rosetta_ros():
    assert hil_episode.TERMINATION_TIMEOUT == node_utils.TERMINATION_TIMEOUT
