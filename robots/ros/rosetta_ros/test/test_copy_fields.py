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

"""copy_fields carries every field between message types that share them."""

from rosetta_interfaces.action import RecordEpisode
from rosetta_interfaces.msg import RecordEpisodeResult, RecordEpisodeStatus
from rosetta_ros.node_utils import copy_fields


def test_status_becomes_feedback():
    status = RecordEpisodeStatus(elapsed_s=2.5, messages_written=7)
    feedback = copy_fields(status, RecordEpisode.Feedback())
    assert (feedback.elapsed_s, feedback.messages_written) == (2.5, 7)


def test_action_result_becomes_result_message():
    result = RecordEpisode.Result(termination_reason="timeout", message="", bag_path="/b", messages_written=3)
    msg = copy_fields(result, RecordEpisodeResult())
    assert msg == RecordEpisodeResult(termination_reason="timeout", message="", bag_path="/b", messages_written=3)
