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

"""The ~/status and ~/result messages carry exactly the action's feedback and result fields.

They are what clients that can't use actions see instead, so they must not
drift from the action definitions.
"""

import pytest

from rosetta_interfaces.action import RecordEpisode, RunPolicy
from rosetta_interfaces.msg import RecordEpisodeResult, RecordEpisodeStatus, RunPolicyResult, RunPolicyStatus


@pytest.mark.parametrize(
    ("message", "action_part"),
    [
        (RecordEpisodeStatus, RecordEpisode.Feedback),
        (RecordEpisodeResult, RecordEpisode.Result),
        (RunPolicyStatus, RunPolicy.Feedback),
        (RunPolicyResult, RunPolicy.Result),
    ],
)
def test_message_matches_the_action(message, action_part):
    assert message.get_fields_and_field_types() == action_part.get_fields_and_field_types()
