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

"""Contract event names -> lerobot's TeleopEvents.

Contract event names are open, so this plugin owns its vocabulary. Reading each
event's value off the device (selectors, malformed fields) is the robot
interface's job; this maps the values it reports.
"""

import pytest
from lerobot.teleoperators.utils import TeleopEvents

from lerobot_teleoperator_rosetta.rosetta_teleop import DEFAULT_EVENTS, EVENT_NAME_TO_ENUMS, teleop_events

SUCCESS, FAILURE, TERMINATE = TeleopEvents.SUCCESS, TeleopEvents.FAILURE, TeleopEvents.TERMINATE_EPISODE


def test_every_lerobot_event_but_rerecord_is_reachable_from_a_contract_name():
    reachable = {e for enums in EVENT_NAME_TO_ENUMS.values() for e in enums}
    assert reachable == set(TeleopEvents) - {TeleopEvents.RERECORD_EPISODE}


def _asserted(event_values):
    """The lerobot events that differ from their defaults."""
    return {event: value for event, value in teleop_events(event_values).items() if value != DEFAULT_EVENTS[event]}


@pytest.mark.parametrize(
    ("event_values", "asserted"),
    [
        ({"end_success": True}, {SUCCESS: True, TERMINATE: True}),
        ({"end_failure": True}, {FAILURE: True, TERMINATE: True}),
        # success and end_success both assert SUCCESS: either one pressed holds it.
        ({"success": True, "end_success": False}, {SUCCESS: True}),
        ({"success": False, "end_success": True}, {SUCCESS: True, TERMINATE: True}),
        ({"success": False}, {}),  # released
        ({"start_episode": True}, {}),  # no lerobot mapping: ignored
        ({}, {}),  # before the first event message
    ],
    ids=["end_success", "end_failure", "or_first", "or_second", "released", "unmapped", "no_events"],
)
def test_contract_event_values_map_onto_lerobot_events(event_values, asserted):
    assert _asserted(event_values) == asserted
