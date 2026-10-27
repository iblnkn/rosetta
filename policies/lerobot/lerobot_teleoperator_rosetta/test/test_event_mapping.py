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

"""Contract event names -> lerobot's TeleopEvents, as the teleop node applies them per message.

Contract event names are open, so this plugin owns its vocabulary.
"""

import threading
from types import SimpleNamespace

import pytest
from lerobot.teleoperators.utils import TeleopEvents

from lerobot_teleoperator_rosetta.rosetta_teleop import (
    DEFAULT_EVENTS,
    EVENT_NAME_TO_ENUMS,
    _RosettaTeleopLifecycleNode,
)

SUCCESS, FAILURE, TERMINATE = TeleopEvents.SUCCESS, TeleopEvents.FAILURE, TeleopEvents.TERMINATE_EPISODE


def test_every_lerobot_event_but_rerecord_is_reachable_from_a_contract_name():
    reachable = {e for enums in EVENT_NAME_TO_ENUMS.values() for e in enums}
    assert reachable == set(TeleopEvents) - {TeleopEvents.RERECORD_EPISODE}


def _apply(select, buttons, before=None):
    """Run the node's events callback on a fake node holding ``before``; return the changed events."""
    state = dict(DEFAULT_EVENTS) | (before or {})
    node = SimpleNamespace(_events_lock=threading.Lock(), _events_state=state)
    _RosettaTeleopLifecycleNode._on_events(node, SimpleNamespace(buttons=buttons), SimpleNamespace(select=select))
    return {event: value for event, value in node._events_state.items() if value != DEFAULT_EVENTS[event]}


@pytest.mark.parametrize(
    ("select", "buttons", "before", "after"),
    [
        ({"end_success": "buttons.0"}, [1], None, {SUCCESS: True, TERMINATE: True}),
        ({"end_failure": "buttons.0"}, [1], None, {FAILURE: True, TERMINATE: True}),
        # success and end_success both assert SUCCESS: either one pressed holds it.
        ({"success": "buttons.0", "end_success": "buttons.1"}, [1, 0], None, {SUCCESS: True}),
        ({"success": "buttons.0", "end_success": "buttons.1"}, [0, 1], None, {SUCCESS: True, TERMINATE: True}),
        ({"success": "buttons.0"}, [0], {SUCCESS: True}, {}),  # release clears
        ({"start_episode": "buttons.0"}, [1], None, {}),  # no lerobot mapping: skipped
        ({"success": "buttons.9"}, [0], {SUCCESS: True}, {SUCCESS: True}),  # malformed: keeps last value
    ],
    ids=["end_success", "end_failure", "or_first", "or_second", "release", "unmapped", "malformed"],
)
def test_events_message_updates_the_lerobot_events(select, buttons, before, after):
    assert _apply(select, buttons, before) == after
