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

"""Pin the contract event names this plugin maps onto lerobot's TeleopEvents.

Contract event names are open, so this plugin owns its vocabulary. The
mapping once left three events unmapped and silently dropped; these tests
fail if a lerobot event becomes unreachable or the composed events change.
"""

from lerobot.teleoperators.utils import TeleopEvents

from lerobot_teleoperator_rosetta.rosetta_teleop import DEFAULT_EVENTS, EVENT_NAME_TO_ENUMS


def test_every_lerobot_event_but_rerecord_is_reachable_from_a_contract_name():
    reachable = {e for enums in EVENT_NAME_TO_ENUMS.values() for e in enums}
    assert reachable == set(TeleopEvents) - {TeleopEvents.RERECORD_EPISODE}


def test_end_success_composes_reward_and_termination():
    assert set(EVENT_NAME_TO_ENUMS["end_success"]) == {TeleopEvents.SUCCESS, TeleopEvents.TERMINATE_EPISODE}


def test_end_failure_composes_reward_and_termination():
    assert set(EVENT_NAME_TO_ENUMS["end_failure"]) == {TeleopEvents.FAILURE, TeleopEvents.TERMINATE_EPISODE}


def test_default_events_cover_full_lerobot_enum():
    assert set(DEFAULT_EVENTS) == set(TeleopEvents)


def test_default_events_start_false():
    assert not any(DEFAULT_EVENTS.values())
