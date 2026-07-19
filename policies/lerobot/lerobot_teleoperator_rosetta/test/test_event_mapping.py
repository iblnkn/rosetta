"""Pin the contract's teleop event vocabulary to the lerobot event mapping.

The mapping once carried names the contract rejects at load
(``terminate_episode``, ``rerecord_episode``) while three real contract
events went unmapped and were silently dropped. These tests fail when either
vocabulary changes without the other.
"""

from lerobot.teleoperators.utils import TeleopEvents
from rosetta.contract.model import TELEOP_EVENT_NAMES

from lerobot_teleoperator_rosetta.rosetta_teleop import DEFAULT_EVENTS, EVENT_NAME_TO_ENUMS

# The one contract event with no lerobot counterpart: lerobot's record loop
# starts episodes itself, so only the rosetta-native path consumes it.
LEROBOT_UNMAPPABLE = {"start_episode"}


def test_every_mapping_key_is_a_contract_event():
    assert set(EVENT_NAME_TO_ENUMS) <= TELEOP_EVENT_NAMES


def test_every_contract_event_maps_or_is_known_unmappable():
    assert set(EVENT_NAME_TO_ENUMS) == TELEOP_EVENT_NAMES - LEROBOT_UNMAPPABLE


def test_end_success_composes_reward_and_termination():
    assert set(EVENT_NAME_TO_ENUMS["end_success"]) == {TeleopEvents.SUCCESS, TeleopEvents.TERMINATE_EPISODE}


def test_end_failure_composes_reward_and_termination():
    assert set(EVENT_NAME_TO_ENUMS["end_failure"]) == {TeleopEvents.FAILURE, TeleopEvents.TERMINATE_EPISODE}


def test_default_events_cover_full_lerobot_enum():
    assert set(DEFAULT_EVENTS) == set(TeleopEvents)


def test_default_events_start_false():
    assert not any(DEFAULT_EVENTS.values())
