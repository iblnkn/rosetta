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
