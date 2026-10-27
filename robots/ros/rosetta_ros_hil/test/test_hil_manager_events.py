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

"""HIL manager: teleop events, labels, services, and the episode it runs and reports.

Events must be edge-triggered. Every Joy message reports every button's level,
so acting on levels lets an unpressed `failure` button clear the reward
override a `success` press just set, and an unpressed `is_intervention` button
force the mux back to policy on every message (making set_intervention
unusable while a joystick publishes).
"""

import threading
from types import SimpleNamespace

import pytest
from rclpy.parameter import Parameter

from rosetta_interfaces.msg import ManageEpisodeResult
from rosetta_ros.node_utils import (
    TERMINATION_CANCELLED,
    TERMINATION_ERROR,
    TERMINATION_NODE_DEACTIVATED,
    TERMINATION_STOPPED,
    wait_until,
)
from rosetta_ros_hil.hil_episode import (
    OUTCOME_FAILURE,
    OUTCOME_SUCCESS,
    OUTCOME_UNLABELED,
    TERMINATION_REWARD_THRESHOLD,
)
from rosetta_ros_hil.hil_manager_node import HilManagerNode

# events_spec.select maps event name -> Joy selector. Button layout:
# 0=is_intervention, 1=end_success, 2=end_failure, 3=success, 4=failure, 5=start_episode.
EVENTS_SPEC = SimpleNamespace(
    select={
        "is_intervention": "buttons.0",
        "end_success": "buttons.1",
        "end_failure": "buttons.2",
        "success": "buttons.3",
        "failure": "buttons.4",
        "start_episode": "buttons.5",
    }
)


def _joy(intervention=0, end_success=0, end_failure=0, success=0, failure=0, start_episode=0):
    return SimpleNamespace(buttons=[intervention, end_success, end_failure, success, failure, start_episode])


class FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(msg)


def _with_fake_topics(node):
    """Install the ~/status, ~/result and hil_intervention publishers setup() would create, without a contract."""
    node._status_pub, node._result_pub = FakePublisher(), FakePublisher()
    node._intervention_pub = FakePublisher()
    return node


def _announced(node):
    return [msg.data for msg in node._intervention_pub.published]


@pytest.fixture
def node(rclpy_ctx):
    n = _with_fake_topics(HilManagerNode())
    yield n
    n.destroy_node()


@pytest.fixture
def episode(node):
    """An episode in progress: the work slot claimed as an accepted goal claims it."""
    _claimed(node)
    yield
    node._busy = False


def test_events_channel_need_not_be_joy(node, episode):
    """The events channel type comes from the contract; a message without `buttons` must still dispatch."""
    spec = SimpleNamespace(select={"success": "data"})
    node._on_teleop_events(SimpleNamespace(data=True), spec)
    assert node._episode.labeled is True


def test_events_outside_hil_vocabulary_are_never_read(node, episode, monkeypatch):
    """Another consumer's event, even with a broken selector, neither dispatches nor warns."""
    logged = []
    logger = SimpleNamespace(warning=logged.append, info=lambda *_: None)
    monkeypatch.setattr(node, "get_logger", lambda: logger)
    spec = SimpleNamespace(select={"rerecord_episode": "buttons.99", "success": "buttons.3"})
    node._on_teleop_events(_joy(success=1), spec)
    assert node._episode.labeled is True
    assert logged == []


@pytest.mark.parametrize(
    ("select", "warnings"),
    [({"success": "buttons.99"}, 1), ({"success": "buttons.98", "failure": "buttons.99"}, 2)],
)
def test_a_bad_selector_is_warned_about_once(node, monkeypatch, select, warnings):
    logged = []
    logger = SimpleNamespace(warning=logged.append, info=lambda *_: None)
    monkeypatch.setattr(node, "get_logger", lambda: logger)
    for _ in range(3):
        node._on_teleop_events(_joy(), SimpleNamespace(select=select))
    assert len(logged) == warnings


def test_override_latches_after_release(node, episode):
    node._on_teleop_events(_joy(success=1), EVENTS_SPEC)
    node._on_teleop_events(_joy(), EVENTS_SPEC)  # release
    assert node._episode.labeled is True  # latched
    # Classifier output is still blocked while the override holds.
    msg = SimpleNamespace(data=0.25)
    node._on_reward_classifier_output(msg, "/reward")
    assert node._episode.reward == node.get_parameter("human_reward_positive").value


def test_failure_press_overrides_with_negative_reward(node, episode):
    node._on_teleop_events(_joy(success=1), EVENTS_SPEC)
    node._on_teleop_events(_joy(failure=1), EVENTS_SPEC)
    assert node._episode.reward == node.get_parameter("human_reward_negative").value
    assert node._episode.labeled is True


def test_intervention_edges_toggle_mux(node):
    assert node._episode.control == "policy"
    node._on_teleop_events(_joy(intervention=1), EVENTS_SPEC)
    assert node._episode.control == "teleop"
    node._on_teleop_events(_joy(intervention=1), EVENTS_SPEC)  # held: no change
    assert node._episode.control == "teleop"
    node._on_teleop_events(_joy(), EVENTS_SPEC)  # release edge
    assert node._episode.control == "policy"


def test_every_control_change_is_announced_even_between_episodes(node):
    """A leader arm following hil_intervention must hear a takeover made with no episode running."""
    assert not node.busy
    node._on_teleop_events(_joy(intervention=1), EVENTS_SPEC)
    node._on_teleop_events(_joy(), EVENTS_SPEC)
    node._on_set_intervention(SimpleNamespace(data=True), SimpleNamespace())
    assert _announced(node) == [1, 0, 1]


def test_service_intervention_not_fought_by_unpressed_joystick(node):
    # set_intervention(True) with the joystick button unpressed: subsequent
    # Joy messages (no edge) must not force the mux back to policy.
    node._on_teleop_events(_joy(), EVENTS_SPEC)  # establish unpressed state
    node._episode.set_control("teleop")  # as set by the service
    node._on_teleop_events(_joy(), EVENTS_SPEC)
    node._on_teleop_events(_joy(), EVENTS_SPEC)
    assert node._episode.control == "teleop"


def test_start_episode_button_starts_when_idle(node, monkeypatch):
    started = []
    fields = ManageEpisodeResult().get_fields_and_field_types()
    monkeypatch.setattr(node, "_run_episode", lambda *a: started.append(a[:3]) or dict.fromkeys(fields, ""))
    node._accepting_work = True

    node._on_teleop_events(_joy(start_episode=1), EVENTS_SPEC)

    # The episode runs on its own thread and releases the work slot when done.
    assert wait_until(lambda: not node.busy, timeout=5.0, poll=0.01)
    # Empty prompt: the button carries no text, and _run_episode resolves it
    # from default_prompt like every other empty-prompt path.
    assert started == [("", 0.0, 0.0)]


@pytest.mark.parametrize(("accepting", "busy"), [(True, True), (False, False)], ids=["busy", "inactive"])
def test_start_episode_button_is_ignored_when_work_cannot_be_claimed(node, monkeypatch, accepting, busy):
    started = []
    monkeypatch.setattr(node, "_run_episode_detached", lambda *a: started.append(a))
    node._accepting_work, node._busy = accepting, busy

    node._on_teleop_events(_joy(start_episode=1), EVENTS_SPEC)

    assert started == []
    assert node.busy is busy
    node._busy = False


@pytest.mark.parametrize(
    ("overrides", "drives_policy"),
    [([], True), ([Parameter("manage_policy_lifecycle", value=False)], False)],
    ids=["default", "manage_policy_lifecycle=false"],
)
def test_manage_policy_lifecycle_decides_whether_the_policy_is_sent_and_cancelled(
    rclpy_ctx, monkeypatch, overrides, drives_policy
):
    node = _with_fake_topics(HilManagerNode(parameter_overrides=overrides))
    try:
        calls = []
        monkeypatch.setattr(
            node,
            "_send_child_goal",
            lambda client, goal, name, **k: calls.append(("send", name)) or SimpleNamespace(accepted=True),
        )
        monkeypatch.setattr(node, "_feedback_loop", lambda *a, **k: TERMINATION_STOPPED)
        monkeypatch.setattr(node, "_cancel_child", lambda handle, name, **k: calls.append(("cancel", name)))
        monkeypatch.setattr(node, "_stop_recorder", lambda: None)

        node._run_episode("t", 0.0, 0.0, threading.Event())

        assert (("send", "Robot policy") in calls, ("cancel", "Robot policy") in calls) == (
            drives_policy,
            drives_policy,
        )
        assert ("send", "Episode recorder") in calls  # unaffected: enable_recording defaults true
    finally:
        node.destroy_node()


# -------------------- Termination reason: reported, not guessed --------------------


def _claimed(node):
    """Claim the work slot the way an accepted goal does, and hand back its event."""
    node._accepting_work = True
    assert node.try_claim_work() is None
    return node._stop_event


@pytest.mark.parametrize(
    "reason",
    [TERMINATION_CANCELLED, TERMINATION_NODE_DEACTIVATED, TERMINATION_STOPPED],
)
def test_feedback_loop_reports_the_recorded_stop_reason(node, reason):
    stop_event = _claimed(node)
    node.signal_stop(reason)
    assert node._feedback_loop(None, 0.0, 0.0, stop_event) == reason


def test_feedback_loop_reports_its_own_timeout(node):
    stop_event = _claimed(node)
    stop_event.clear()
    assert node._feedback_loop(None, 0.01, 0.0, stop_event) == "timeout"
    assert node.stop_reason is None  # loop-decided, nobody signalled


# -------------------- Outcome: did the robot do the task? --------------------


@pytest.mark.parametrize(
    ("button", "outcome", "reward_param", "stops"),
    [
        ("success", OUTCOME_SUCCESS, "human_reward_positive", False),
        ("failure", OUTCOME_FAILURE, "human_reward_negative", False),
        ("end_success", OUTCOME_SUCCESS, "human_reward_positive", True),
        ("end_failure", OUTCOME_FAILURE, "human_reward_negative", True),
    ],
)
def test_buttons_label_the_episode(node, episode, button, outcome, reward_param, stops):
    """Each press labels the episode; the end_* buttons also stop it. Every other button is unpressed."""
    node._on_teleop_events(_joy(**{button: 1}), EVENTS_SPEC)
    reward = node.get_parameter(reward_param).value
    assert (node._episode.outcome, node._episode.reward, node._stop_event.is_set()) == (outcome, reward, stops)


def test_reward_override_service_labels_like_the_button(node, episode):
    node._on_set_reward_override(SimpleNamespace(data=False), SimpleNamespace())
    assert node._episode.outcome == OUTCOME_FAILURE
    node._on_set_reward_override(SimpleNamespace(data=True), SimpleNamespace())
    assert node._episode.outcome == OUTCOME_SUCCESS


def test_clearing_the_override_retracts_the_label(node, episode):
    node._on_teleop_events(_joy(success=1), EVENTS_SPEC)
    assert node._episode.outcome == OUTCOME_SUCCESS
    node._on_clear_reward_override(None, SimpleNamespace())
    # Not "failure" -- retracting a claim is not making the opposite one.
    assert node._episode.outcome == OUTCOME_UNLABELED


def test_end_episode_service_labels_and_stops(node):
    _claimed(node)
    resp = node._on_end_episode(SimpleNamespace(data=False), SimpleNamespace())
    assert resp.success is True
    assert node._episode.outcome == OUTCOME_FAILURE
    # A labelled, deliberate end -- not a cancel.
    assert node.stop_reason == TERMINATION_STOPPED


@pytest.mark.parametrize("button", ["success", "failure", "end_success", "end_failure"])
def test_label_buttons_are_ignored_when_idle(node, button):
    published = []
    node._publish_human_reward = published.append
    node._on_teleop_events(_joy(**{button: 1}), EVENTS_SPEC)
    assert node._episode.labeled is False
    assert published == []


@pytest.mark.parametrize(
    ("service", "service_request"),
    [
        ("_on_set_reward_override", SimpleNamespace(data=True)),
        ("_on_clear_reward_override", None),
        ("_on_end_episode", SimpleNamespace(data=True)),
    ],
)
def test_label_services_refuse_when_idle(node, service, service_request):
    resp = getattr(node, service)(service_request, SimpleNamespace())
    assert (resp.success, resp.message) == (False, "No active episode")


def _run_episode_with(node, monkeypatch, reason, reward_threshold=0.0, during=None):
    """Run an episode whose feedback loop immediately ends with ``reason``.

    ``during`` runs inside the loop, standing in for whatever a human did while
    the episode was live -- it must happen there, since _run_episode resets the
    episode's label on the way in.
    """

    def feedback_loop(*_a, **_k):
        if during is not None:
            during()
        return reason

    monkeypatch.setattr(node, "_send_child_goal", lambda *a, **k: SimpleNamespace(accepted=True))
    monkeypatch.setattr(node, "_feedback_loop", feedback_loop)
    monkeypatch.setattr(node, "_cancel_child", lambda *a, **k: None)
    monkeypatch.setattr(node, "_stop_recorder", lambda: None)
    return node._run_episode("t", 0.0, reward_threshold, threading.Event())


@pytest.mark.parametrize(("held", "control"), [(True, "teleop"), (False, "policy")])
def test_an_episode_starts_with_whoever_holds_intervention(node, monkeypatch, held, control):
    """A human still holding intervention at the start keeps the arm: holding is not a new press."""
    node._on_teleop_events(_joy(intervention=1), EVENTS_SPEC)
    if not held:
        node._on_teleop_events(_joy(), EVENTS_SPEC)
    seen = []
    _run_episode_with(node, monkeypatch, TERMINATION_STOPPED, during=lambda: seen.append(node._episode.control))
    assert seen == [control]


def test_reward_threshold_promotes_an_unlabeled_episode_to_success(node, monkeypatch):
    # Crossing the threshold the caller specified IS the success criterion, so
    # nobody has to also press a button to say so.
    fields = _run_episode_with(node, monkeypatch, TERMINATION_REWARD_THRESHOLD, reward_threshold=1.0)
    assert fields["termination_reason"] == TERMINATION_REWARD_THRESHOLD
    assert fields["outcome"] == OUTCOME_SUCCESS


def test_a_human_label_reaches_the_result(node, episode, monkeypatch):
    fields = _run_episode_with(
        node,
        monkeypatch,
        TERMINATION_STOPPED,
        during=lambda: node._on_teleop_events(_joy(end_failure=1), EVENTS_SPEC),
    )
    assert fields["termination_reason"] == TERMINATION_STOPPED
    assert fields["outcome"] == OUTCOME_FAILURE
    assert fields["final_reward"] == node.get_parameter("human_reward_negative").value


def test_a_child_that_fails_to_start_reports_and_publishes_an_error(node, monkeypatch):
    """The fields dict defaults to `error` so no early return can forget to say
    the episode never really began, and ~/result still hears about it."""
    monkeypatch.setattr(node, "_send_child_goal", lambda *a, **k: None)
    monkeypatch.setattr(node, "_cancel_child", lambda *a, **k: None)
    fields = node._run_episode("t", 0.0, 0.0, threading.Event())
    assert fields["termination_reason"] == TERMINATION_ERROR
    assert "Failed to start" in fields["message"]
    (published,) = node._result_pub.published
    assert published.termination_reason == TERMINATION_ERROR


def _capture_child_prompts(node, monkeypatch, seen):
    monkeypatch.setattr(
        node,
        "_send_child_goal",
        lambda _c, goal, *a, **k: seen.append(goal.prompt) or SimpleNamespace(accepted=True),
    )
    monkeypatch.setattr(node, "_feedback_loop", lambda *a, **k: TERMINATION_STOPPED)
    monkeypatch.setattr(node, "_cancel_child", lambda *a, **k: None)
    monkeypatch.setattr(node, "_stop_recorder", lambda: None)


def test_an_empty_prompt_falls_back_to_the_node_default(node, monkeypatch):
    """A caller with no prompt to give -- a button, a bare `{}` goal, a
    dashboard with no text box -- gets the node's configured one, and it
    reaches the children."""
    node.set_parameters([Parameter("default_prompt", value="pick up the cube")])
    seen = []
    _capture_child_prompts(node, monkeypatch, seen)

    node._run_episode("", 0.0, 0.0, threading.Event())
    assert seen and all(p == "pick up the cube" for p in seen)


def test_an_explicit_prompt_beats_the_node_default(node, monkeypatch):
    node.set_parameters([Parameter("default_prompt", value="fallback")])
    seen = []
    _capture_child_prompts(node, monkeypatch, seen)

    node._run_episode("open the drawer", 0.0, 0.0, threading.Event())
    assert seen and all(p == "open the drawer" for p in seen)


# -------------------- ~/status and ~/result: visible without an action client --------------------


def test_an_episode_publishes_the_result_it_returns(node, monkeypatch):
    fields = _run_episode_with(node, monkeypatch, TERMINATION_REWARD_THRESHOLD, reward_threshold=1.0)
    (published,) = node._result_pub.published
    assert published == ManageEpisodeResult(**fields)


def test_status_is_published_without_a_goal(node):
    node.set_parameters([Parameter("feedback_rate_hz", value=200.0)])
    assert node._feedback_loop(None, 0.05, 0.0, threading.Event()) == "timeout"
    assert node._status_pub.published
    assert all(s.control_source == "policy" for s in node._status_pub.published)
