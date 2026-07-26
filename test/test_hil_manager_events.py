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

"""HIL manager teleop-event edge detection and episode exclusion.

Regressions pinned:
- level-triggered event handling: every Joy message reports every button's
  level, so the unpressed `failure` button's release branch cleared the
  reward override that a `success` press had just set (the override never
  stuck, and the classifier resumed overwriting the reward), and the
  unpressed `is_intervention` button forced the mux back to policy on every
  message, making the set_intervention service unusable while a joystick
  publishes;
- ManageEpisode goal accept raced (`_policy_goal_handle` set far downstream).
"""

import threading
import time
from types import SimpleNamespace

import pytest
from rclpy.action import GoalResponse
from rclpy.parameter import Parameter

from rosetta.robots.ros2.nodes.hil_manager_node import HilManagerNode
from rosetta.robots.ros2.nodes.node_utils import (
    OUTCOME_FAILURE,
    OUTCOME_SUCCESS,
    OUTCOME_UNLABELED,
    TERMINATION_CANCELLED,
    TERMINATION_ERROR,
    TERMINATION_NODE_DEACTIVATED,
    TERMINATION_REWARD_THRESHOLD,
    TERMINATION_STOPPED,
)

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


@pytest.fixture
def node(rclpy_ctx):
    n = HilManagerNode()
    yield n
    n.destroy_node()


def test_success_override_survives_unpressed_failure_button(node):
    # The regression: success pressed, failure unpressed in the SAME message.
    node._on_teleop_events(_joy(success=1), EVENTS_SPEC)
    assert node._human_reward_override is True
    assert node._current_reward == node.get_parameter("human_reward_positive").value


def test_override_latches_after_release(node):
    node._on_teleop_events(_joy(success=1), EVENTS_SPEC)
    node._on_teleop_events(_joy(), EVENTS_SPEC)  # release
    assert node._human_reward_override is True  # latched
    # Classifier output is still blocked while the override holds.
    msg = SimpleNamespace(data=0.25)
    node._on_reward_classifier_output(msg, "/reward")
    assert node._current_reward == node.get_parameter("human_reward_positive").value


def test_held_button_fires_once(node):
    published = []
    node._publish_human_reward = published.append
    for _ in range(3):
        node._on_teleop_events(_joy(success=1), EVENTS_SPEC)
    assert len(published) == 1  # press edge only, not per message


def test_failure_press_overrides_with_negative_reward(node):
    node._on_teleop_events(_joy(success=1), EVENTS_SPEC)
    node._on_teleop_events(_joy(failure=1), EVENTS_SPEC)
    assert node._current_reward == node.get_parameter("human_reward_negative").value
    assert node._human_reward_override is True


def test_intervention_edges_toggle_mux(node):
    assert node._control_source == "policy"
    node._on_teleop_events(_joy(intervention=1), EVENTS_SPEC)
    assert node._control_source == "teleop"
    node._on_teleop_events(_joy(intervention=1), EVENTS_SPEC)  # held: no change
    assert node._control_source == "teleop"
    node._on_teleop_events(_joy(), EVENTS_SPEC)  # release edge
    assert node._control_source == "policy"


def test_service_intervention_not_fought_by_unpressed_joystick(node):
    # set_intervention(True) with the joystick button unpressed: subsequent
    # Joy messages (no edge) must not force the mux back to policy.
    node._on_teleop_events(_joy(), EVENTS_SPEC)  # establish unpressed state
    with node._mux_lock:
        node._control_source = "teleop"  # as set by the service
    node._on_teleop_events(_joy(), EVENTS_SPEC)
    node._on_teleop_events(_joy(), EVENTS_SPEC)
    assert node._control_source == "teleop"


def test_end_success_stops_episode_with_positive_reward(node):
    node._stop_event = threading.Event()  # the running episode's claim-time event
    node._on_teleop_events(_joy(end_success=1), EVENTS_SPEC)
    assert node._stop_event.is_set()
    assert node._human_reward_override is True
    assert node._current_reward == node.get_parameter("human_reward_positive").value


def test_end_failure_stops_episode_with_negative_reward(node):
    node._stop_event = threading.Event()  # the running episode's claim-time event
    node._on_teleop_events(_joy(end_failure=1), EVENTS_SPEC)
    assert node._stop_event.is_set()
    assert node._human_reward_override is True
    assert node._current_reward == node.get_parameter("human_reward_negative").value


def test_end_failure_reward_distinct_from_neutral_baseline(node):
    # The whole point of a distinct sentinel: an explicit failure must not be
    # numerically indistinguishable from "nobody labeled this episode".
    assert node.get_parameter("human_reward_negative").value != 0.0


def test_episode_goal_accept_is_mutually_exclusive(node):
    node._accepting_work = True
    assert node._on_goal(None) == GoalResponse.ACCEPT
    assert node._on_goal(None) == GoalResponse.REJECT
    node._busy = False
    assert node._on_goal(None) == GoalResponse.ACCEPT
    node._busy = False


def test_start_episode_button_starts_when_idle(node, monkeypatch):
    started = []
    monkeypatch.setattr(node, "_run_episode_detached", lambda *a: started.append(a))
    node._accepting_work = True

    node._on_teleop_events(_joy(start_episode=1), EVENTS_SPEC)

    # Runs on a background thread; give it a moment to invoke the stub.
    for _ in range(50):
        if started:
            break
        time.sleep(0.01)
    # Empty prompt: the button carries no text, and _run_episode resolves it
    # from default_prompt like every other empty-prompt path.
    assert started == [("", 0.0, 0.0)]
    node._busy = False


def test_start_episode_button_noop_when_already_busy(node, monkeypatch):
    started = []
    monkeypatch.setattr(node, "_run_episode_detached", lambda *a: started.append(a))
    node._accepting_work = True
    node._busy = True  # simulate an episode already running

    node._on_teleop_events(_joy(start_episode=1), EVENTS_SPEC)

    assert started == []
    node._busy = False


def test_start_episode_button_noop_when_node_inactive(node, monkeypatch):
    started = []
    monkeypatch.setattr(node, "_run_episode_detached", lambda *a: started.append(a))
    node._accepting_work = False

    node._on_teleop_events(_joy(start_episode=1), EVENTS_SPEC)

    assert started == []
    assert not node.busy


def test_manage_policy_lifecycle_false_skips_policy_send_and_cancel(rclpy_ctx, monkeypatch):
    node = HilManagerNode(parameter_overrides=[Parameter("manage_policy_lifecycle", value=False)])
    try:
        calls = []
        monkeypatch.setattr(
            node,
            "_send_child_goal",
            lambda client, goal, name, **k: calls.append(("send", name)) or SimpleNamespace(accepted=True),
        )
        monkeypatch.setattr(node, "_feedback_loop", lambda *a, **k: "human_stop")
        monkeypatch.setattr(node, "_cancel_child", lambda handle, name, **k: calls.append(("cancel", name)))
        monkeypatch.setattr(node, "_stop_recorder", lambda: None)

        node._run_episode("t", 0.0, 0.0, threading.Event())

        names_by_action = {action: [n for a, n in calls if a == action] for action in ("send", "cancel")}
        assert "Robot policy" not in names_by_action["send"]
        assert "Robot policy" not in names_by_action["cancel"]
        assert "Episode recorder" in names_by_action["send"]  # unaffected: enable_recording defaults true
    finally:
        node.destroy_node()


def test_manage_policy_lifecycle_true_sends_and_cancels_policy_by_default(node, monkeypatch):
    calls = []
    monkeypatch.setattr(
        node,
        "_send_child_goal",
        lambda client, goal, name, **k: calls.append(("send", name)) or SimpleNamespace(accepted=True),
    )
    monkeypatch.setattr(node, "_feedback_loop", lambda *a, **k: "human_stop")
    monkeypatch.setattr(node, "_cancel_child", lambda handle, name, **k: calls.append(("cancel", name)))
    monkeypatch.setattr(node, "_stop_recorder", lambda: None)

    node._run_episode("t", 0.0, 0.0, threading.Event())

    names_by_action = {action: [n for a, n in calls if a == action] for action in ("send", "cancel")}
    assert "Robot policy" in names_by_action["send"]
    assert "Robot policy" in names_by_action["cancel"]


# -------------------- Termination reason: reported, not guessed --------------------


def _claimed(node):
    """Claim the work slot the way an accepted goal does, and hand back its event."""
    node._accepting_work = True
    assert node._try_claim_work() is None
    return node._stop_event


@pytest.mark.parametrize(
    "reason",
    [TERMINATION_CANCELLED, TERMINATION_NODE_DEACTIVATED, TERMINATION_STOPPED],
)
def test_feedback_loop_reports_the_recorded_stop_reason(node, reason):
    stop_event = _claimed(node)
    node._signal_stop(reason)
    assert node._feedback_loop(None, 0.0, 0.0, stop_event) == reason


def test_deactivate_is_not_reported_as_a_human_stop(node):
    """The regression: the loop used to derive its reason from
    is_cancel_requested, so anything that was not an action cancel -- a
    deactivate included -- came back as `human_stop`. An episode the node took
    away is not an episode a human ended."""
    stop_event = _claimed(node)
    node._stop_and_secure(wait_timeout=0.1)
    assert node._feedback_loop(None, 0.0, 0.0, stop_event) == TERMINATION_NODE_DEACTIVATED


def test_feedback_loop_reports_its_own_timeout(node):
    stop_event = _claimed(node)
    stop_event.clear()
    assert node._feedback_loop(None, 0.01, 0.0, stop_event) == "timeout"
    assert node.stop_reason is None  # loop-decided, nobody signalled


# -------------------- Outcome: did the robot do the task? --------------------


def test_episode_starts_unlabeled(node):
    assert node._episode_outcome == OUTCOME_UNLABELED


@pytest.mark.parametrize(
    ("joy_kwargs", "expected"),
    [
        ({"success": 1}, OUTCOME_SUCCESS),
        ({"failure": 1}, OUTCOME_FAILURE),
        ({"end_success": 1}, OUTCOME_SUCCESS),
        ({"end_failure": 1}, OUTCOME_FAILURE),
    ],
)
def test_buttons_label_the_episode(node, joy_kwargs, expected):
    node._stop_event = threading.Event()  # the running episode's claim-time event
    node._on_teleop_events(_joy(**joy_kwargs), EVENTS_SPEC)
    assert node._episode_outcome == expected


def test_reward_override_service_labels_like_the_button(node):
    node._on_set_reward_override(SimpleNamespace(data=False), SimpleNamespace())
    assert node._episode_outcome == OUTCOME_FAILURE
    node._on_set_reward_override(SimpleNamespace(data=True), SimpleNamespace())
    assert node._episode_outcome == OUTCOME_SUCCESS


def test_clearing_the_override_retracts_the_label(node):
    node._on_teleop_events(_joy(success=1), EVENTS_SPEC)
    assert node._episode_outcome == OUTCOME_SUCCESS
    node._on_clear_reward_override(None, SimpleNamespace())
    # Not "failure" -- retracting a claim is not making the opposite one.
    assert node._episode_outcome == OUTCOME_UNLABELED


def test_end_episode_service_labels_and_stops(node):
    _claimed(node)
    resp = node._on_end_episode(SimpleNamespace(data=False), SimpleNamespace())
    assert resp.success is True
    assert node._episode_outcome == OUTCOME_FAILURE
    # A labelled, deliberate end -- not a cancel.
    assert node.stop_reason == TERMINATION_STOPPED


def test_end_episode_service_refuses_when_idle(node):
    resp = node._on_end_episode(SimpleNamespace(data=True), SimpleNamespace())
    assert resp.success is False
    assert "No active episode" in resp.message


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


def test_reward_threshold_promotes_an_unlabeled_episode_to_success(node, monkeypatch):
    # Crossing the threshold the caller specified IS the success criterion, so
    # nobody has to also press a button to say so.
    fields = _run_episode_with(node, monkeypatch, TERMINATION_REWARD_THRESHOLD, reward_threshold=1.0)
    assert fields["termination_reason"] == TERMINATION_REWARD_THRESHOLD
    assert fields["outcome"] == OUTCOME_SUCCESS


def test_a_human_label_reaches_the_result(node, monkeypatch):
    fields = _run_episode_with(
        node,
        monkeypatch,
        TERMINATION_STOPPED,
        during=lambda: node._on_teleop_events(_joy(end_failure=1), EVENTS_SPEC),
    )
    assert fields["termination_reason"] == TERMINATION_STOPPED
    assert fields["outcome"] == OUTCOME_FAILURE
    assert fields["final_reward"] == node.get_parameter("human_reward_negative").value


def test_an_unlabeled_timeout_stays_unlabeled(node, monkeypatch):
    # Distinct from a classifier honestly reporting 0.0 -- which is the whole
    # reason `outcome` exists rather than being inferred from final_reward.
    fields = _run_episode_with(node, monkeypatch, "timeout")
    assert fields["outcome"] == OUTCOME_UNLABELED


def test_a_child_that_fails_to_start_reports_an_error(node, monkeypatch):
    """The fields dict defaults to `error` so no early return can forget to say
    the episode never really began."""
    monkeypatch.setattr(node, "_send_child_goal", lambda *a, **k: None)
    monkeypatch.setattr(node, "_cancel_child", lambda *a, **k: None)
    fields = node._run_episode("t", 0.0, 0.0, threading.Event())
    assert fields["termination_reason"] == TERMINATION_ERROR
    assert "Failed to start" in fields["message"]


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
