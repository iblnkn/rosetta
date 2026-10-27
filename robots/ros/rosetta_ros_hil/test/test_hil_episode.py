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

"""HIL episode rules: labels latch, edges fire once, episodes end on time or reward."""

import pytest

from rosetta_ros.node_utils import TERMINATION_STOPPED, TERMINATION_TIMEOUT
from rosetta_ros_hil.hil_episode import (
    OUTCOME_FAILURE,
    OUTCOME_SUCCESS,
    OUTCOME_UNLABELED,
    TERMINATION_REWARD_THRESHOLD,
    Control,
    EdgeDetector,
    EpisodeState,
    ignored_events,
    resolve_outcome,
    termination_reason,
)


@pytest.mark.parametrize("control", [Control.POLICY, Control.TELEOP])
def test_new_episode_starts_with_the_given_control_and_no_label(control):
    state = EpisodeState()
    state.set_control(Control.TELEOP if control == Control.POLICY else Control.POLICY)
    state.label(True, 1.0)
    state.reset(control)
    assert state.snapshot() == (control, 0.0, OUTCOME_UNLABELED)


@pytest.mark.parametrize(("success", "outcome"), [(True, OUTCOME_SUCCESS), (False, OUTCOME_FAILURE)])
def test_label_latches_reward_and_outcome(success, outcome):
    state = EpisodeState()
    state.label(success, 0.5)
    assert (state.reward, state.outcome, state.labeled) == (0.5, outcome, True)


def test_label_blocks_classifier_rewards_until_cleared():
    state = EpisodeState()
    state.label(True, 1.0)
    assert state.accept_classifier_reward(0.2) is False
    assert state.reward == 1.0
    state.clear_label()
    assert state.accept_classifier_reward(0.2) is True
    assert (state.reward, state.labeled) == (0.2, False)


@pytest.mark.parametrize(
    ("sequence", "edges"),
    [
        ([{"a": True}], [[("a", True)]]),
        ([{"a": True}, {"a": True}], [[("a", True)], []]),  # held: fires once
        ([{"a": True}, {"a": False}], [[("a", True)], [("a", False)]]),
        ([{"a": False}], [[]]),  # never pressed: no release edge
        ([{"a": True, "b": True}, {"a": False, "b": True}], [[("a", True), ("b", True)], [("a", False)]]),
    ],
)
def test_edge_detector_reports_only_changes(sequence, edges):
    detector = EdgeDetector()
    assert [detector.update(levels) for levels in sequence] == edges


@pytest.mark.parametrize(
    ("elapsed", "max_duration", "reward", "threshold", "reason"),
    [
        (5.0, 5.0, 0.0, 0.0, TERMINATION_TIMEOUT),
        (4.9, 5.0, 0.0, 0.0, None),
        (100.0, 0.0, 0.0, 0.0, None),  # no time limit
        (1.0, 0.0, 0.8, 0.8, TERMINATION_REWARD_THRESHOLD),
        (1.0, 0.0, 0.9, 0.0, None),  # no threshold
        (5.0, 5.0, 1.0, 0.5, TERMINATION_TIMEOUT),  # timeout checked first
    ],
)
def test_termination_reason(elapsed, max_duration, reward, threshold, reason):
    assert termination_reason(elapsed, max_duration, reward, threshold) == reason


@pytest.mark.parametrize(
    ("outcome", "reason", "resolved"),
    [
        (OUTCOME_UNLABELED, TERMINATION_REWARD_THRESHOLD, OUTCOME_SUCCESS),
        (OUTCOME_FAILURE, TERMINATION_REWARD_THRESHOLD, OUTCOME_FAILURE),  # the human's label wins
        (OUTCOME_UNLABELED, TERMINATION_STOPPED, OUTCOME_UNLABELED),
        (OUTCOME_UNLABELED, TERMINATION_TIMEOUT, OUTCOME_UNLABELED),
    ],
)
def test_resolve_outcome(outcome, reason, resolved):
    assert resolve_outcome(outcome, reason) == resolved


@pytest.mark.parametrize(
    ("names", "ignored"),
    [
        (["success", "end_failure"], []),
        (["success", "rerecord_episode"], ["rerecord_episode"]),
        (["succes", "zz", "start_episode"], ["succes", "zz"]),
    ],
)
def test_ignored_events_are_the_names_outside_hil_vocabulary(names, ignored):
    assert ignored_events(names) == ignored
