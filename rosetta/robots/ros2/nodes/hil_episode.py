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

"""Episode state and rules for the HIL manager, with no ROS node involved."""

from __future__ import annotations

import threading
from enum import StrEnum

from rosetta.robots.ros2.node_utils import TERMINATION_TIMEOUT
from rosetta_interfaces.action import ManageEpisode

# HIL's own vocabulary, declared by ManageEpisode on top of the shared reasons.
TERMINATION_REWARD_THRESHOLD = ManageEpisode.Result.TERMINATION_REWARD_THRESHOLD
OUTCOME_SUCCESS = ManageEpisode.Result.OUTCOME_SUCCESS
OUTCOME_FAILURE = ManageEpisode.Result.OUTCOME_FAILURE
OUTCOME_UNLABELED = ManageEpisode.Result.OUTCOME_UNLABELED


class Control(StrEnum):
    """Who drives the robot's command topics."""

    POLICY = "policy"
    TELEOP = "teleop"


class EpisodeState:
    """Who is in control, the current reward, and the human's verdict. Thread-safe.

    A human label (success or failure) latches both the reward and the outcome
    until it is cleared or the episode resets; while labeled, classifier
    rewards are ignored.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._control = Control.POLICY
        self._reward = 0.0
        self._outcome = OUTCOME_UNLABELED

    def reset(self) -> None:
        """Start a new episode: policy in control, zero reward, unlabeled."""
        with self._lock:
            self._control = Control.POLICY
            self._reward = 0.0
            self._outcome = OUTCOME_UNLABELED

    def set_control(self, control: Control) -> None:
        with self._lock:
            self._control = Control(control)

    def label(self, success: bool, reward: float) -> None:
        """Latch the human's verdict and the reward that goes with it."""
        with self._lock:
            self._reward = reward
            self._outcome = OUTCOME_SUCCESS if success else OUTCOME_FAILURE

    def clear_label(self) -> None:
        """Retract the verdict; classifier rewards flow again."""
        with self._lock:
            self._outcome = OUTCOME_UNLABELED

    def accept_classifier_reward(self, reward: float) -> bool:
        """Take a classifier reward unless a human label holds. Returns whether it was taken."""
        with self._lock:
            if self._outcome != OUTCOME_UNLABELED:
                return False
            self._reward = reward
            return True

    def snapshot(self) -> tuple[Control, float, str]:
        """``(control, reward, outcome)`` read together."""
        with self._lock:
            return self._control, self._reward, self._outcome

    @property
    def control(self) -> Control:
        return self.snapshot()[0]

    @property
    def reward(self) -> float:
        return self.snapshot()[1]

    @property
    def outcome(self) -> str:
        return self.snapshot()[2]

    @property
    def labeled(self) -> bool:
        return self.outcome != OUTCOME_UNLABELED


class EdgeDetector:
    """Turns per-message button levels into press and release edges. Thread-safe.

    Not reset between episodes: a button held across an episode boundary is
    not a new press.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pressed: dict[str, bool] = {}

    def update(self, levels: dict[str, bool]) -> list[tuple[str, bool]]:
        """Return ``(name, pressed)`` for every button whose level changed, in input order."""
        edges = []
        with self._lock:
            for name, pressed in levels.items():
                if pressed != self._pressed.get(name, False):
                    edges.append((name, pressed))
                self._pressed[name] = pressed
        return edges


def termination_reason(elapsed: float, max_duration: float, reward: float, reward_threshold: float) -> str | None:
    """Why the episode should end now, or None. A non-positive limit disables that check."""
    if max_duration > 0.0 and elapsed >= max_duration:
        return TERMINATION_TIMEOUT
    if reward_threshold > 0.0 and reward >= reward_threshold:
        return TERMINATION_REWARD_THRESHOLD
    return None


def resolve_outcome(outcome: str, reason: str) -> str:
    """Final outcome: reaching the reward threshold is success, unless a human said otherwise."""
    if outcome == OUTCOME_UNLABELED and reason == TERMINATION_REWARD_THRESHOLD:
        return OUTCOME_SUCCESS
    return outcome
