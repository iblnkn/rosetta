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

"""Client node deactivate ordering and goal-accept exclusion.

Regressions pinned:
- on_deactivate used to send the safety action BEFORE stopping the runner, so
  the still-running policy loop published at least one more action over it —
  the robot's last command on the wire was a stale policy action, not the
  safety action.
- _on_goal used to check `self._active_goal is not None`, which _execute set
  only later: under the MultiThreadedExecutor two goals could both be
  accepted, running two policy loops against one runner/bridge.
"""

import threading

import pytest
from rclpy.action import GoalResponse
from rosetta.robots.ros2.nodes.policy_runner_node import PolicyRunnerNode


@pytest.fixture
def node(rclpy_ctx):
    n = PolicyRunnerNode()
    yield n
    n.destroy_node()


class _OrderedFakes:
    """Bridge/runner fakes that record the global call order."""

    def __init__(self):
        self.calls: list[str] = []

    def make_bridge(self):
        outer = self

        class Bridge:
            def send_safety_action(self):
                outer.calls.append("safety")

        return Bridge()

    def make_runner(self, on_stop=None):
        outer = self

        class Runner:
            def request_stop(self):
                outer.calls.append("request_stop")
                if on_stop is not None:
                    on_stop()

        return Runner()


def test_safety_action_sent_last_after_runner_stops(node):
    fakes = _OrderedFakes()
    node._bridge = fakes.make_bridge()
    node._stop_event = threading.Event()
    # Goal in flight: busy releases when the runner is told to stop.
    node._busy.try_acquire()
    node._runner = fakes.make_runner(on_stop=node._busy.release)

    node._stop_and_secure(wait_timeout=2.0)

    assert node._stop_event.is_set()
    assert fakes.calls == ["request_stop", "safety"]


def test_safety_action_still_sent_when_goal_hangs(node):
    fakes = _OrderedFakes()
    node._bridge = fakes.make_bridge()
    node._runner = fakes.make_runner()
    node._busy.try_acquire()  # goal never ends

    node._stop_and_secure(wait_timeout=0.2)

    assert fakes.calls == ["request_stop", "safety"]  # safety is last-best-effort
    node._busy.release()


def test_goal_accept_is_mutually_exclusive(node):
    node._accepting_goals = True
    assert node._on_goal(None) == GoalResponse.ACCEPT
    assert node._on_goal(None) == GoalResponse.REJECT  # busy until release
    node._busy.release()
    assert node._on_goal(None) == GoalResponse.ACCEPT
    node._busy.release()
