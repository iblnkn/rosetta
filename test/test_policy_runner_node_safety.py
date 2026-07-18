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
from rclpy.action import CancelResponse, GoalResponse
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
    node._busy = True
    node._runner = fakes.make_runner(on_stop=lambda: setattr(node, "_busy", False))

    node._stop_and_secure(wait_timeout=2.0)

    assert node._stop_event.is_set()
    assert fakes.calls == ["request_stop", "safety"]


def test_safety_action_still_sent_when_goal_hangs(node):
    fakes = _OrderedFakes()
    node._bridge = fakes.make_bridge()
    node._runner = fakes.make_runner()
    node._busy = True  # goal never ends

    node._stop_and_secure(wait_timeout=0.2)

    assert fakes.calls == ["request_stop", "safety"]  # safety is last-best-effort
    node._busy = False


def test_goal_accept_is_mutually_exclusive(node):
    node._accepting_work = True
    assert node._on_goal(None) == GoalResponse.ACCEPT
    assert node._on_goal(None) == GoalResponse.REJECT  # busy until release
    node._busy = False
    assert node._on_goal(None) == GoalResponse.ACCEPT
    node._busy = False


def test_goal_rejected_before_activation(node):
    assert node._on_goal(None) == GoalResponse.REJECT  # initial state: not accepting
    assert not node.busy  # a rejected goal claims nothing


def test_cancel_in_accept_to_execute_window_is_not_lost(node):
    # The stop event must exist from the moment a goal is accepted, and a
    # cancel landing before _execute binds the goal must still take effect:
    # the cancel callback skips signaling (no bound goal to route to), and
    # _goal_work honors it by re-checking is_cancel_requested after binding.
    fakes = _OrderedFakes()
    node._runner = fakes.make_runner()
    node._accepting_work = True

    assert node._on_goal(None) == GoalResponse.ACCEPT
    assert node._stop_event is not None and not node._stop_event.is_set()

    class Handle:
        goal_id = object()
        is_cancel_requested = True  # cancel accepted in the accept->bind window

    assert node._on_cancel(Handle()) == CancelResponse.ACCEPT  # unrouted: no bound goal yet
    assert not node._stop_event.is_set()  # ...so the callback signals nothing

    with node._goal_work(Handle()) as stop_event:
        assert stop_event.is_set()  # the bind-time re-check honored the cancel
    assert fakes.calls == ["request_stop"]
    assert not node.busy  # released by _goal_work


def test_stale_cancel_does_not_stop_next_goal(node):
    # A late cancel callback for finished goal A must not set goal B's stop
    # event: cancels route by goal id against the bound goal.
    node._accepting_work = True

    class Handle:
        def __init__(self):
            self.goal_id = object()
            self.is_cancel_requested = False

    goal_a, goal_b = Handle(), Handle()

    assert node._on_goal(None) == GoalResponse.ACCEPT
    with node._goal_work(goal_a):
        pass  # goal A runs and finishes

    assert node._on_goal(None) == GoalResponse.ACCEPT  # goal B claims the slot
    with node._goal_work(goal_b) as stop_b:
        node._on_cancel(goal_a)  # stale cancel for A arrives late
        assert not stop_b.is_set()  # B unaffected
        node._on_cancel(goal_b)  # a cancel for the bound goal still works
        assert stop_b.is_set()


def test_feedback_loop_logs_and_stops_on_adapter_error(rclpy_ctx):
    from rclpy.parameter import Parameter

    node = PolicyRunnerNode(parameter_overrides=[Parameter("feedback_rate_hz", value=50.0)])
    try:

        class Runner:
            def feedback(self):
                raise RuntimeError("adapter bug")

        class Handle:
            is_cancel_requested = False

            def publish_feedback(self, msg):
                raise AssertionError("must not publish after a feedback() error")

        node._runner = Runner()
        # Returns (via break) instead of propagating and killing the thread.
        node._feedback_loop(Handle(), threading.Event())
    finally:
        node.destroy_node()


def test_teardown_nulls_runner_then_propagates(node):
    """Null-first keeps _teardown_runner idempotent, but the failure is NOT
    swallowed: a framework resource that failed to release (e.g. a policy
    server subprocess) must surface through the base's guarded teardown."""

    class Runner:
        def teardown(self):
            raise RuntimeError("boom")

    node._runner = Runner()
    with pytest.raises(RuntimeError, match="boom"):
        node._teardown_runner()
    assert node._runner is None
    node._teardown_runner()  # second call is a clean no-op
