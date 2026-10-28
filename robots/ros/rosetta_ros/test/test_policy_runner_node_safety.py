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

"""PolicyRunnerNode's own stop hooks, feedback thread and runner teardown.

Base work-gate semantics (claims, cancel routing, safety-after-stop) are tested
on RosettaLifecycleNode in test_lifecycle_node.py. These cover what this node
plugs into them: on deactivate the runner is asked to stop before the safety
action goes out, so the safety action is the robot's last command on the wire.
"""

import threading
from types import SimpleNamespace

import pytest
from node_helpers import FakeGoalHandle, FakeRunner
from rclpy.parameter import Parameter

from rosetta_ros.nodes.policy_runner_node import PolicyRunnerNode


@pytest.fixture
def node(rclpy_ctx):
    n = PolicyRunnerNode()
    yield n
    n.destroy_node()


def test_safety_action_sent_last_after_runner_stops(node):
    calls: list[str] = []
    node._bridge = SimpleNamespace(send_safety_action=lambda: calls.append("safety"))
    node._stop_event = threading.Event()

    def stop_releases_the_goal():
        # Goal in flight: busy releases when the runner is told to stop.
        calls.append("request_stop")
        node._busy = False

    node._busy = True
    node._runner = FakeRunner(on_stop=stop_releases_the_goal)

    node._stop_and_secure(wait_timeout=2.0)

    assert node._stop_event.is_set()
    assert calls == ["request_stop", "safety"]


class _BrokenFeedbackRunner(FakeRunner):
    def feedback(self):
        raise RuntimeError("adapter bug")


class _BrokenTeardownRunner(FakeRunner):
    def teardown(self):
        raise RuntimeError("boom")


def test_feedback_loop_logs_and_stops_on_adapter_error(rclpy_ctx, monkeypatch):
    """An adapter whose feedback() raises ends the status thread (logged) rather
    than spinning on the error or publishing anything."""
    node = PolicyRunnerNode(parameter_overrides=[Parameter("feedback_rate_hz", value=50.0)])
    try:
        errors: list[str] = []
        monkeypatch.setattr(node, "get_logger", lambda: SimpleNamespace(error=errors.append))
        node._runner = _BrokenFeedbackRunner()
        handle = FakeGoalHandle()
        loop = threading.Thread(target=node._feedback_loop, args=(handle, threading.Event()), daemon=True)
        loop.start()
        loop.join(timeout=2.0)
        assert not loop.is_alive(), "feedback loop kept running after feedback() raised"
        assert handle.feedback == []
        (logged,) = errors
        assert "Feedback loop stopped" in logged
        assert "adapter bug" in logged  # the traceback names the adapter's error
    finally:
        node.destroy_node()


def test_teardown_nulls_runner_then_propagates(node):
    """Null-first keeps _teardown_runner idempotent, but the failure is NOT
    swallowed: a framework resource that failed to release (e.g. a policy
    server subprocess) must surface through the base's guarded teardown."""
    node._runner = _BrokenTeardownRunner()
    with pytest.raises(RuntimeError, match="boom"):
        node._teardown_runner()
    assert node._runner is None
    node._teardown_runner()  # second call is a clean no-op
