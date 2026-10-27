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

import pytest
from node_helpers import FakeGoalHandle
from rclpy.parameter import Parameter

from rosetta_ros.nodes.policy_runner_node import PolicyRunnerNode


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


def test_feedback_loop_logs_and_stops_on_adapter_error(rclpy_ctx):
    """An adapter whose feedback() raises ends the status thread (logged) rather
    than spinning on the error or publishing anything."""
    node = PolicyRunnerNode(parameter_overrides=[Parameter("feedback_rate_hz", value=50.0)])
    try:

        class Runner:
            def feedback(self):
                raise RuntimeError("adapter bug")

        node._runner = Runner()
        handle = FakeGoalHandle()
        loop = threading.Thread(target=node._feedback_loop, args=(handle, threading.Event()), daemon=True)
        loop.start()
        loop.join(timeout=2.0)
        assert not loop.is_alive(), "feedback loop kept running after feedback() raised"
        assert handle.feedback == []
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
