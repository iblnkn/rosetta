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

"""_execute: terminal transitions, reason mapping, and the busy-release guarantee.

A leaked busy claim rejects every subsequent goal until node restart, so the
release-on-every-exit-path behavior is the invariant these tests pin: happy
path and runner crash (traceback logged, non-empty client message). That the
release survives an exception anywhere in the work is pinned on the base's
goal_work in test_lifecycle_node.py.

They also pin the RunnerResult -> termination_reason mapping. RunnerResult.success
means only "the run ended without error", and a runner that was told to stop
returns success=True on purpose -- it knows it was stopped, not by whom. Naming
the cause is the node's job, and it reads the reason the stop recorded.
"""

from types import SimpleNamespace

import pytest
from node_helpers import FakeGoalHandle, FakePublisher, FakeRunner, claim

from rosetta.policies import PolicyRunner, RunnerResult
from rosetta_ros.node_utils import (
    TERMINATION_CANCELLED,
    TERMINATION_COMPLETED,
    TERMINATION_ERROR,
    TERMINATION_NODE_DEACTIVATED,
    TERMINATION_TIMEOUT,
    wait_until,
)
from rosetta_ros.nodes.policy_runner_node import PolicyRunnerNode


@pytest.fixture
def node(rclpy_ctx):
    n = PolicyRunnerNode()
    # The topics setup() would create, without loading a contract.
    n._status_pub, n._result_pub = FakePublisher(), FakePublisher()
    yield n
    n.destroy_node()


def _handle(max_duration_s=0.0, cancel_requested=False):
    request = SimpleNamespace(prompt="pick up cube", max_duration_s=max_duration_s)
    return FakeGoalHandle(request=request, cancel_requested=cancel_requested)


def test_fake_runner_implements_the_policy_runner_protocol():
    # A method added to or renamed in PolicyRunner fails here, not silently in the fake.
    assert isinstance(FakeRunner(), PolicyRunner)


def test_execute_success_finishes_goal_and_releases_busy(node):
    node._runner = FakeRunner()
    claim(node)
    handle = _handle()

    result = node._execute(handle)

    # Nobody asked it to stop, so the control loop finishing on its own is a
    # completion. The runner's own message only restates that, so it does not
    # travel -- a non-empty message means something went wrong.
    assert result.termination_reason == TERMINATION_COMPLETED
    assert result.message == ""
    assert handle.calls == ["succeed"]
    assert not node.busy


def test_execute_runner_crash_aborts_with_typed_message(node):
    def boom(_task):
        raise RuntimeError("boom")

    node._runner = FakeRunner(run=boom)
    claim(node)
    handle = _handle()

    result = node._execute(handle)

    assert result.termination_reason == TERMINATION_ERROR
    assert result.message == "RuntimeError: boom"  # never an empty client message
    assert handle.calls == ["abort"]
    assert not node.busy


def test_execute_reports_a_failed_run_as_an_error(node):
    node._runner = FakeRunner(run=lambda _t: RunnerResult(success=False, message="connect refused"))
    claim(node)
    handle = _handle()

    result = node._execute(handle)

    assert result.termination_reason == TERMINATION_ERROR
    assert result.message == "connect refused"
    assert handle.calls == ["abort"]


@pytest.mark.parametrize(
    ("signalled", "terminal"),
    [
        (TERMINATION_CANCELLED, "canceled"),
        (TERMINATION_NODE_DEACTIVATED, "abort"),
    ],
)
def test_execute_reports_the_recorded_stop_reason_over_the_runners_view(node, signalled, terminal):
    """The runner returns success=True for any stop it was asked to make. The
    node names which stop it was, so "cancelled" never gets reported as a
    completion."""

    def stopped(_task):
        return RunnerResult(success=True, message="Stopped")

    node._runner = FakeRunner(run=stopped)
    claim(node)
    node.signal_stop(signalled)
    handle = _handle(cancel_requested=signalled == TERMINATION_CANCELLED)

    result = node._execute(handle)

    assert result.termination_reason == signalled
    assert handle.calls == [terminal]


def test_goal_max_duration_stops_the_run_and_reports_timeout(node):
    """A policy run is one blocking call into the framework, so the deadline is
    a timer rather than a loop check. It must name its reason like any other
    stop, and reach the runner through request_stop()."""

    def run_until_stopped(_task):
        assert node._runner.stopped.wait(5.0), "deadline never asked the runner to stop"
        return RunnerResult(success=True, message="Stopped")

    node._runner = FakeRunner(run=run_until_stopped)
    claim(node)

    result = node._execute(_handle(max_duration_s=0.05))

    assert result.termination_reason == TERMINATION_TIMEOUT
    assert not node.busy


def test_the_deadline_does_not_fire_on_a_run_that_finishes_first(node):
    """A deadline left armed past its run would stop whichever claim holds the slot when it fires."""
    node._runner = FakeRunner()
    claim(node)
    result = node._execute(_handle(max_duration_s=0.05))
    assert result.termination_reason == TERMINATION_COMPLETED

    claim(node)  # the next run
    assert not wait_until(lambda: node.stop_reason is not None, timeout=0.3)


@pytest.mark.parametrize("via_action", [False, True], ids=["service", "action"])
@pytest.mark.parametrize(
    "run_result",
    [RunnerResult(success=True, message="done"), RunnerResult(success=False, message="boom")],
    ids=["clean", "failed"],
)
def test_every_run_publishes_the_result_it_returns(node, via_action, run_result):
    """~/result is how a client that did not send the goal (or started by service) learns how a run ended."""
    node._runner = FakeRunner(run=lambda _task: run_result)
    claim(node)
    result = node._execute(_handle()) if via_action else node._run("pick up cube")
    (published,) = node._result_pub.published
    assert (published.termination_reason, published.message) == (result.termination_reason, result.message)


def test_status_is_published_without_a_goal(node):
    """The status thread runs for service starts too, publishing no action feedback."""
    from rclpy.parameter import Parameter

    node.set_parameters([Parameter("feedback_rate_hz", value=200.0)])

    def run(_task):
        assert wait_until(lambda: node._status_pub.published, timeout=2.0, poll=0.005)
        return RunnerResult(success=True, message="done")

    node._runner = FakeRunner(run=run)
    claim(node)
    node._run("pick up cube")
    assert node._status_pub.published
