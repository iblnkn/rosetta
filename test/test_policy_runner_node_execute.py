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
path, runner crash (traceback logged, non-empty client message), and even a
finish_goal that itself raises.

They also pin the RunnerResult -> termination_reason mapping. RunnerResult.success
means only "the run ended without error", and a runner that was told to stop
returns success=True on purpose -- it knows it was stopped, not by whom. Naming
the cause is the node's job, and it reads the reason the stop recorded.
"""

import threading
from types import SimpleNamespace

import pytest

from rosetta.policies import RunnerResult
from rosetta.robots.ros2.nodes.node_utils import (
    TERMINATION_CANCELLED,
    TERMINATION_COMPLETED,
    TERMINATION_ERROR,
    TERMINATION_NODE_DEACTIVATED,
    TERMINATION_TIMEOUT,
)
from rosetta.robots.ros2.nodes.policy_runner_node import PolicyRunnerNode


@pytest.fixture
def node(rclpy_ctx):
    n = PolicyRunnerNode()
    yield n
    n.destroy_node()


class _FakeHandle:
    is_cancel_requested = False

    def __init__(self, prompt="pick up cube", max_duration_s=0.0):
        self.request = SimpleNamespace(prompt=prompt, max_duration_s=max_duration_s)
        self.terminal = None

    def publish_feedback(self, _msg):
        pass

    def succeed(self):
        self.terminal = "succeed"

    def abort(self):
        self.terminal = "abort"

    def canceled(self):
        self.terminal = "canceled"


class _FakeRunner:
    def __init__(self, run=None):
        self._run = run

    def run(self, frames, *, task, stop_event):
        if self._run is not None:
            return self._run(task)
        return RunnerResult(success=True, message="done")

    def feedback(self):
        return SimpleNamespace(queue_depth=0, published_actions=0, status="ok")

    def request_stop(self):
        """Called by _unblock_stop, which every _signal_stop runs."""


def _claim(node):
    node._accepting_work = True
    assert node._try_claim_work() is None


def test_execute_success_finishes_goal_and_releases_busy(node):
    node._runner = _FakeRunner()
    _claim(node)
    handle = _FakeHandle()

    result = node._execute(handle)

    # Nobody asked it to stop, so the control loop finishing on its own is a
    # completion. The runner's own message only restates that, so it does not
    # travel -- a non-empty message means something went wrong.
    assert result.termination_reason == TERMINATION_COMPLETED
    assert result.message == ""
    assert handle.terminal == "succeed"
    assert not node.busy


def test_execute_runner_crash_aborts_with_typed_message(node):
    def boom(_task):
        raise RuntimeError("boom")

    node._runner = _FakeRunner(run=boom)
    _claim(node)
    handle = _FakeHandle()

    result = node._execute(handle)

    assert result.termination_reason == TERMINATION_ERROR
    assert result.message == "RuntimeError: boom"  # never an empty client message
    assert handle.terminal == "abort"
    assert not node.busy


def test_execute_reports_a_failed_run_as_an_error(node):
    node._runner = _FakeRunner(run=lambda _t: RunnerResult(success=False, message="connect refused"))
    _claim(node)
    handle = _FakeHandle()

    result = node._execute(handle)

    assert result.termination_reason == TERMINATION_ERROR
    assert result.message == "connect refused"
    assert handle.terminal == "abort"


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

    node._runner = _FakeRunner(run=stopped)
    _claim(node)
    node._signal_stop(signalled)
    handle = _FakeHandle()
    handle.is_cancel_requested = signalled == TERMINATION_CANCELLED

    result = node._execute(handle)

    assert result.termination_reason == signalled
    assert handle.terminal == terminal


def test_unblock_stop_asks_the_runner_to_stop(node):
    stopped = []
    node._runner = SimpleNamespace(request_stop=lambda: stopped.append(True))
    node._unblock_stop()
    assert stopped == [True]


def test_goal_max_duration_stops_the_run_and_reports_timeout(node):
    """A policy run is one blocking call into the framework, so the deadline is
    a timer rather than a loop check. It must name its reason like any other
    stop, and reach the runner through request_stop()."""

    def run_until_stopped(_task):
        assert node._runner.stopped.wait(5.0), "deadline never asked the runner to stop"
        return RunnerResult(success=True, message="Stopped")

    runner = _FakeRunner(run=run_until_stopped)
    runner.stopped = threading.Event()
    runner.request_stop = runner.stopped.set
    node._runner = runner
    _claim(node)

    result = node._execute(_FakeHandle(max_duration_s=0.05))

    assert result.termination_reason == TERMINATION_TIMEOUT
    assert not node.busy


def test_the_deadline_does_not_fire_on_a_run_that_finishes_first(node):
    node._runner = _FakeRunner()
    _claim(node)

    result = node._execute(_FakeHandle(max_duration_s=30.0))

    # Cancelled in the finally: a timer left armed would signal a stop against
    # whichever claim happened to hold the slot 30 seconds later.
    assert result.termination_reason == TERMINATION_COMPLETED


def test_execute_releases_busy_even_when_finish_goal_raises(node):
    node._runner = _FakeRunner()
    _claim(node)
    handle = _FakeHandle()
    handle.succeed = None  # finish_goal will raise TypeError

    with pytest.raises(TypeError):
        node._execute(handle)

    assert not node.busy
