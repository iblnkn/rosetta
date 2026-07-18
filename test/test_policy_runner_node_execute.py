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

"""_execute: terminal transitions and the busy-release guarantee.

A leaked busy claim rejects every subsequent goal until node restart, so the
release-on-every-exit-path behavior is the invariant these tests pin: happy
path, runner crash (traceback logged, non-empty client message), and even a
finish_goal that itself raises.
"""

from types import SimpleNamespace

import pytest
from rosetta.policies import RunnerResult
from rosetta.robots.ros2.nodes.policy_runner_node import PolicyRunnerNode


@pytest.fixture
def node(rclpy_ctx):
    n = PolicyRunnerNode()
    yield n
    n.destroy_node()


class _FakeHandle:
    is_cancel_requested = False

    def __init__(self, prompt="pick up cube"):
        self.request = SimpleNamespace(prompt=prompt)
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


def _claim(node):
    node._accepting_work = True
    assert node._try_claim_work() is None


def test_execute_success_finishes_goal_and_releases_busy(node):
    node._runner = _FakeRunner()
    _claim(node)
    handle = _FakeHandle()

    result = node._execute(handle)

    assert result.success and result.message == "done"
    assert handle.terminal == "succeed"
    assert not node.busy


def test_execute_runner_crash_aborts_with_typed_message(node):
    def boom(_task):
        raise RuntimeError("boom")

    node._runner = _FakeRunner(run=boom)
    _claim(node)
    handle = _FakeHandle()

    result = node._execute(handle)

    assert not result.success
    assert result.message == "RuntimeError: boom"  # never an empty client message
    assert handle.terminal == "abort"
    assert not node.busy


def test_execute_releases_busy_even_when_finish_goal_raises(node):
    node._runner = _FakeRunner()
    _claim(node)
    handle = _FakeHandle()
    handle.succeed = None  # finish_goal will raise TypeError

    with pytest.raises(TypeError):
        node._execute(handle)

    assert not node.busy
