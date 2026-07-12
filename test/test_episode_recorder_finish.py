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

"""Terminal-transition tests for finish_goal (shared by all action nodes).

Regression guard for the recorder cancel-service brick: rcl only allows
canceled() from the CANCELING state (entered only by an action-protocol
cancel). A Trigger-service cancel leaves the goal EXECUTING, so it must
finish via abort() — the old code called canceled() there, which raised
RCLError outside the try/except and leaked _busy/_is_recording/_goal_handle,
rejecting every subsequent recording until node restart. The helper now
lives in node_utils and is also used by the client and HIL manager nodes.
"""

from types import SimpleNamespace

from rosetta.robots.ros2.nodes.node_utils import finish_goal


class FakeGoalHandle:
    """Records which terminal transition was requested."""

    def __init__(self, cancel_requested: bool = False):
        self.is_cancel_requested = cancel_requested
        self.calls: list[str] = []

    def canceled(self):
        self.calls.append("canceled")

    def abort(self):
        self.calls.append("abort")

    def succeed(self):
        self.calls.append("succeed")


def _result(success=True, message=""):
    return SimpleNamespace(success=success, message=message)


def test_action_protocol_cancel_finishes_canceled():
    handle = FakeGoalHandle(cancel_requested=True)
    result = _result()
    finish_goal(handle, result, success_message="ok")
    assert handle.calls == ["canceled"]
    assert result.success is False
    assert result.message == "Cancelled"


def test_service_cancel_finishes_aborted_never_canceled():
    # The regression: canceled() from EXECUTING raises in rcl; the service
    # path must use abort().
    handle = FakeGoalHandle(cancel_requested=False)
    result = _result()
    finish_goal(handle, result, service_cancelled=True, success_message="ok")
    assert handle.calls == ["abort"]
    assert "canceled" not in handle.calls
    assert result.success is False
    assert result.message == "Cancelled via service"


def test_normal_completion_succeeds():
    handle = FakeGoalHandle()
    result = _result(success=True)
    finish_goal(handle, result, success_message="Recorded 42 messages")
    assert handle.calls == ["succeed"]
    assert result.success is True
    assert result.message == "Recorded 42 messages"


def test_failure_aborts_and_keeps_message():
    handle = FakeGoalHandle()
    result = _result(success=False, message="writer exploded")
    finish_goal(handle, result)
    assert handle.calls == ["abort"]
    assert result.success is False
    assert result.message == "writer exploded"


def test_action_cancel_takes_precedence_over_service_cancel():
    # Both flags set: the goal IS in CANCELING, so canceled() is the legal
    # and correct terminal state.
    handle = FakeGoalHandle(cancel_requested=True)
    result = _result()
    finish_goal(handle, result, service_cancelled=True, success_message="ok")
    assert handle.calls == ["canceled"]
