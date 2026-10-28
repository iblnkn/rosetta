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

finish_goal maps a work loop's ``termination_reason`` onto the one rcl-legal
terminal transition for it. The mapping it implements:

    a client took the work away  -> CANCELED
    the work reached a defined end -> SUCCEEDED
    the server stopped the work  -> ABORTED

Two constraints it must respect, both of which have bitten before:

- ``canceled()`` is legal ONLY from CANCELING. Calling it from EXECUTING raises
  RCLError out of the execute callback, which once leaked the busy flag and
  rejected every subsequent recording until node restart. So the
  ``is_cancel_requested`` branch decides first, and a "cancelled" reason on a
  goal that never entered CANCELING must still abort rather than raise.
- finish_goal must never write ``termination_reason``. The work loop is its
  single writer, which is what keeps the terminal state and the reported reason
  from disagreeing.

These use a fake goal handle and so can only check which transition was
*requested*. That the transition is legal and that a client sees the terminal
status is proved against a real ActionServer in
test_action_cancel_terminal_state.py.
"""

from types import SimpleNamespace

import pytest
from node_helpers import FakeGoalHandle

from rosetta_ros.node_utils import (
    TERMINATION_CANCELLED,
    TERMINATION_COMPLETED,
    TERMINATION_ERROR,
    TERMINATION_STOPPED,
    TERMINATION_TIMEOUT,
    finish_goal,
)


def _result(termination_reason, message=""):
    return SimpleNamespace(termination_reason=termination_reason, message=message)


@pytest.mark.parametrize(
    "reason",
    [
        TERMINATION_STOPPED,
        TERMINATION_TIMEOUT,
        TERMINATION_COMPLETED,
        "reward_threshold",  # an app's own reason (HIL declares this one) succeeds too
    ],
)
def test_work_that_reached_a_defined_end_succeeds(reason):
    handle = FakeGoalHandle()
    finish_goal(handle, _result(reason))
    assert handle.calls == ["succeed"]


def test_cancelled_reason_without_canceling_state_aborts():
    # The original brick, in its current form: a "cancelled" reason recorded
    # against a goal that never entered CANCELING must NOT call canceled().
    # Reachable if a cancel forward is dropped but the reason was already
    # latched.
    handle = FakeGoalHandle(cancel_requested=False)
    finish_goal(handle, _result(TERMINATION_CANCELLED))
    assert handle.calls == ["abort"]


@pytest.mark.parametrize("reason", [TERMINATION_CANCELLED, TERMINATION_ERROR])
def test_cancel_state_wins_over_the_recorded_reason(reason):
    # The goal IS in CANCELING, so canceled() is the only legal terminal
    # transition regardless of what the loop recorded.
    # finish_goal never rewrites the result either: the loop is its single writer.
    handle = FakeGoalHandle(cancel_requested=True)
    result = _result(reason, message="Recorded 42 messages")
    finish_goal(handle, result)
    assert handle.calls == ["canceled"]
    assert (result.termination_reason, result.message) == (reason, "Recorded 42 messages")
