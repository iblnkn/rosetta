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

"""send_goal and cancel_goal: never raise, and say why a child goal did not start."""

from types import SimpleNamespace

import pytest

from rosetta_ros.child_goals import cancel_goal, send_goal


class Logger:
    def __init__(self):
        self.lines = []

    def info(self, msg):
        self.lines.append(("info", msg))

    def warning(self, msg):
        self.lines.append(("warning", msg))

    def error(self, msg):
        self.lines.append(("error", msg))


class Future:
    def __init__(self, result=None, done=True):
        self._result, self._done = result, done

    def done(self):
        return self._done

    def result(self):
        return self._result


class Client:
    def __init__(self, *, available=True, response=None):
        self.available, self.response, self.sent = available, response, []

    def wait_for_server(self, timeout_sec):
        return self.available

    def send_goal_async(self, goal, feedback_callback=None):
        self.sent.append(goal)
        return self.response


def _send(client, logger):
    return send_goal(client, "goal", name="Child", logger=logger, server_timeout=0, response_timeout=0.02)


def test_accepted_goal_returns_its_handle():
    handle = SimpleNamespace(accepted=True)
    logger = Logger()
    assert _send(Client(response=Future(handle)), logger) is handle
    assert logger.lines == [("info", "Child started")]


@pytest.mark.parametrize(
    ("client", "error"),
    [
        (Client(available=False), "Child action server not available"),
        (Client(response=Future(done=False)), "Child goal send timed out"),
        (Client(response=Future(SimpleNamespace(accepted=False))), "Child goal rejected"),
    ],
)
def test_goal_that_does_not_start_returns_none_and_says_why(client, error):
    logger = Logger()
    assert _send(client, logger) is None
    assert logger.lines == [("error", error)]


class Handle:
    def __init__(self, *, cancel_done=True, result=None, result_done=True, raises=None):
        self._cancel_done, self._result, self._result_done, self._raises = cancel_done, result, result_done, raises

    def cancel_goal_async(self):
        if self._raises:
            raise self._raises
        return Future(done=self._cancel_done)

    def get_result_async(self):
        return Future(SimpleNamespace(result=self._result), done=self._result_done)


def _cancel(handle, logger, **kwargs):
    return cancel_goal(handle, name="Child", logger=logger, cancel_timeout=0.02, result_timeout=0.02, **kwargs)


def test_cancelling_no_goal_is_a_no_op():
    logger = Logger()
    assert _cancel(None, logger) is None
    assert logger.lines == []


def test_cancel_with_wait_returns_the_result():
    assert _cancel(Handle(result="bag"), Logger(), wait_result=True) == "bag"


def test_cancel_timeout_only_warns_and_still_waits_for_the_result():
    logger = Logger()
    assert _cancel(Handle(cancel_done=False, result="bag"), logger, wait_result=True) == "bag"
    assert ("warning", "Child cancel timed out") in logger.lines


def test_result_timeout_returns_none():
    logger = Logger()
    assert _cancel(Handle(result_done=False), logger, wait_result=True) is None
    assert ("warning", "Child result timed out after cancel") in logger.lines


def test_cancel_failure_is_logged_not_raised():
    logger = Logger()
    assert _cancel(Handle(raises=RuntimeError("gone")), logger) is None
    assert logger.lines == [("warning", "Failed to cancel Child: gone")]
