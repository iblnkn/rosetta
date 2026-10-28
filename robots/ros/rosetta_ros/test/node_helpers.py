# Copyright 2026 Isaac Blankenau
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

"""Fakes and helpers shared by the node tests that drive work-gate callbacks directly."""

import threading
import uuid
from types import SimpleNamespace

from rosetta.policies import RunnerFeedback, RunnerResult


class FakeGoalHandle:
    """An action goal handle that records which terminal transition was requested.

    Each handle gets its own goal id, so cancel routing can tell two handles apart.
    """

    def __init__(self, *, cancel_requested=False, request=None, goal_uuid=None, on_feedback=None):
        self.is_cancel_requested = cancel_requested
        self.request = request
        # 16 raw bytes, the shape rcl hands over.
        self.goal_id = SimpleNamespace(uuid=goal_uuid if goal_uuid is not None else uuid.uuid4().bytes)
        self.calls: list[str] = []
        self.feedback: list = []
        self._on_feedback = on_feedback

    def canceled(self, response=None):
        self.calls.append("canceled")

    def abort(self, response=None):
        self.calls.append("abort")

    def succeed(self, response=None):
        self.calls.append("succeed")

    def publish_feedback(self, feedback):
        # Copy: work loops reuse one Feedback instance across iterations.
        fields = feedback.get_fields_and_field_types()
        self.feedback.append(SimpleNamespace(**{k: getattr(feedback, k) for k in fields}))
        if self._on_feedback is not None:
            self._on_feedback()


class FakePublisher:
    """Collects published messages; ``on_publish`` runs after each one."""

    def __init__(self, on_publish=None):
        self.published = []
        self._on_publish = on_publish

    def publish(self, msg):
        self.published.append(msg)
        if self._on_publish is not None:
            self._on_publish()


class FakeRunner:
    """A PolicyRunner that runs nothing: no framework, no robot.

    ``run(task)`` replaces the run body (default: a clean finish).
    ``request_stop()`` sets ``stopped`` and then calls ``on_stop``. Subclass
    to make another method misbehave.
    """

    def __init__(self, run=None, *, on_stop=None):
        self._run = run
        self._on_stop = on_stop
        self.stopped = threading.Event()

    def setup(self, host, contract) -> None:
        pass

    def run(self, frames, *, task: str, stop_event: threading.Event) -> RunnerResult:
        if self._run is not None:
            return self._run(task)
        return RunnerResult(success=True, message="done")

    def feedback(self) -> RunnerFeedback:
        return RunnerFeedback()

    def request_stop(self) -> None:
        self.stopped.set()
        if self._on_stop is not None:
            self._on_stop()

    def teardown(self) -> None:
        pass


def claim(node):
    """Open acceptance and claim the work slot, as an accepted goal does."""
    node._accepting_work = True
    assert node.try_claim_work() is None
