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

"""_record state reconciliation: no exit path may brick subsequent recordings.

Runs EpisodeRecorderNode._record unbound on a fake (no rclpy init), with the
per-recording stop state pre-armed the way the work-gate claim does at accept
time. Every path — success, open failure, metadata failure, service cancel,
bad feedback rate — must release the busy guard, leave the stop event alone
(each claim arms a fresh one; _record owns none of that state), and end an
action goal via an rcl-legal transition.
"""

import threading
from types import SimpleNamespace

from rosetta.robots.ros2.nodes.episode_recorder_node import EpisodeRecorderNode
from rosetta.robots.ros2.rosetta_lifecycle_node import RosettaLifecycleNode


class _FakeLogger:
    def debug(self, _msg):
        pass

    def info(self, _msg):
        pass

    def warning(self, _msg):
        pass

    def error(self, _msg):
        pass


class FakeGoalHandle:
    """Records which terminal transition was requested."""

    def __init__(self, cancel_requested: bool = False):
        self.is_cancel_requested = cancel_requested
        self.calls: list[str] = []
        self.feedback: list = []

    def canceled(self):
        self.calls.append("canceled")

    def abort(self):
        self.calls.append("abort")

    def succeed(self):
        self.calls.append("succeed")

    def publish_feedback(self, feedback):
        self.feedback.append(feedback)


class FakeRecorder:
    """Enough of EpisodeRecorderNode for _record to run unbound.

    Constructed pre-armed (busy claimed, stop event created), exactly the
    state _arm_recording leaves behind at accept time. The stop event is
    pre-set so the record loop exits on its first wait.
    """

    # The real work-guard machinery, unbound like _record below: the fake
    # must exercise the same release/bind semantics the node runs.
    _goal_work = RosettaLifecycleNode._goal_work
    _signal_stop = RosettaLifecycleNode._signal_stop

    def __init__(self, tmp_path, *, feedback_rate=50.0, open_error=None, metadata_error=None):
        self._busy = True  # claim held, as _arm_recording leaves it at accept time
        self._work_gate = threading.Lock()
        self._active_goal = None
        self._stop_event = threading.Event()
        self._stop_event.set()
        self._cancel_requested = False
        self._messages_written = 0
        self._topic_msg_counts = {}
        self._writer_lock = threading.Lock()
        self._discovered_topics = []
        self._topics = []
        self._contract_text = ""
        self._last_bag_dir = None
        self._params = {"default_max_duration": 0.0, "feedback_rate_hz": feedback_rate}
        self._bag_dir = tmp_path / "ep"
        self._open_error = open_error
        self._metadata_error = metadata_error
        self.close_calls = 0
        self.metadata_calls = 0

    def get_logger(self):
        return _FakeLogger()

    def get_parameter(self, name):
        return SimpleNamespace(value=self._params[name])

    def _create_bag_dir(self):
        return self._bag_dir

    def _open_writer(self, bag_dir):
        if self._open_error is not None:
            raise self._open_error
        bag_dir.mkdir(parents=True)  # the real writer creates the directory
        self._messages_written = 7  # stand-in for messages written during the episode

    def _close_writer(self):
        self.close_calls += 1

    def _write_metadata(self, bag_dir, prompt, contract_text):
        self.metadata_calls += 1
        if self._metadata_error is not None:
            raise self._metadata_error

    def _log_topic_summary(self, bag_dir, discovered_topics, elapsed):
        pass


_record = EpisodeRecorderNode._record


def _assert_reconciled(fake):
    assert fake._busy is False
    assert fake._stop_event is not None  # _record never touches the claim's event
    assert fake.close_calls >= 1


def test_clean_stop_succeeds_and_reconciles(tmp_path):
    fake = FakeRecorder(tmp_path)
    handle = FakeGoalHandle()
    result = _record(fake, "pick up cube", handle)
    assert handle.calls == ["succeed"]
    assert result.success is True
    assert result.messages_written == 7
    assert result.message == "Recorded 7 messages"
    assert fake.metadata_calls == 1
    assert fake._last_bag_dir == fake._bag_dir
    _assert_reconciled(fake)


def test_service_start_has_no_goal_transitions(tmp_path):
    fake = FakeRecorder(tmp_path)
    result = _record(fake, "pick up cube")
    assert result.success is True
    assert result.messages_written == 7
    _assert_reconciled(fake)


def test_open_writer_failure_aborts_without_metadata(tmp_path):
    fake = FakeRecorder(tmp_path, open_error=RuntimeError("disk full"))
    handle = FakeGoalHandle()
    result = _record(fake, "p", handle)
    assert handle.calls == ["abort"]
    assert result.success is False
    assert "disk full" in result.message
    assert fake.metadata_calls == 0  # writer never opened: nothing to annotate
    assert fake._last_bag_dir is None  # no bag dir on disk
    _assert_reconciled(fake)


def test_metadata_failure_fails_result_but_keeps_counts_and_bag(tmp_path):
    fake = FakeRecorder(tmp_path, metadata_error=RuntimeError("metadata.yaml never appeared"))
    handle = FakeGoalHandle()
    result = _record(fake, "p", handle)
    assert handle.calls == ["abort"]
    assert result.success is False
    assert "metadata.yaml" in result.message
    assert result.messages_written == 7
    # The partial bag exists and must stay deletable via ~/delete_last_bag.
    assert fake._last_bag_dir == fake._bag_dir
    _assert_reconciled(fake)


def test_service_cancel_finishes_aborted_with_cancel_message(tmp_path):
    fake = FakeRecorder(tmp_path)
    fake._cancel_requested = True  # what _on_cancel_service sets before _signal_stop
    handle = FakeGoalHandle()
    result = _record(fake, "p", handle)
    assert handle.calls == ["abort"]
    assert result.success is False
    assert result.message == "Cancelled via service"
    _assert_reconciled(fake)


def test_action_cancel_finishes_canceled(tmp_path):
    fake = FakeRecorder(tmp_path)
    handle = FakeGoalHandle(cancel_requested=True)
    result = _record(fake, "p", handle)
    assert handle.calls == ["canceled"]
    assert result.success is False
    _assert_reconciled(fake)


def test_unexpected_error_before_open_aborts_cleanly(tmp_path):
    # feedback_rate_hz=0 is unrepresentable via ROS parameters
    # (positive_rate_descriptor), but any unexpected pre-open exception must
    # still abort the goal and reconcile state.
    fake = FakeRecorder(tmp_path, feedback_rate=0.0)  # 1/0 raises in the loop setup
    handle = FakeGoalHandle()
    result = _record(fake, "p", handle)
    assert handle.calls == ["abort"]
    assert result.success is False
    _assert_reconciled(fake)


def test_goal_transition_raising_still_releases_busy(tmp_path):
    class ExplodingHandle(FakeGoalHandle):
        def succeed(self):
            raise RuntimeError("rcl said no")

    fake = FakeRecorder(tmp_path)
    handle = ExplodingHandle()
    try:
        _record(fake, "p", handle)
    except RuntimeError:
        pass
    _assert_reconciled(fake)


def test_stop_event_untouched_at_busy_release(tmp_path):
    """A new accept arms a fresh event the moment _busy is released; _record
    must never write self._stop_event, or it could clobber the new claim's."""
    fake = FakeRecorder(tmp_path)
    claimed_event = fake._stop_event
    _record(fake, "p")
    # The claim was released (work done), but _record left self._stop_event
    # exactly as _arm_recording set it — reassigning it would stop the wrong
    # claim's work once the next accept arms a fresh event.
    assert fake._busy is False
    assert fake._stop_event is claimed_event
