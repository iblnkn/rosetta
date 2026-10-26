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

"""_record state reconciliation and reason reporting.

Runs EpisodeRecorderNode._record unbound on a fake (no rclpy init), with the
per-recording stop state pre-armed the way the work-gate claim does at accept
time. Every path — success, open failure, metadata failure, cancel, failed bag
write, bad feedback rate — must release the busy guard, leave the stop event
alone (each claim arms a fresh one; _record owns none of that state), end an
action goal via an rcl-legal transition, and report why it ended.

The reason precedence _record implements, and which these tests pin:

    an exception on this thread   >  a stop somebody signalled  >
    the loop's own decision       >  a plain stop

Signalled beats loop-decided so a failed bag write landing in the same tick as
a timeout still reports as an error rather than a clean finish.
"""

import threading
from types import SimpleNamespace

import yaml
from std_srvs.srv import Trigger

from rosetta_ros.bag_metadata import BAG_CUSTOM_DATA_KEY, BAG_GOAL_ID_KEY, BAG_METADATA_KEY
from rosetta_ros.node_utils import (
    TERMINATION_CANCELLED,
    TERMINATION_ERROR,
    TERMINATION_NODE_DEACTIVATED,
    TERMINATION_STOPPED,
    TERMINATION_TIMEOUT,
)
from rosetta_ros.nodes.episode_recorder_node import EpisodeRecorderNode
from rosetta_ros.rosetta_lifecycle_node import RosettaLifecycleNode


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

    def __init__(self, cancel_requested: bool = False, on_feedback=None):
        self.is_cancel_requested = cancel_requested
        self.calls: list[str] = []
        self.feedback: list = []
        self._on_feedback = on_feedback
        # 16 raw bytes, the shape rcl hands over.
        self.goal_id = SimpleNamespace(uuid=bytes(range(16)))

    def canceled(self):
        self.calls.append("canceled")

    def abort(self):
        self.calls.append("abort")

    def succeed(self):
        self.calls.append("succeed")

    def publish_feedback(self, feedback):
        # Copy: _record reuses one Feedback instance across the loop.
        fields = feedback.get_fields_and_field_types()
        self.feedback.append(SimpleNamespace(**{k: getattr(feedback, k) for k in fields}))
        if self._on_feedback is not None:
            self._on_feedback()


class FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(msg)


class FakeRecorder:
    """Enough of EpisodeRecorderNode for _record to run unbound.

    Constructed pre-armed (busy claimed, stop event created, no stop reason),
    exactly the state try_claim_work leaves behind at accept time. The stop
    event is pre-set by default so the record loop exits on its first wait;
    tests that need the loop to actually run clear it.
    """

    # The real work-guard machinery, unbound like _record below: the fake must
    # exercise the same release/bind and reason-latching semantics the node runs.
    goal_work = RosettaLifecycleNode.goal_work
    signal_stop = RosettaLifecycleNode.signal_stop
    _record_stop = RosettaLifecycleNode._record_stop
    unblock_stop = RosettaLifecycleNode.unblock_stop
    stop_reason = RosettaLifecycleNode.stop_reason
    stop_detail = RosettaLifecycleNode.stop_detail
    # Real, not stubbed: the goal id it derives lands in bag metadata, so a
    # broken conversion would silently ship bags with no provenance.
    _goal_id_str = staticmethod(EpisodeRecorderNode._goal_id_str)

    def __init__(self, tmp_path, *, feedback_rate=50.0, max_duration=0.0, open_error=None, metadata_error=False):
        self._busy = True  # claim held, as try_claim_work leaves it at accept time
        self._work_gate = threading.Lock()
        self._active_goal = None
        self._stop_event = threading.Event()
        self._stop_event.set()
        self._stop_reason = None
        self._stop_detail = ""
        self._messages_written = 0
        self._topic_msg_counts = {}
        self._writer_lock = threading.Lock()
        self._writer = None
        self._topics = []
        self._contract_text = ""
        self._bag_lock = threading.Lock()
        self._status_pub = FakePublisher()
        self._result_pub = FakePublisher()
        self._active_bag = None
        self._discard_active = False
        self._last_bag_dir = None
        self._params = {"default_max_duration_s": max_duration, "feedback_rate_hz": feedback_rate}
        self._bag_dir = tmp_path / "ep"
        self._open_error = open_error
        self._metadata_error = metadata_error
        self.close_calls = 0

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
        return []

    def _close_writer(self):
        # The real writer writes metadata.yaml on close; metadata_error
        # simulates a bag whose metadata.yaml is missing.
        self.close_calls += 1
        if self._bag_dir.exists() and not self._metadata_error:
            (self._bag_dir / "metadata.yaml").write_text(yaml.safe_dump({BAG_METADATA_KEY: {}}))

    def custom_data(self) -> dict:
        meta = yaml.safe_load((self._bag_dir / "metadata.yaml").read_text())
        return meta[BAG_METADATA_KEY].get(BAG_CUSTOM_DATA_KEY, {})

    def _log_topic_summary(self, bag_dir, discovered_topics, elapsed):
        pass


_record = EpisodeRecorderNode._record


def _delete_last_bag(fake):
    return EpisodeRecorderNode._on_delete_last_bag_service(fake, None, Trigger.Response())


def _assert_reconciled(fake):
    assert fake._busy is False
    assert fake._stop_event is not None  # _record never touches the claim's event
    assert fake.close_calls >= 1


def test_clean_stop_succeeds_and_reconciles(tmp_path):
    fake = FakeRecorder(tmp_path)
    handle = FakeGoalHandle()
    result = _record(fake, "pick up cube", handle)
    assert handle.calls == ["succeed"]
    assert result.termination_reason == TERMINATION_STOPPED
    assert result.messages_written == 7
    # message carries only what no other field does. A clean end has nothing to
    # add, so it stays empty rather than restating messages_written.
    assert result.message == ""
    # Provenance: the bag names the goal that produced it, in canonical hex.
    assert fake.custom_data()[BAG_GOAL_ID_KEY] == "00010203-0405-0607-0809-0a0b0c0d0e0f"
    assert fake._last_bag_dir == fake._bag_dir
    _assert_reconciled(fake)


def test_service_start_has_no_goal_transitions(tmp_path):
    fake = FakeRecorder(tmp_path)
    result = _record(fake, "pick up cube")
    # The service path gets a fully populated reason too: _record's return value
    # is uniform across both start paths.
    assert result.termination_reason == TERMINATION_STOPPED
    assert result.messages_written == 7
    # No goal, so no goal id to record -- empty, never a fabricated one.
    assert BAG_GOAL_ID_KEY not in fake.custom_data()
    _assert_reconciled(fake)


def test_goal_max_duration_overrides_the_node_default(tmp_path):
    # The point of the goal field: a timed recording without a `ros2 param set`
    # first, and without racing another client's parameter write.
    fake = FakeRecorder(tmp_path, max_duration=0.0)  # node default: no limit
    fake._stop_event.clear()
    handle = FakeGoalHandle()
    result = _record(fake, "p", handle, max_duration=0.01)
    assert result.termination_reason == TERMINATION_TIMEOUT
    assert handle.calls == ["succeed"]


def test_node_default_applies_when_the_goal_leaves_it_unset(tmp_path):
    fake = FakeRecorder(tmp_path, max_duration=0.01)
    fake._stop_event.clear()
    handle = FakeGoalHandle()
    result = _record(fake, "p", handle, max_duration=0.0)
    assert result.termination_reason == TERMINATION_TIMEOUT


def test_open_writer_failure_aborts_without_metadata(tmp_path):
    fake = FakeRecorder(tmp_path, open_error=RuntimeError("disk full"))
    handle = FakeGoalHandle()
    result = _record(fake, "p", handle)
    assert handle.calls == ["abort"]
    assert result.termination_reason == TERMINATION_ERROR
    assert "disk full" in result.message
    assert not fake._bag_dir.exists()  # writer never opened: nothing to annotate
    assert fake._last_bag_dir is None  # no bag dir on disk
    _assert_reconciled(fake)


def test_metadata_failure_fails_result_but_keeps_counts_and_bag(tmp_path):
    fake = FakeRecorder(tmp_path, metadata_error=True)
    handle = FakeGoalHandle()
    result = _record(fake, "p", handle)
    assert handle.calls == ["abort"]
    assert result.termination_reason == TERMINATION_ERROR
    assert "metadata.yaml" in result.message
    assert result.messages_written == 7
    # The partial bag exists and must stay deletable via ~/delete_last_bag.
    assert fake._last_bag_dir == fake._bag_dir
    _assert_reconciled(fake)


def test_action_cancel_finishes_canceled(tmp_path):
    # What the base's on_cancel does, including for the cancels that
    # ~/cancel_recording forwards to the action server's own cancel service.
    fake = FakeRecorder(tmp_path)
    fake._stop_reason = TERMINATION_CANCELLED
    handle = FakeGoalHandle(cancel_requested=True)
    result = _record(fake, "p", handle)
    assert handle.calls == ["canceled"]
    assert result.termination_reason == TERMINATION_CANCELLED
    # A cancelled recording still hands back its bag: CANCELED says who ended
    # the work, not that anything was lost.
    assert result.messages_written == 7
    assert result.bag_path == str(fake._bag_dir)
    _assert_reconciled(fake)


def test_deactivate_mid_recording_reports_node_deactivated(tmp_path):
    fake = FakeRecorder(tmp_path)
    fake._stop_reason = TERMINATION_NODE_DEACTIVATED
    handle = FakeGoalHandle()
    result = _record(fake, "p", handle)
    # The server chose to stop the goal, which the ROS 2 action docs call an
    # abort. The reason is what tells it apart from a genuine error.
    assert handle.calls == ["abort"]
    assert result.termination_reason == TERMINATION_NODE_DEACTIVATED
    _assert_reconciled(fake)


def test_bag_write_failure_aborts_with_error_reason(tmp_path):
    """A failed bag write must abort the goal with an error reason.

    The write exception is caught inside the subscription callback, so it never
    reaches _record's try/except. The callback must record the reason when it
    sets the stop event; otherwise the loop exits normally and the goal
    SUCCEEDS with a truncated bag.
    """
    fake = FakeRecorder(tmp_path)
    fake.signal_stop(TERMINATION_ERROR, "Write failed on /camera/image: no space left on device")
    handle = FakeGoalHandle()
    result = _record(fake, "p", handle)
    assert handle.calls == ["abort"]
    assert result.termination_reason == TERMINATION_ERROR
    assert "/camera/image" in result.message  # names the topic that failed
    assert result.messages_written == 7  # the partial bag is still reported
    _assert_reconciled(fake)


def test_signalled_error_beats_a_concurrent_timeout(tmp_path):
    """Precedence: a failed write in the same tick as a timeout is an error."""
    fake = FakeRecorder(tmp_path, max_duration=0.01)
    fake._stop_event.clear()  # let the loop actually run and hit its timeout
    fake.signal_stop(TERMINATION_ERROR, "Write failed on /scan: boom")
    handle = FakeGoalHandle()
    result = _record(fake, "p", handle)
    assert result.termination_reason == TERMINATION_ERROR
    assert handle.calls == ["abort"]


def test_timeout_reports_timeout_and_succeeds(tmp_path):
    fake = FakeRecorder(tmp_path, max_duration=0.01)
    fake._stop_event.clear()  # nobody signals; the loop decides on its own
    handle = FakeGoalHandle()
    result = _record(fake, "p", handle)
    assert handle.calls == ["succeed"]
    assert result.termination_reason == TERMINATION_TIMEOUT
    assert fake.stop_reason is None  # loop-decided, not signalled
    _assert_reconciled(fake)


def test_feedback_reports_elapsed_and_message_count(tmp_path):
    """Feedback answers "is it running and is data flowing". Time remaining is
    not reported: a client that set max_duration_s on the goal can subtract,
    and a field that needed a -1 sentinel to describe the common case
    (unlimited) was worse than not having it."""
    fake = FakeRecorder(tmp_path)
    fake._stop_event.clear()
    handle = FakeGoalHandle(on_feedback=fake._stop_event.set)
    _record(fake, "p", handle)
    assert len(handle.feedback) == 1
    assert handle.feedback[0].elapsed_s > 0.0
    assert handle.feedback[0].messages_written == 7


def test_unexpected_error_before_open_aborts_cleanly(tmp_path):
    # feedback_rate_hz=0 is unrepresentable via ROS parameters
    # (positive_rate_descriptor), but any unexpected pre-open exception must
    # still abort the goal and reconcile state.
    fake = FakeRecorder(tmp_path, feedback_rate=0.0)  # 1/0 raises in the loop setup
    handle = FakeGoalHandle()
    result = _record(fake, "p", handle)
    assert handle.calls == ["abort"]
    assert result.termination_reason == TERMINATION_ERROR
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


def test_stop_state_untouched_at_busy_release(tmp_path):
    """A new accept arms a fresh event and clears the reason the moment _busy is
    released; _record must never write either, or it could clobber the new
    claim's state."""
    fake = FakeRecorder(tmp_path)
    claimed_event = fake._stop_event
    fake.signal_stop(TERMINATION_CANCELLED)
    _record(fake, "p")
    assert fake._busy is False
    assert fake._stop_event is claimed_event
    assert fake._stop_reason == TERMINATION_CANCELLED  # read, never rewritten


def test_delete_during_recording_discards_it_when_it_ends(tmp_path):
    fake = FakeRecorder(tmp_path)
    fake._stop_event.clear()
    previous = tmp_path / "previous"
    previous.mkdir()
    fake._last_bag_dir = previous
    responses = []

    def delete_then_stop():
        responses.append(_delete_last_bag(fake))
        fake.signal_stop(TERMINATION_CANCELLED)

    _record(fake, "p", FakeGoalHandle(on_feedback=delete_then_stop))
    assert responses[0].success is True
    assert not fake._bag_dir.exists()
    # The discarded bag was the last one; a second delete must not reach the older bag.
    assert fake._last_bag_dir is None
    assert previous.exists()
    _assert_reconciled(fake)


def test_delete_when_idle_removes_the_last_bag(tmp_path):
    fake = FakeRecorder(tmp_path)
    _record(fake, "p")
    response = _delete_last_bag(fake)
    assert response.success is True
    assert not fake._bag_dir.exists()
    assert fake._last_bag_dir is None


def test_delete_with_no_bag_fails(tmp_path):
    response = _delete_last_bag(FakeRecorder(tmp_path))
    assert response.success is False
    assert response.message == "No bag to delete"


def test_service_start_publishes_status_and_its_result(tmp_path):
    """A service-started recording has no goal, so the topics are how a client follows it."""
    fake = FakeRecorder(tmp_path, feedback_rate=200.0, max_duration=0.03)
    fake._stop_event.clear()
    result = _record(fake, "p")
    assert fake._status_pub.published, "no status while recording"
    assert all(s.messages_written == 7 for s in fake._status_pub.published)
    (published,) = fake._result_pub.published
    fields = ("termination_reason", "message", "bag_path", "messages_written")
    assert [getattr(published, f) for f in fields] == [getattr(result, f) for f in fields]


def test_failed_recording_still_publishes_its_result(tmp_path):
    fake = FakeRecorder(tmp_path, open_error=RuntimeError("disk full"))
    _record(fake, "p", FakeGoalHandle())
    (published,) = fake._result_pub.published
    assert published.termination_reason == TERMINATION_ERROR
    assert "disk full" in published.message
