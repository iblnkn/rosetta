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

Runs the real EpisodeRecorderNode with only the rosbag2 writer stubbed, after a
real work-gate claim. Every path (success, open failure, metadata failure,
cancel, failed bag write, an error mid-recording) must release the busy guard,
leave the claim's stop event alone, end an action goal via an rcl-legal
transition, and report why it ended.

The reason precedence _record implements, and which these tests pin:

    an exception on this thread   >  a stop somebody signalled  >
    the loop's own decision       >  a plain stop

Signalled beats loop-decided so a failed bag write landing in the same tick as
a timeout still reports as an error rather than a clean finish.
"""

import pytest
import yaml
from node_helpers import FakeGoalHandle, FakePublisher, claim
from rclpy.parameter import Parameter
from std_srvs.srv import Trigger

from rosetta_ros.bag_metadata import BAG_CUSTOM_DATA_KEY, BAG_GOAL_ID_KEY, BAG_METADATA_KEY, BAG_PROMPT_KEY
from rosetta_ros.node_utils import (
    TERMINATION_CANCELLED,
    TERMINATION_ERROR,
    TERMINATION_NODE_DEACTIVATED,
    TERMINATION_STOPPED,
    TERMINATION_TIMEOUT,
)
from rosetta_ros.nodes.episode_recorder_node import EpisodeRecorderNode


class _Recorder(EpisodeRecorderNode):
    """The real node, its rosbag2 writer replaced by a directory.

    The stub writer "writes" 7 messages, and on close writes the metadata.yaml
    rosbag2 would (unless ``metadata_error``), so provenance has a file to land in.
    """

    def __init__(self, bag_dir, *, open_error=None, metadata_error=False, **params):
        params.setdefault("feedback_rate_hz", 50.0)
        super().__init__(parameter_overrides=[Parameter(k, value=v) for k, v in params.items()])
        self._status_pub, self._result_pub = FakePublisher(), FakePublisher()
        self.bag_dir = bag_dir
        self._open_error = open_error
        self._metadata_error = metadata_error
        self.close_calls = 0

    def _create_bag_dir(self):
        return self.bag_dir

    def _open_writer(self, bag_dir):
        if self._open_error is not None:
            raise self._open_error
        bag_dir.mkdir(parents=True)
        self._messages_written = 7
        return []

    def _close_writer(self):
        self.close_calls += 1
        if self.bag_dir.exists() and not self._metadata_error:
            (self.bag_dir / "metadata.yaml").write_text(yaml.safe_dump({BAG_METADATA_KEY: {}}))

    def custom_data(self) -> dict:
        meta = yaml.safe_load((self.bag_dir / "metadata.yaml").read_text())
        return meta[BAG_METADATA_KEY].get(BAG_CUSTOM_DATA_KEY, {})


@pytest.fixture
def make_recorder(rclpy_ctx, tmp_path):
    """Build a recorder. By default a recording is claimed and already told to
    stop, so the record loop exits on its first wait; ``running=True`` leaves
    the stop event clear, ``idle=True`` claims nothing."""
    nodes = []

    def make(*, running=False, idle=False, **kwargs):
        node = _Recorder(tmp_path / "ep", **kwargs)
        nodes.append(node)
        if not idle:
            claim(node)
            node.claimed_event = node._stop_event
            if not running:
                node._stop_event.set()
        return node

    yield make
    for node in nodes:
        node.destroy_node()


def _delete_last_bag(node):
    return node._on_delete_last_bag_service(None, Trigger.Response())


def _assert_reconciled(node):
    assert not node.busy
    assert node._stop_event is node.claimed_event  # _record never touches the claim's event
    assert node.close_calls >= 1


def test_clean_stop_succeeds_and_reconciles(make_recorder):
    rec = make_recorder()
    handle = FakeGoalHandle(goal_uuid=bytes(range(16)))
    result = rec._record("pick up cube", handle)
    assert handle.calls == ["succeed"]
    assert result.termination_reason == TERMINATION_STOPPED
    assert result.messages_written == 7
    # message carries only what no other field does. A clean end has nothing to
    # add, so it stays empty rather than restating messages_written.
    assert result.message == ""
    # Provenance: the bag names the goal that produced it, in canonical hex.
    assert rec.custom_data()[BAG_GOAL_ID_KEY] == "00010203-0405-0607-0809-0a0b0c0d0e0f"
    _assert_reconciled(rec)


def test_service_start_reports_its_reason_and_records_no_goal_id(make_recorder):
    rec = make_recorder()
    result = rec._record("pick up cube")
    # The service path gets a fully populated reason too: _record's return value
    # is uniform across both start paths.
    assert result.termination_reason == TERMINATION_STOPPED
    assert result.messages_written == 7
    # No goal, so no goal id to record -- empty, never a fabricated one.
    assert BAG_GOAL_ID_KEY not in rec.custom_data()
    _assert_reconciled(rec)


@pytest.mark.parametrize(
    ("node_default", "goal_max_duration"),
    [
        # The point of the goal field: a timed recording without a `ros2 param set`
        # first, and without racing another client's parameter write.
        (0.0, 0.01),
        # The node default applies when the goal leaves max_duration unset.
        (0.01, 0.0),
    ],
    ids=["goal", "node_default"],
)
def test_timeout_reports_timeout_and_succeeds(make_recorder, node_default, goal_max_duration):
    rec = make_recorder(running=True, default_max_duration_s=node_default)
    handle = FakeGoalHandle()
    result = rec._record("p", handle, max_duration=goal_max_duration)
    assert handle.calls == ["succeed"]
    assert result.termination_reason == TERMINATION_TIMEOUT
    _assert_reconciled(rec)


def test_open_writer_failure_aborts_without_metadata(make_recorder):
    rec = make_recorder(open_error=RuntimeError("disk full"))
    handle = FakeGoalHandle()
    result = rec._record("p", handle)
    assert handle.calls == ["abort"]
    assert result.termination_reason == TERMINATION_ERROR
    assert "disk full" in result.message
    assert not rec.bag_dir.exists()  # writer never opened: nothing to annotate
    assert _delete_last_bag(rec).success is False  # no bag dir on disk
    _assert_reconciled(rec)


def test_metadata_failure_fails_result_but_keeps_counts_and_bag(make_recorder):
    rec = make_recorder(metadata_error=True)
    handle = FakeGoalHandle()
    result = rec._record("p", handle)
    assert handle.calls == ["abort"]
    assert result.termination_reason == TERMINATION_ERROR
    assert "metadata.yaml" in result.message
    assert result.messages_written == 7
    _assert_reconciled(rec)
    # The partial bag exists and must stay deletable via ~/delete_last_bag.
    assert _delete_last_bag(rec).success is True
    assert not rec.bag_dir.exists()


def test_error_mid_recording_aborts_and_keeps_the_bag(make_recorder):
    rec = make_recorder(running=True)

    def publish_fails(_msg):
        raise RuntimeError("publisher gone")

    rec._status_pub.publish = publish_fails
    handle = FakeGoalHandle()
    result = rec._record("p", handle)
    assert handle.calls == ["abort"]
    assert result.termination_reason == TERMINATION_ERROR
    assert "publisher gone" in result.message
    # What was written before the error is still a bag, annotated and deletable.
    assert rec.custom_data()[BAG_PROMPT_KEY] == "p"
    _assert_reconciled(rec)
    assert _delete_last_bag(rec).success is True
    assert not rec.bag_dir.exists()


def test_action_cancel_finishes_canceled(make_recorder):
    # What the base's on_cancel does, including for the cancels that
    # ~/cancel_recording forwards to the action server's own cancel service.
    rec = make_recorder()
    rec.signal_stop(TERMINATION_CANCELLED)
    handle = FakeGoalHandle(cancel_requested=True)
    result = rec._record("p", handle)
    assert handle.calls == ["canceled"]
    assert result.termination_reason == TERMINATION_CANCELLED
    # A cancelled recording still hands back its bag: CANCELED says who ended
    # the work, not that anything was lost.
    assert result.messages_written == 7
    assert result.bag_path == str(rec.bag_dir)
    # The reason is read, never rewritten: a new claim owns it once busy drops.
    assert rec.stop_reason == TERMINATION_CANCELLED
    _assert_reconciled(rec)


@pytest.mark.parametrize(
    ("reason", "detail"),
    [
        # The write exception is caught inside the subscription callback and
        # never reaches _record's try/except, so the reason it signals is the
        # only thing standing between a failed write and a SUCCEEDED goal.
        (TERMINATION_ERROR, "Write failed on /camera/image: no space left on device"),
        # The server chose to stop the goal, which the ROS 2 action docs call
        # an abort. The reason is what tells it apart from a genuine error.
        (TERMINATION_NODE_DEACTIVATED, ""),
    ],
)
def test_a_signalled_stop_aborts_with_its_reason(make_recorder, reason, detail):
    rec = make_recorder()
    rec.signal_stop(reason, detail)
    handle = FakeGoalHandle()
    result = rec._record("p", handle)
    assert handle.calls == ["abort"]
    assert (result.termination_reason, result.message) == (reason, detail)
    assert result.messages_written == 7  # the partial bag is still reported
    _assert_reconciled(rec)


def test_signalled_error_beats_a_concurrent_timeout(make_recorder):
    """Precedence: a failed write latched in the same tick as a timeout is an error.

    The reason is latched without setting the stop event, so the loop runs on
    and reaches its own timeout with the error already recorded.
    """
    rec = make_recorder(running=True)
    with rec._work_gate:
        rec._stop_reason, rec._stop_detail = TERMINATION_ERROR, "Write failed on /scan: boom"
    handle = FakeGoalHandle()
    result = rec._record("p", handle, max_duration=0.01)
    assert result.termination_reason == TERMINATION_ERROR
    assert result.message == "Write failed on /scan: boom"
    assert handle.calls == ["abort"]


def test_feedback_reports_elapsed_and_message_count(make_recorder):
    rec = make_recorder(running=True)
    handle = FakeGoalHandle(on_feedback=rec._stop_event.set)
    rec._record("p", handle)
    assert len(handle.feedback) == 1
    assert handle.feedback[0].elapsed_s > 0.0
    assert handle.feedback[0].messages_written == 7


def test_delete_during_recording_discards_it_when_it_ends(make_recorder, tmp_path):
    rec = make_recorder(running=True)
    previous = tmp_path / "previous"
    previous.mkdir()
    rec._last_bag_dir = previous
    responses = []

    def delete_then_stop():
        responses.append(_delete_last_bag(rec))
        rec.signal_stop(TERMINATION_CANCELLED)

    rec._record("p", FakeGoalHandle(on_feedback=delete_then_stop))
    assert responses[0].success is True
    assert not rec.bag_dir.exists()
    # The discarded bag was the last one; a second delete must not reach the older bag.
    assert _delete_last_bag(rec).success is False
    assert previous.exists()
    _assert_reconciled(rec)


def test_delete_when_idle_removes_the_last_bag(make_recorder):
    rec = make_recorder()
    rec._record("p")
    response = _delete_last_bag(rec)
    assert response.success is True
    assert not rec.bag_dir.exists()
    assert _delete_last_bag(rec).success is False  # only the last bag, never an older one


def test_delete_with_no_bag_fails(make_recorder):
    response = _delete_last_bag(make_recorder(idle=True))
    assert response.success is False
    assert response.message == "No bag to delete"


def _claimed_after(make_recorder, tmp_path, **kwargs):
    """An idle recorder with a finished bag on disk, then a new recording claimed."""
    rec = make_recorder(idle=True, **kwargs)
    previous = tmp_path / "previous"
    previous.mkdir()
    rec._last_bag_dir = previous
    claim(rec)
    rec._stop_event.set()
    return rec, previous


def test_delete_between_accept_and_start_keeps_the_previous_bag(make_recorder, tmp_path):
    rec, previous = _claimed_after(make_recorder, tmp_path)
    response = _delete_last_bag(rec)
    assert response.success is False
    assert "starting" in response.message
    assert previous.exists()


def test_delete_after_a_recording_with_no_bag_keeps_the_previous_bag(make_recorder, tmp_path):
    rec, previous = _claimed_after(make_recorder, tmp_path, open_error=RuntimeError("disk full"))
    rec._record("p", FakeGoalHandle())
    response = _delete_last_bag(rec)
    assert response.success is False
    assert previous.exists()


@pytest.mark.parametrize(
    ("open_error", "reason", "statuses"),
    [(None, TERMINATION_STOPPED, 1), (RuntimeError("disk full"), TERMINATION_ERROR, 0)],
    ids=["clean", "failed"],
)
def test_service_start_publishes_status_and_its_result(make_recorder, open_error, reason, statuses):
    """A service-started recording has no goal, so the topics are how a client follows it."""
    rec = make_recorder(running=True, feedback_rate_hz=200.0, open_error=open_error)
    rec._status_pub = FakePublisher(on_publish=rec._stop_event.set)
    result = rec._record("p")
    assert len(rec._status_pub.published) == statuses  # a recording that never opened has no status
    assert all(status.messages_written == 7 for status in rec._status_pub.published)
    (published,) = rec._result_pub.published
    assert published.termination_reason == reason
    fields = ("termination_reason", "message", "bag_path", "messages_written")
    assert [getattr(published, f) for f in fields] == [getattr(result, f) for f in fields]
