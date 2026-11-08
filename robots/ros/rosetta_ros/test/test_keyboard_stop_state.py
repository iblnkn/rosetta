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

"""The keyboard drives the recorder's RecordEpisode action and reports each goal's own result.

A real keyboard node talks to a stand-in recorder: an action server for
``/record_episode`` and a ``delete_last_bag`` service, as the real recorder
serves them by default. Keys are pressed by calling the keyboard's key
handlers, and the test reads what the keyboard prints.

The stand-in holds each result until the test releases it, so the tests can
put a key press between a stop and the recording's result, the window in which
the service-based keyboard reported "Saved" for a deleted bag.
"""

import threading
import time

import pytest
import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger

from rosetta_interfaces.action import RecordEpisode
from rosetta_ros.node_utils import TERMINATION_CANCELLED, TERMINATION_ERROR, TERMINATION_TIMEOUT
from rosetta_ros.nodes.episode_keyboard_node import EpisodeKeyboardNode


class _FakeRecorder:
    """Serves RecordEpisode and delete_last_bag the way the recorder does, one bag per goal.

    A goal runs until it is cancelled or ``time_out()`` ends it, then waits for
    ``release()`` before returning its result, as the recorder does while it
    closes the bag. As in the recorder, delete_last_bag during a recording
    marks that recording's bag for deletion when it ends.
    """

    def __init__(self):
        self.node = rclpy.create_node("episode_recorder")
        group = ReentrantCallbackGroup()
        self._server = ActionServer(
            self.node,
            RecordEpisode,
            "/record_episode",
            self._execute,
            cancel_callback=lambda _goal: CancelResponse.REJECT if self.refuse_cancel else CancelResponse.ACCEPT,
            callback_group=group,
        )
        self.node.create_service(Trigger, "~/delete_last_bag", self._delete, callback_group=group)
        self._timed_out = threading.Event()
        self._released = threading.Event()
        self._released.set()  # results return at once unless a test holds them
        self._closing = False
        self.refuse_cancel = False
        self.refuse_delete = False  # as the recorder does between accepting a goal and opening its bag
        self.fail_with = ""  # end the next recording in an error with this message
        self.bags = 0
        self.active = ""  # the recording in progress
        self.discard_active = False
        self.last_bag = ""
        self.deleted = []
        self.delete_calls = 0

    def time_out(self):
        self._timed_out.set()

    def hold(self):
        self._released.clear()

    def release(self):
        self._released.set()

    def close(self):
        """End any goal still running, so the executor can shut down."""
        self._closing = True
        self._timed_out.set()
        self._released.set()

    def _execute(self, goal_handle):
        self.bags += 1
        self.active = bag = f"bags/{self.bags}"
        while not goal_handle.is_cancel_requested and not self._timed_out.wait(0.01):
            pass
        if not self._closing:
            self._timed_out.clear()  # here, not at the start: time_out() may come first
        self._released.wait(timeout=10.0)
        self.active = ""
        if self.discard_active:
            self.discard_active = False
            self.deleted.append(bag)
        else:
            self.last_bag = bag
        result = RecordEpisode.Result(bag_path=bag, messages_written=5)
        if self.fail_with:
            result.termination_reason, result.message = TERMINATION_ERROR, self.fail_with
            goal_handle.abort()
        elif goal_handle.is_cancel_requested:
            result.termination_reason = TERMINATION_CANCELLED
            goal_handle.canceled()
        else:
            result.termination_reason = TERMINATION_TIMEOUT
            goal_handle.succeed()
        return result

    def _delete(self, _request, response):
        self.delete_calls += 1
        if self.refuse_delete:
            response.success, response.message = False, "A recording is starting; it has no bag yet"
            return response
        if self.active:
            self.discard_active = True
            response.success = True
            response.message = f"Will delete {self.active} when the recording ends"
            return response
        response.success = bool(self.last_bag)
        response.message = f"Deleted: {self.last_bag}" if self.last_bag else "No bag to delete"
        if self.last_bag:
            self.deleted.append(self.last_bag)
        self.last_bag = ""
        return response


class _NoKeys:
    def read_bytes(self):
        return b""


@pytest.fixture
def rig(capsys):
    """A stand-in recorder, a keyboard and a reader of what the keyboard printed."""
    recorder = _FakeRecorder()
    keyboard = EpisodeKeyboardNode(_NoKeys())
    executor = MultiThreadedExecutor()
    executor.add_node(recorder.node)
    executor.add_node(keyboard)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()

    printed = []

    def wait_for(text, count=1, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            printed.append(capsys.readouterr().out)
            if "".join(printed).count(text) >= count:
                return "".join(printed)
            time.sleep(0.01)
        raise AssertionError(f"keyboard never printed {text!r}; it printed:\n{''.join(printed)}")

    deadline = time.monotonic() + 10.0
    while not keyboard._action_client.server_is_ready() and time.monotonic() < deadline:
        time.sleep(0.01)
    rig = type("Rig", (), {})()
    rig.recorder, rig.keyboard, rig.wait_for = recorder, keyboard, wait_for
    rig.start = lambda: (keyboard._handle_start(), wait_for("Recording started"))

    def start_again():
        """Press r until a second recording starts, which means the first one's result was handled."""
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            keyboard._handle_start()
            try:
                return wait_for("Recording started", count=2, timeout=0.2)
            except AssertionError:
                continue
        raise AssertionError("a second recording never started")

    rig.start_again = start_again
    yield rig
    recorder.close()
    executor.shutdown(timeout_sec=5.0)
    spinner.join(timeout=5.0)
    keyboard.destroy_node()
    recorder.node.destroy_node()


def test_a_recording_that_hits_its_time_limit_is_reported_and_ends(rig):
    # No key is pressed. The service-based keyboard printed nothing and stayed in REC.
    rig.start()
    rig.recorder.time_out()
    rig.wait_for("Saved 5 messages (time limit): bags/1")
    rig.start_again()  # a new episode starts: the last one is over


def test_s_reports_the_saved_bag_without_a_reason(rig):
    rig.start()
    rig.keyboard._handle_stop()
    assert "Saved 5 messages: bags/1" in rig.wait_for("Saved")


@pytest.mark.parametrize("stop_first", [False, True], ids=["d-while-recording", "s-then-d"])
def test_d_before_the_result_deletes_the_bag_without_reporting_it_saved(rig, stop_first):
    # The result is held as the recorder holds it while closing the bag, so d
    # lands before it. Only the deletion is reported, and the deleted bag is
    # the one that recording wrote.
    rig.start()
    rig.recorder.hold()
    if stop_first:
        rig.keyboard._handle_stop()
    rig.keyboard._handle_discard()
    rig.wait_for("Discarded: Will delete bags/1 when the recording ends")
    rig.recorder.release()
    out = rig.start_again()  # the discarded recording's result has been handled
    assert "Saved" not in out
    assert rig.recorder.deleted == ["bags/1"]


def test_d_after_the_result_deletes_the_last_bag(rig):
    rig.start()
    rig.keyboard._handle_stop()
    rig.wait_for("Saved")
    rig.keyboard._handle_discard()
    rig.wait_for("Discarded: Deleted: bags/1")


def test_a_refused_stop_leaves_the_keyboard_recording(rig):
    # The recorder is still recording, so s must be pressable again.
    rig.start()
    rig.recorder.refuse_cancel = True
    rig.keyboard._handle_stop()
    rig.wait_for("Stop failed")
    rig.recorder.refuse_cancel = False
    rig.keyboard._handle_stop()
    rig.wait_for("Saved 5 messages: bags/1")


def test_a_recording_that_fails_while_being_discarded_still_shows_its_error(rig):
    rig.start()
    rig.recorder.hold()
    rig.recorder.fail_with = "Write failed on /cam: disk full"
    rig.keyboard._handle_discard()
    rig.recorder.release()
    rig.wait_for("Recording failed: Write failed on /cam: disk full")


def test_a_refused_discard_still_reports_the_kept_bag_and_d_deletes_it(rig):
    # d pressed right after r: the recorder has accepted the goal but has no
    # bag yet, so it refuses the delete. The stop still goes through, and the
    # bag it keeps must be reported and stay discardable.
    rig.start()
    rig.recorder.refuse_delete = True
    rig.keyboard._handle_discard()
    rig.wait_for("Delete failed: A recording is starting; it has no bag yet")
    rig.wait_for("Saved 5 messages: bags/1")
    rig.recorder.refuse_delete = False
    rig.keyboard._handle_discard()
    rig.wait_for("Discarded: Deleted: bags/1")
    assert rig.recorder.deleted == ["bags/1"]


def test_d_with_a_refused_stop_still_deletes_the_bag(rig):
    # The recorder accepts the delete, then refuses the stop. The recording
    # goes on, and its bag is still deleted when s ends it.
    rig.start()
    rig.recorder.refuse_cancel = True
    rig.keyboard._handle_discard()
    rig.wait_for("Stop failed")
    rig.recorder.refuse_cancel = False
    rig.keyboard._handle_stop()
    out = rig.start_again()  # the discarded recording's result has been handled
    assert "Saved" not in out
    assert rig.recorder.deleted == ["bags/1"]


def test_a_second_d_before_the_result_sends_no_second_delete(rig):
    # Once the recording has ended, a second delete would find no bag, fail,
    # and make the keyboard report the deleted bag saved.
    rig.start()
    rig.recorder.hold()
    rig.keyboard._handle_stop()
    rig.keyboard._handle_discard()
    rig.wait_for("Discarded: Will delete bags/1 when the recording ends")
    rig.keyboard._handle_discard()
    rig.wait_for("Already discarding this episode")
    rig.recorder.release()
    out = rig.start_again()
    assert "Saved" not in out
    assert rig.recorder.deleted == ["bags/1"]
    assert rig.recorder.delete_calls == 1
