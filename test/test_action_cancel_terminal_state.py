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

"""Live-graph proof of the terminal-state contract, read off the wire.

Every other action-plane test in this package drives node callbacks directly
with a hand-rolled fake goal handle. A fake handle records which transition was
*requested*; it cannot show that the transition was legal, nor what terminal
GoalStatus a client actually receives. That is the whole claim here, so this
test builds a real ActionServer, a real ActionClient, and asserts the status
that comes back from ``get_result_async``.

The contract, in one line each:

    a client took the work away    -> CANCELED
    the work reached a defined end -> SUCCEEDED
    the server stopped the work    -> ABORTED

The load-bearing case is the first one. ``~/cancel_recording`` and friends exist
because Foxglove can call services but not actions; they forward to the action
server's own ``_action/cancel_goal`` so that a dashboard button and a `ros2
action` cancel are the same event down to the terminal state. Faking it with a
bare stop signal used to end the goal ABORTED, so the same human gesture
produced two different terminal states depending on which button was pressed.

Deliberately NOT EpisodeRecorderNode: that needs a contract file, a bag
directory, and a live topic graph, and a failure in any of those would be
indistinguishable from a failure of the mechanism under test. This uses the real
base class and the real helpers, with a work loop reduced to an event wait.
"""

import threading
import time

import pytest
import rclpy
from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from rclpy.action import ActionClient, ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from rosetta.robots.ros2.nodes.node_utils import (
    TERMINATION_ERROR,
    TERMINATION_TIMEOUT,
    finish_goal,
)
from rosetta.robots.ros2.rosetta_lifecycle_node import RosettaLifecycleNode
from rosetta_interfaces.action import RecordEpisode

# Distinctive on purpose. This workspace has no ROS_DOMAIN_ID isolation, so a
# probe named `record_episode` would match a recorder running on the developer's
# machine. Relative, so it exercises the same name expansion the cancel client
# relies on.
ACTION_NAME = "rosetta_cancel_probe_action"


class _ProbeNode(RosettaLifecycleNode):
    """The real base class and helpers, with the work loop reduced to a wait."""

    def __init__(self):
        super().__init__("rosetta_cancel_probe")
        self._cbg = ReentrantCallbackGroup()
        self._server = None
        self._cancel_client = None
        #: "wait" blocks until somebody stops it; "timeout" ends on its own;
        #: "error" fails immediately.
        self.mode = "wait"
        self.started = threading.Event()

    def _setup(self):
        self._server = ActionServer(
            self,
            RecordEpisode,
            ACTION_NAME,
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
            callback_group=self._cbg,
        )
        self._cancel_client = self.create_client(
            CancelGoal,
            f"{ACTION_NAME}/_action/cancel_goal",
            callback_group=self._cbg,
        )

    def _teardown(self):
        if self._server is not None:
            self._server.destroy()
            self._server = None
        if self._cancel_client is not None:
            self.destroy_client(self._cancel_client)
            self._cancel_client = None

    def _execute(self, goal_handle):
        result = RecordEpisode.Result()
        with self._goal_work(goal_handle) as stop_event:
            self.started.set()
            if self.mode == "error":
                result.termination_reason = TERMINATION_ERROR
            elif self.mode == "timeout":
                stop_event.wait(0.05)  # nobody signals; the loop decides
                result.termination_reason = TERMINATION_TIMEOUT
            else:
                stop_event.wait(10.0)
                result.termination_reason = self.stop_reason or TERMINATION_TIMEOUT
            result.message = "probe"
            finish_goal(goal_handle, result)
            return result


@pytest.fixture
def graph(rclpy_ctx):
    """A spinning two-node graph: the probe server and a client to drive it."""
    node = _ProbeNode()
    client_node = rclpy.create_node("rosetta_cancel_probe_client")
    # MultiThreadedExecutor, spun on its own thread: the execute callback blocks
    # for the length of the goal, so a single worker (or spin_once on the test
    # thread) could never process the cancel that ends it. 4 threads matches
    # what spin_lifecycle_node runs in production.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.add_node(client_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    node.trigger_configure()
    node.trigger_activate()
    client = ActionClient(client_node, RecordEpisode, ACTION_NAME)
    assert client.wait_for_server(timeout_sec=10.0), "probe action server never appeared"
    try:
        yield node, client
    finally:
        # Release the execute callback first: a failed assertion must not leave
        # it holding an executor worker into the next test.
        node._signal_stop(TERMINATION_TIMEOUT)
        client.destroy()
        executor.shutdown(timeout_sec=5.0)
        spin_thread.join(timeout=5.0)
        node.destroy_node()
        client_node.destroy_node()


def _send_goal(client):
    send_future = client.send_goal_async(RecordEpisode.Goal(prompt="probe"))
    goal_handle = _result_of(send_future)
    assert goal_handle.accepted
    return goal_handle


def _result_of(future, timeout_s: float = 10.0):
    deadline = time.monotonic() + timeout_s
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert future.done(), "future never completed"
    return future.result()


def _terminal(goal_handle):
    """The status a client actually receives -- the thing under test."""
    return _result_of(goal_handle.get_result_async()).status


def test_service_cancel_ends_canceled(graph):
    """THE claim: a Foxglove-style service cancel is a real cancel."""
    node, client = graph
    goal_handle = _send_goal(client)
    assert node.started.wait(5.0)

    # Exactly what ~/cancel_recording, ~/cancel_episode and ~/cancel_policy do.
    node._cancel_current_work(node._cancel_client)

    assert _terminal(goal_handle) == GoalStatus.STATUS_CANCELED


def test_action_protocol_cancel_ends_canceled(graph):
    """Baseline. If this ever diverges from the test above, the fix regressed:
    the two paths must land on the same status, not merely each on 'a' status."""
    node, client = graph
    goal_handle = _send_goal(client)
    assert node.started.wait(5.0)

    cancel_response = _result_of(goal_handle.cancel_goal_async())
    assert cancel_response.return_code == CancelGoal.Response.ERROR_NONE

    assert _terminal(goal_handle) == GoalStatus.STATUS_CANCELED


def test_work_reaching_its_own_end_succeeds(graph):
    """Guards the over-correction where every stop becomes a cancel."""
    node, client = graph
    node.mode = "timeout"
    goal_handle = _send_goal(client)

    assert _terminal(goal_handle) == GoalStatus.STATUS_SUCCEEDED


def test_error_ends_aborted(graph):
    """ABORTED must keep meaning 'something went wrong'."""
    node, client = graph
    node.mode = "error"
    goal_handle = _send_goal(client)

    assert _terminal(goal_handle) == GoalStatus.STATUS_ABORTED


def test_deactivate_ends_aborted_naming_the_deactivate(graph):
    """A lifecycle deactivate is the SERVER choosing to stop the goal, which the
    ROS 2 action docs define as an abort. termination_reason is what keeps it
    distinguishable from a genuine error."""
    node, client = graph
    goal_handle = _send_goal(client)
    assert node.started.wait(5.0)

    node.trigger_deactivate()

    wrapped = _result_of(goal_handle.get_result_async())
    assert wrapped.status == GoalStatus.STATUS_ABORTED
    assert wrapped.result.termination_reason == "node_deactivated"


def test_cancel_with_no_goal_in_flight_is_harmless(graph):
    """The service-start path has no goal handle for a cancel request to match,
    so _cancel_current_work signals the reason directly instead."""
    node, _client = graph
    node._accepting_work = True
    assert node._try_claim_work() is None
    try:
        node._cancel_current_work(node._cancel_client)
        assert node.stop_reason == "cancelled"
    finally:
        node._busy = False
