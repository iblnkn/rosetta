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

"""RosettaLifecycleNode/BridgeLifecycleNode: transition discipline and safety ordering.

The deactivate test pins the one line whose order is load-bearing:
send_safety_action() runs BEFORE super().on_deactivate() disables the
lifecycle publishers. Swapped, the safety publish becomes a silent no-op and
nothing else fails.

The error-path tests pin on_configure's fail-loud contract (rclpy swallows
callback exceptions with no log, so the callback must catch, log, and return
ERROR itself) and on_error's honesty: recovery to unconfigured only when
teardown actually cleaned up; FAILURE (finalized) when it could not.

The base-level tests pin the work gate: shutdown-from-active stops work and
secures BEFORE teardown, cleanup refuses while work runs, and a claim always
leaves behind a stop event that _stop_and_secure() sets.
"""

import threading
import time

import numpy as np
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import TransitionCallbackReturn
from sensor_msgs.msg import JointState

from rosetta.contract.model import SafetyBehavior
from rosetta.contract.schema import Align, Channel, Source
from rosetta.contract.specs import ActionStreamSpec
from rosetta.robots.ros2.rclpy_utils import lifecycle_state_label
from rosetta.robots.ros2.rosetta_lifecycle_node import BridgeLifecycleNode, RosettaLifecycleNode


def _act_spec(topic: str, names: list[str], safety: SafetyBehavior) -> ActionStreamSpec:
    return ActionStreamSpec(
        key="action",
        names=list(names),
        fps=1,
        source=Source(
            channel=Channel(topic=topic, type="sensor_msgs/msg/JointState", safety=safety),
            align=Align("hold", "receive"),
        ),
        dtype="float64",
    )


def _spin_until(executor, predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.05)
        if predicate():
            return True
    return False


@pytest.fixture
def node():
    # fps=1 keeps the watchdog period at 2s, far beyond the test's runtime,
    # so any safety frame observed here can only come from on_deactivate.
    n = BridgeLifecycleNode(
        "lifecycle_node_under_test",
        observation_specs=[],
        action_specs=[_act_spec("/lifecycle_cmd", ["position.j1", "position.j2"], SafetyBehavior.ZEROS)],
        fps=1,
    )
    yield n
    n.destroy_node()


def test_deactivate_publishes_safety_action_and_disarms_watchdog(rclpy_ctx, node):
    sub_node = rclpy.create_node("lifecycle_fixture_sub")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(sub_node)
    received: list[list[float]] = []
    try:
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        assert node.is_active

        sub_node.create_subscription(JointState, "/lifecycle_cmd", lambda m: received.append(list(m.position)), 10)

        # Re-publish while polling: proves the wire is up (subscription
        # matched) before deactivating, without sleeping on discovery.
        def policy_frame_received():
            node.bridge.publish_frame({"action": np.array([1.0, 2.0])})
            return [1.0, 2.0] in received

        assert _spin_until(executor, policy_frame_received)

        assert node.trigger_deactivate() == TransitionCallbackReturn.SUCCESS

        # The zeros frame was published during on_deactivate, while the
        # lifecycle publisher was still active; it must reach the wire.
        assert _spin_until(executor, lambda: [0.0, 0.0] in received), (
            f"safety frame never arrived; received: {received}"
        )
        # Securing disarms the watchdog until the next published frame.
        assert node.bridge._last_action_ns is None
    finally:
        executor.remove_node(node)
        executor.remove_node(sub_node)
        sub_node.destroy_node()


def test_configure_failure_recovers_to_unconfigured_after_teardown(rclpy_ctx, node):
    """A partially-built bridge is torn down (via on_error) and reconfigurable."""

    def boom():
        raise RuntimeError("boom")

    # Raises at the END of bridge.setup(): subscriptions/publishers already
    # exist, so recovery genuinely depends on on_error tearing them down
    # (setup() itself refuses to run twice without a teardown).
    node.bridge._should_use_watchdog = boom
    assert node.trigger_configure() != TransitionCallbackReturn.SUCCESS
    assert lifecycle_state_label(node) == "unconfigured"
    assert not node.is_configured
    assert node.bridge._node is None  # partial setup was torn down

    del node.bridge._should_use_watchdog
    assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
    assert node.is_configured
    assert node.trigger_cleanup() == TransitionCallbackReturn.SUCCESS


def test_configure_failure_with_broken_teardown_finalizes(rclpy_ctx, node):
    """When error-handling teardown itself fails, the node must not claim recovery."""

    def setup_boom(_node):
        raise RuntimeError("setup boom")

    def teardown_boom():
        raise RuntimeError("teardown boom")

    node.bridge.setup = setup_boom
    node.bridge.teardown = teardown_boom
    assert node.trigger_configure() != TransitionCallbackReturn.SUCCESS
    assert lifecycle_state_label(node) == "finalized"


# -------------------- Base work-gate semantics --------------------


class _RecordingNode(RosettaLifecycleNode):
    """Minimal subclass recording hook calls, for base-transition tests."""

    def __init__(self, name):
        super().__init__(name)
        self.calls: list[str] = []

    def _setup(self):
        self.calls.append("setup")

    def _teardown(self):
        self.calls.append("teardown")

    def _signal_stop(self):
        self.calls.append("signal_stop")
        super()._signal_stop()

    def _send_safety_action(self):
        self.calls.append("safety")


@pytest.fixture
def base_node(rclpy_ctx):
    n = _RecordingNode("base_node_under_test")
    yield n
    n.destroy_node()


def test_configure_failure_logs_and_recovers_via_on_error(rclpy_ctx):
    class BoomNode(_RecordingNode):
        def _setup(self):
            raise RuntimeError("setup boom")

    node = BoomNode("base_configure_boom")
    try:
        assert node.trigger_configure() != TransitionCallbackReturn.SUCCESS
        # ERROR routed through on_error: partial state was torn down and the
        # node is reconfigurable (unconfigured), not wedged or finalized.
        assert lifecycle_state_label(node) == "unconfigured"
        assert "teardown" in node.calls
    finally:
        node.destroy_node()


def test_cleanup_refused_while_work_in_progress(base_node):
    base_node._busy = True
    assert base_node.on_cleanup(None) == TransitionCallbackReturn.FAILURE
    assert "teardown" not in base_node.calls  # resources untouched under live work
    base_node._busy = False
    assert base_node.on_cleanup(None) == TransitionCallbackReturn.SUCCESS


def test_shutdown_from_active_stops_and_secures_before_teardown(base_node):
    assert base_node.trigger_configure() == TransitionCallbackReturn.SUCCESS
    assert base_node.trigger_activate() == TransitionCallbackReturn.SUCCESS
    assert base_node._try_claim_work() is None
    # Work ends when told to stop, as a well-behaved goal would.
    stop_event = base_node._stop_event
    base_node.calls.clear()

    def signal_and_release():
        base_node.calls.append("signal_stop")
        stop_event.set()
        base_node._busy = False

    base_node._signal_stop = signal_and_release
    assert base_node.trigger_shutdown() == TransitionCallbackReturn.SUCCESS
    assert base_node.calls == ["signal_stop", "safety", "teardown"]


def test_shutdown_teardown_runs_even_when_safety_send_raises(base_node):
    """Found live: a broken safety encoder skipped teardown on shutdown and
    orphaned the policy-server subprocess. The two steps must fail
    independently — the process is dying and teardown must still run."""

    def safety_boom():
        base_node.calls.append("safety")
        raise RuntimeError("encoder broke")

    base_node._send_safety_action = safety_boom
    assert base_node.on_shutdown(None) == TransitionCallbackReturn.ERROR  # failure surfaced
    assert "teardown" in base_node.calls  # ...but resources still released


def test_goal_work_releases_and_unbinds_on_exception(base_node):
    base_node._accepting_work = True
    assert base_node._try_claim_work() is None

    class Handle:
        goal_id = object()
        is_cancel_requested = False

    with pytest.raises(RuntimeError, match="boom"):
        with base_node._goal_work(Handle()):
            raise RuntimeError("boom")
    assert not base_node.busy  # a crashed execution can't brick the node
    assert base_node._active_goal is None


def test_claim_and_stop_serialize_on_the_work_gate(base_node):
    base_node._accepting_work = True
    assert base_node._try_claim_work() is None
    claimed_event = base_node._stop_event
    assert claimed_event is not None and not claimed_event.is_set()

    base_node._stop_and_secure(wait_timeout=0.1)  # work is wedged; bounded wait

    # The claim's event was set (the goal WILL see the stop), and acceptance
    # is closed: no new work can be claimed after deactivation began.
    assert claimed_event.is_set()
    assert base_node._try_claim_work() is not None
    base_node._busy = False
    assert base_node._try_claim_work() is not None  # still inactive


def test_concurrent_claims_admit_exactly_one_winner(base_node):
    # The goal-accept race the work gate exists to close: under a
    # MultiThreadedExecutor + ReentrantCallbackGroup, two goal requests could
    # both pass the busy check before either _execute ran, driving two loops
    # against one runner/bridge. _try_claim_work's check-and-set under the gate
    # must admit exactly one of a concurrent burst. (This pins the invariant
    # BusyGuard's dedicated unit test used to guard before it was folded in.)
    base_node._accepting_work = True
    start = threading.Barrier(32)
    wins: list[int] = []

    def worker():
        start.wait()
        if base_node._try_claim_work() is None:
            wins.append(1)

    threads = [threading.Thread(target=worker) for _ in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(wins) == 1
