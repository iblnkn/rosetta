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

"""
The single lifecycle-node idiom for rosetta.

:class:`RosettaLifecycleNode` owns the six-callback transition discipline:
fail-loud configure, one teardown path shared by cleanup/shutdown/error, and
a work gate that keeps goal acceptance coherent with deactivation. Subclasses
implement ``_setup()``/``_teardown()`` (and, where relevant, extend
``_signal_stop()``/``_send_safety_action()``) and never override the
lifecycle callbacks themselves.

:class:`BridgeLifecycleNode` adds :class:`TopicBridge` ownership for hosts
whose specs are known at construction time (the framework adapter packages).
Nodes that resolve their resources from ROS parameters at configure time
subclass :class:`RosettaLifecycleNode` directly.

Imports ``BusyGuard``/``wait_until`` from ``nodes.node_utils`` — the nodes
package depends on this module's class, not the other way around, so the
cross-directory import is cycle-free.
"""

from __future__ import annotations

import threading
import traceback
from contextlib import contextmanager

from rclpy.action import CancelResponse, GoalResponse
from rclpy.lifecycle import Node, State, TransitionCallbackReturn

from rosetta.contract.specs import ActionStreamSpec, ObservationStreamSpec
from rosetta.robots.ros2.nodes.node_utils import BusyGuard, wait_until
from rosetta.robots.ros2.ros2_utils import LIFECYCLE_CONFIGURED_LABELS, lifecycle_state_label
from rosetta.robots.ros2.topic_bridge import TopicBridge


class RosettaLifecycleNode(Node):
    """Lifecycle node base: one transition discipline for every rosetta node.

    All configure/teardown failures are exceptions: ``_setup()`` raises, the
    base catches, logs the traceback (rclpy swallows transition-callback
    exceptions silently), and returns ERROR so ``on_error`` tears down
    whatever was partially built.

    Work (a goal, a recording, an episode) is claimed via
    :meth:`_try_claim_work` (usually through the default :meth:`_on_goal`)
    and released by the :meth:`_goal_work` context manager wrapping its
    execution. The claim and :meth:`_stop_and_secure` serialize on one lock,
    so every accepted piece of work has a stop event that deactivation will
    set, and no work is accepted after acceptance is closed. Action cancels
    are routed by goal id (:meth:`_on_cancel`), so a stale cancel for a
    finished goal can never stop the goal that replaced it.
    """

    #: Bounded wait for in-progress work to end during deactivate/shutdown.
    STOP_WORK_TIMEOUT_SEC: float = 5.0

    def __init__(self, node_name: str, **kwargs):
        super().__init__(node_name, enable_logger_service=True, **kwargs)
        self._busy = BusyGuard()
        self._accepting_work = False
        self._stop_event: threading.Event | None = None
        self._active_goal = None  # bound by _goal_work for cancel routing
        # Serializes work claims against _stop_and_secure: a claim always
        # leaves behind an event the stop path can set, and a stop always
        # closes acceptance before any later claim can be checked.
        self._work_gate = threading.Lock()

    # -------------------- Subclass hooks --------------------

    def _setup(self) -> None:
        """Create all resources on configure. Raise on any failure."""
        raise NotImplementedError

    def _teardown(self) -> None:
        """Destroy everything ``_setup()`` created. Must tolerate partial state."""
        raise NotImplementedError

    def _signal_stop(self) -> None:
        """Ask in-progress work to stop. Extend to add e.g. runner.request_stop()."""
        ev = self._stop_event
        if ev is not None:
            ev.set()

    def _send_safety_action(self) -> None:
        """Secure the robot after work stops. No-op unless the node commands hardware."""

    # -------------------- Work gate --------------------

    def _try_claim_work(self) -> str | None:
        """Atomically: check acceptance, claim ``_busy``, arm a fresh stop event.

        Returns None on success, else the rejection reason — handlers put it
        in their response instead of re-deriving it from unsynchronized state.

        The fresh event per claim (never cleared, never nulled) is what makes
        a cancel or deactivate landing in the accept->start window reliable:
        there is always an event for :meth:`_signal_stop` to set, and a stale
        set event from finished work can never leak into new work.
        """
        with self._work_gate:
            if not self._accepting_work:
                return "node not active"
            if not self._busy.try_acquire():
                return "already busy"
            self._stop_event = threading.Event()
            return None

    def _on_goal(self, _goal_request) -> GoalResponse:
        """Default action goal callback: accept iff the work slot is claimable."""
        reason = self._try_claim_work()
        if reason is not None:
            self.get_logger().warning(f"Goal rejected: {reason}")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_cancel(self, goal_handle) -> CancelResponse:
        """Default action cancel callback: stop work only if the cancel targets the bound goal.

        Routed by goal id: under the reentrant callback group, a cancel for
        goal A can run after A finished and goal B claimed the slot — an
        unconditional ``_signal_stop()`` would kill B (the stale-cancel
        race). A cancel landing in the accept->bind window skips the signal
        here; :meth:`_goal_work` re-checks ``is_cancel_requested`` after
        binding, closing that window. Service-initiated stop paths (recorder
        cancel service, HIL stop_episode) deliberately keep
        cancel-the-current semantics — they carry no goal id to route by.
        """
        with self._work_gate:
            active = self._active_goal
            if active is not None and goal_handle.goal_id == active.goal_id:
                self._signal_stop()
        return CancelResponse.ACCEPT

    @contextmanager
    def _goal_work(self, goal_handle):
        """Bind claimed work for cancel routing; yield its stop event; release on exit.

        Wraps the execution of work claimed via :meth:`_try_claim_work`.
        ``goal_handle=None`` marks service/button-initiated work (no cancel
        routing). Releasing ``_busy`` on every exit path is what keeps a
        crashed execution from rejecting all subsequent work until restart.
        """
        with self._work_gate:
            self._active_goal = goal_handle
            stop_event = self._stop_event
        # A cancel accepted between claim and bind found _active_goal unbound
        # and didn't signal — honor it now.
        if goal_handle is not None and goal_handle.is_cancel_requested:
            self._signal_stop()
        try:
            yield stop_event
        finally:
            with self._work_gate:
                self._active_goal = None
            self._busy.release()

    def _stop_and_secure(self, wait_timeout: float | None = None) -> None:
        """Close acceptance, stop work, wait bounded, then secure — in that order.

        The safety action must be the LAST command on the wire: sending it
        before the work loop stops lets one more command land over it. Runs
        before ``super().on_deactivate()`` so lifecycle publishers are still
        active for the safety send.
        """
        if wait_timeout is None:
            wait_timeout = self.STOP_WORK_TIMEOUT_SEC
        with self._work_gate:
            self._accepting_work = False
            self._signal_stop()
        if not wait_until(lambda: not self._busy.busy, timeout=wait_timeout):
            self.get_logger().warning(f"Work did not stop within {wait_timeout:.1f}s; sending safety action anyway")
        self._send_safety_action()

    # -------------------- Lifecycle callbacks (final) --------------------

    def on_configure(self, _state: State) -> TransitionCallbackReturn:
        try:
            self._setup()
        except Exception:
            # rclpy swallows exceptions from transition callbacks (returns
            # ERROR with no log), so this is the only place the traceback can
            # surface. ERROR routes through on_error, which tears down
            # whatever _setup() partially built.
            self.get_logger().error(f"Configure failed:\n{traceback.format_exc()}")
            return TransitionCallbackReturn.ERROR
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        result = super().on_activate(state)  # publishers first, then accept work
        if result == TransitionCallbackReturn.SUCCESS:
            with self._work_gate:
                self._accepting_work = True
        return result

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self._stop_and_secure()
        return super().on_deactivate(state)

    def on_cleanup(self, _state: State) -> TransitionCallbackReturn:
        if self._busy.busy:
            # Never yank resources out from under live work. A goal that
            # outlived deactivate's bounded wait is a real bug; surface it
            # instead of tearing down around it (shutdown still proceeds).
            self.get_logger().error("Cleanup refused: work still in progress")
            return TransitionCallbackReturn.FAILURE
        try:
            self._teardown()
        except Exception:
            self.get_logger().error(f"Cleanup failed:\n{traceback.format_exc()}")
            return TransitionCallbackReturn.ERROR
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, _state: State) -> TransitionCallbackReturn:
        # Shutdown is legal straight from active: stop/wait/secure before
        # teardown so the robot's last input is the safety action, not a
        # stale command. Cheap when idle. The two steps fail independently:
        # the process is dying, and _teardown() must still release external
        # resources (e.g. a policy-server subprocess) even when the safety
        # send itself is what broke.
        result = TransitionCallbackReturn.SUCCESS
        try:
            self._stop_and_secure()
        except Exception:
            self.get_logger().error(f"Stop/secure failed during shutdown:\n{traceback.format_exc()}")
            result = TransitionCallbackReturn.ERROR
        try:
            self._teardown()
        except Exception:
            self.get_logger().error(f"Shutdown teardown failed:\n{traceback.format_exc()}")
            result = TransitionCallbackReturn.ERROR
        return result

    def on_error(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().error(f"Error in state '{state.label}', tearing down")
        try:
            self._stop_and_secure()
        except Exception:
            # Logged and carried on: a failed stop/secure must not block the
            # teardown that makes the node recoverable.
            self.get_logger().error(f"Stop/secure failed during error handling:\n{traceback.format_exc()}")
        try:
            self._teardown()
        except Exception:
            # A node whose entities may have leaked is not safely
            # reconfigurable; FAILURE finalizes it instead of pretending
            # the recovery succeeded.
            self.get_logger().error(f"Teardown failed during error handling:\n{traceback.format_exc()}")
            return TransitionCallbackReturn.FAILURE
        return TransitionCallbackReturn.SUCCESS

    # -------------------- Lifecycle state --------------------
    # Answered from this node's own state machine (like the teleop node), so
    # both are safe to query before configure -- adapters call is_configured
    # precisely to decide whether to trigger_configure().

    @property
    def is_active(self) -> bool:
        return lifecycle_state_label(self) == "active"

    @property
    def is_configured(self) -> bool:
        return lifecycle_state_label(self) in LIFECYCLE_CONFIGURED_LABELS


class BridgeLifecycleNode(RosettaLifecycleNode):
    """Lifecycle node owning a :class:`TopicBridge` (``self.bridge``).

    For hosts whose specs are known at construction time — the framework
    adapter packages (e.g. ``lerobot_robot_rosetta.Rosetta``). Adapters that
    host extra ROS entities subclass it and extend the ``_setup()`` /
    ``_teardown()`` hooks (see the teleoperator adapter).
    """

    def __init__(
        self,
        node_name: str,
        observation_specs: list[ObservationStreamSpec],
        action_specs: list[ActionStreamSpec],
        fps: int,
        **kwargs,
    ):
        super().__init__(node_name, **kwargs)
        self.bridge = TopicBridge(observation_specs, action_specs, fps)

    def _setup(self) -> None:
        """Create ROS entities on configure. Subclasses extend via super()."""
        self.bridge.setup(self)

    def _teardown(self) -> None:
        """Destroy everything _setup() created. Subclasses extend via super()."""
        self.bridge.teardown()

    def _send_safety_action(self) -> None:
        # Also disarms the bridge's watchdog.
        self.bridge.send_safety_action()
