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

"""The single lifecycle-node idiom for rosetta.

:class:`RosettaLifecycleNode` owns the lifecycle transitions. It gives every
rosetta node one discipline: fail-loud configure, one teardown path shared by
cleanup/shutdown/error, and a work gate that keeps goal acceptance coherent
with deactivation. Subclasses implement ``_setup()`` and ``_teardown()``, and
where relevant extend ``_signal_stop()`` and ``_send_safety_action()``. They
never override the lifecycle callbacks themselves.

:class:`BridgeLifecycleNode` adds :class:`TopicBridge` ownership for hosts
whose specs are known at construction time (the framework adapter packages).
Nodes that resolve their resources from ROS parameters at configure time
subclass :class:`RosettaLifecycleNode` directly.

``wait_until`` is imported from ``nodes.node_utils``. The nodes package
depends on this module's class, not the reverse, so the cross-directory
import is cycle-free.
"""

from __future__ import annotations

import threading
import traceback
from contextlib import contextmanager

from rclpy.action import CancelResponse, GoalResponse
from rclpy.lifecycle import Node, State, TransitionCallbackReturn

from rosetta.contract.specs import ActionStreamSpec, ObservationStreamSpec
from rosetta.robots.ros2.nodes.node_utils import wait_until
from rosetta.robots.ros2.rclpy_utils import LIFECYCLE_CONFIGURED_LABELS, lifecycle_state_label
from rosetta.robots.ros2.topic_bridge import TopicBridge


class RosettaLifecycleNode(Node):
    """Lifecycle node base with one transition discipline for every rosetta node.

    Configure and teardown failures are surfaced as exceptions. ``_setup()``
    raises, the base catches, logs the traceback, and returns ERROR so
    ``on_error`` tears down whatever was partially built. The base logs
    because rclpy swallows transition-callback exceptions silently, leaving no
    other place for the traceback to surface.

    Work (a goal, a recording, an episode) is claimed via
    :meth:`_try_claim_work`, usually through the default :meth:`_on_goal`, and
    released by the :meth:`_goal_work` context manager wrapping its execution.
    The claim and :meth:`_stop_and_secure` serialize on one lock, so every
    accepted piece of work has a stop event that deactivation will set, and no
    work is accepted after acceptance is closed. Action cancels are routed by
    goal id (:meth:`_on_cancel`), so a stale cancel for a finished goal can
    never stop the goal that replaced it.
    """

    #: Bounded wait for in-progress work to end during deactivate/shutdown.
    STOP_WORK_TIMEOUT_SEC: float = 5.0

    def __init__(self, node_name: str, **kwargs):
        super().__init__(node_name, enable_logger_service=True, **kwargs)
        # One lock guards all work-gate state below. threading.Lock is not
        # reentrant, so reads from outside a claim (the `busy` property,
        # _stop_and_secure's wait, on_cleanup) go through the property, while
        # code already holding the gate touches the fields directly.
        self._work_gate = threading.Lock()
        self._busy = False  # True while a claim is in progress
        self._accepting_work = False
        self._stop_event: threading.Event | None = None
        self._active_goal = None  # bound by _goal_work for cancel routing

    # -------------------- Subclass hooks --------------------

    def _setup(self) -> None:
        """Create all resources on configure.

        :raises Exception: on any failure. The base catches it and routes to
            ``on_error`` for teardown, so raising is the way to report a failed
            configure.
        """
        raise NotImplementedError

    def _teardown(self) -> None:
        """Destroy everything ``_setup()`` created.

        Must tolerate partial state, since ``on_error`` calls it after a
        configure that raised partway through, and cleanup/shutdown/error all
        share this one path.
        """
        raise NotImplementedError

    def _signal_stop(self) -> None:
        """Ask in-progress work to stop.

        Sets the current stop event if one is armed. Extend to add a further
        stop signal such as ``runner.request_stop()``.
        """
        ev = self._stop_event
        if ev is not None:
            ev.set()

    def _send_safety_action(self) -> None:
        """Secure the robot after work has stopped.

        No-op unless the node commands hardware. See
        :class:`BridgeLifecycleNode` for the hardware-commanding override.
        """

    # -------------------- Work gate --------------------

    def _try_claim_work(self) -> str | None:
        """Claim the single work slot under the gate, all-or-nothing.

        Checks acceptance, claims ``_busy``, and arms a fresh stop event as one
        atomic step under :attr:`_work_gate`.

        A fresh event is armed per claim and never cleared or nulled. That is
        what makes a cancel or deactivate landing in the accept->start window
        reliable. There is always an event for :meth:`_signal_stop` to set, and
        a stale set event from finished work can never leak into new work.

        :returns: None on success, otherwise the rejection reason. Handlers put
            the returned string in their response instead of re-deriving it
            from unsynchronized state.
        """
        with self._work_gate:
            if not self._accepting_work:
                return "node not active"
            if self._busy:
                return "already busy"
            self._busy = True
            self._stop_event = threading.Event()
            return None

    def _on_goal(self, _goal_request) -> GoalResponse:
        """Default action goal callback. Accept only if the work slot is claimable.

        :param _goal_request: the incoming goal, unused (acceptance depends on
            the work slot, not on goal contents).
        :returns: ACCEPT when :meth:`_try_claim_work` succeeds, else REJECT.
        """
        reason = self._try_claim_work()
        if reason is not None:
            self.get_logger().warning(f"Goal rejected: {reason}")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_cancel(self, goal_handle) -> CancelResponse:
        """Default action cancel callback. Stop work only if the cancel targets the bound goal.

        The cancel is routed by goal id. Under the reentrant callback group a
        cancel for goal A can run after A finished and goal B claimed the slot.
        An unconditional ``_signal_stop()`` would then kill B, which is the
        stale-cancel race. A cancel landing in the accept->bind window finds no
        bound goal and skips the signal here; :meth:`_goal_work` re-checks
        ``is_cancel_requested`` after binding, which closes that window.
        Service-initiated stop paths (recorder cancel service, HIL
        stop_episode) deliberately keep cancel-the-current semantics, since
        they carry no goal id to route by.

        :param goal_handle: the goal the cancel request targets.
        :returns: ACCEPT always. Acceptance acknowledges the request; whether
            work actually stops depends on the goal-id match.
        """
        with self._work_gate:
            active = self._active_goal
            if active is not None and goal_handle.goal_id == active.goal_id:
                self._signal_stop()
        return CancelResponse.ACCEPT

    @contextmanager
    def _goal_work(self, goal_handle):
        """Bind claimed work for cancel routing, yield its stop event, release on exit.

        Wraps the execution of work already claimed via
        :meth:`_try_claim_work`. Releasing ``_busy`` in the ``finally`` on every
        exit path, including exceptions, is what keeps a crashed execution from
        rejecting all subsequent work until restart.

        :param goal_handle: the action goal handle, or None for
            service/button-initiated work that has no cancel routing.
        :yields: the stop event armed by the claim, for the work loop to poll.
        """
        with self._work_gate:
            self._active_goal = goal_handle
            stop_event = self._stop_event
        # A cancel accepted in the accept->bind window found _active_goal
        # unbound and skipped the signal. Re-check and honor it now.
        if goal_handle is not None and goal_handle.is_cancel_requested:
            self._signal_stop()
        try:
            yield stop_event
        finally:
            with self._work_gate:
                self._active_goal = None
                self._busy = False

    def _stop_and_secure(self, wait_timeout: float | None = None) -> None:
        """Close acceptance, stop work, wait bounded, then secure, in that order.

        The safety action must be the LAST command on the wire. Sending it
        before the work loop stops would let one more command land over it and
        leave the robot unsecured. Callers run this before
        ``super().on_deactivate()`` so lifecycle publishers are still active for
        the safety send.

        :param wait_timeout: seconds to wait for work to stop before securing
            anyway. Defaults to :attr:`STOP_WORK_TIMEOUT_SEC` when None.
        """
        if wait_timeout is None:
            wait_timeout = self.STOP_WORK_TIMEOUT_SEC
        with self._work_gate:
            self._accepting_work = False
            self._signal_stop()
        if not wait_until(lambda: not self.busy, timeout=wait_timeout):
            self.get_logger().warning(f"Work did not stop within {wait_timeout:.1f}s; sending safety action anyway")
        self._send_safety_action()

    # -------------------- Lifecycle callbacks (final) --------------------

    def on_configure(self, _state: State) -> TransitionCallbackReturn:
        """Build resources via ``_setup()``. Route any failure to ``on_error``.

        :returns: SUCCESS if ``_setup()`` completed, else ERROR.
        """
        try:
            self._setup()
        except Exception:
            # rclpy swallows exceptions raised from transition callbacks and
            # just returns ERROR with no log, so this is the only place the
            # traceback can surface. ERROR routes through on_error, which tears
            # down whatever _setup() partially built.
            self.get_logger().error(f"Configure failed:\n{traceback.format_exc()}")
            return TransitionCallbackReturn.ERROR
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """Activate publishers, then open work acceptance.

        Acceptance opens only after ``super().on_activate()`` activates the
        lifecycle publishers, so no accepted goal can run before its output
        publishers are live.

        :returns: the base transition result, unchanged.
        """
        result = super().on_activate(state)
        if result == TransitionCallbackReturn.SUCCESS:
            with self._work_gate:
                self._accepting_work = True
        return result

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """Stop and secure work, then deactivate publishers.

        :meth:`_stop_and_secure` runs first so the safety action goes out while
        the lifecycle publishers are still active.

        :returns: the base transition result.
        """
        self._stop_and_secure()
        return super().on_deactivate(state)

    def on_cleanup(self, _state: State) -> TransitionCallbackReturn:
        """Tear down resources once idle. Refuse while work is in progress.

        :returns: FAILURE if work is still in progress (tearing down under live
            work is a bug to surface, not to work around), ERROR if
            ``_teardown()`` raised, else SUCCESS.
        """
        if self.busy:
            # Never yank resources out from under live work. A goal that
            # outlived deactivate's bounded wait is a real bug, so surface it
            # instead of tearing down around it. Shutdown still proceeds.
            self.get_logger().error("Cleanup refused: work still in progress")
            return TransitionCallbackReturn.FAILURE
        try:
            self._teardown()
        except Exception:
            self.get_logger().error(f"Cleanup failed:\n{traceback.format_exc()}")
            return TransitionCallbackReturn.ERROR
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, _state: State) -> TransitionCallbackReturn:
        """Secure then tear down, tolerating a shutdown straight from active.

        The two steps fail independently on purpose. The process is dying, and
        ``_teardown()`` must still release external resources (for example a
        policy-server subprocess) even when the safety send itself is what
        broke.

        :returns: SUCCESS if both steps succeeded, ERROR if either raised.
        """
        # Stop/wait/secure before teardown so the robot's last input is the
        # safety action, not a stale command. Cheap when already idle.
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
        """Recover from a failed transition by securing and tearing down.

        The return value decides the node's fate under the lifecycle contract.
        SUCCESS sends it to unconfigured, where it can be reconfigured. FAILURE
        finalizes it. Teardown success is the deciding condition: a node whose
        entities may have leaked is not safely reconfigurable, so a failed
        teardown returns FAILURE rather than pretending recovery succeeded.

        :returns: SUCCESS when ``_teardown()`` succeeded (node recoverable to
            unconfigured), FAILURE when it raised (node finalized).
        """
        self.get_logger().error(f"Error in state '{state.label}', tearing down")
        try:
            self._stop_and_secure()
        except Exception:
            # Logged and carried on. A failed stop/secure must not block the
            # teardown that makes the node recoverable.
            self.get_logger().error(f"Stop/secure failed during error handling:\n{traceback.format_exc()}")
        try:
            self._teardown()
        except Exception:
            self.get_logger().error(f"Teardown failed during error handling:\n{traceback.format_exc()}")
            return TransitionCallbackReturn.FAILURE
        return TransitionCallbackReturn.SUCCESS

    # -------------------- Lifecycle state --------------------

    @property
    def busy(self) -> bool:
        """True while a claimed piece of work is in progress.

        Reads ``_busy`` under the work gate, so it never races the claim or
        release in :meth:`_try_claim_work` and :meth:`_goal_work`. Callers must
        not already hold the gate, since the lock is not reentrant. The internal
        claim/release paths touch ``self._busy`` directly instead.
        """
        with self._work_gate:
            return self._busy

    @property
    def is_active(self) -> bool:
        return lifecycle_state_label(self) == "active"

    @property
    def is_configured(self) -> bool:
        return lifecycle_state_label(self) in LIFECYCLE_CONFIGURED_LABELS


class BridgeLifecycleNode(RosettaLifecycleNode):
    """Lifecycle node owning a :class:`TopicBridge` (``self.bridge``).

    For hosts whose specs are known at construction time, meaning the framework
    adapter packages (for example ``lerobot_robot_rosetta.Rosetta``). Adapters
    that host extra ROS entities subclass this and extend the ``_setup()`` and
    ``_teardown()`` hooks via ``super()`` (see the teleoperator adapter).
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
        """Create the bridge's ROS entities on configure. Subclasses extend via super()."""
        self.bridge.setup(self)

    def _teardown(self) -> None:
        """Destroy the bridge's ROS entities. Subclasses extend via super()."""
        self.bridge.teardown()

    def _send_safety_action(self) -> None:
        """Publish the bridge's safety action, which also disarms its watchdog."""
        self.bridge.send_safety_action()
