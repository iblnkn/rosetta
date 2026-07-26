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

"""Helpers used only by the lifecycle nodes in rosetta.robots.ros2.nodes.

Node-side concerns: the termination vocabulary and rcl-legal action termination
(finish_goal), cancelling a node's own action goals (request_cancel_all), QoS
introspection/conversion for bag topic metadata, polling waits (wait_until),
and the shared node entry point (spin_lifecycle_node). QoS-dict parsing and
lifecycle-state helpers shared with the wider ros2 layer stay in
rosetta.robots.ros2.rclpy_utils. Work-slot mutual exclusion lives on
RosettaLifecycleNode itself (the ``busy`` property + work gate).
"""

from __future__ import annotations

import time
from typing import Callable

import rclpy
from action_msgs.srv import CancelGoal
from rcl_interfaces.msg import FloatingPointRange, ParameterDescriptor
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode
from rclpy.qos import QoSProfile
from rclpy.signals import SignalHandlerOptions
from rosbag2_py._storage import (
    Duration as Rosbag2Duration,
)
from rosbag2_py._storage import (
    QoS as Rosbag2QoS,
)
from rosbag2_py._storage import (
    rmw_qos_durability_policy_t,
    rmw_qos_history_policy_t,
    rmw_qos_liveliness_policy_t,
    rmw_qos_reliability_policy_t,
)

from rosetta_interfaces.action import ManageEpisode, RunPolicy

# =============================================================================
# Parameter declaration
# =============================================================================


def positive_rate_descriptor(description: str) -> ParameterDescriptor:
    """Descriptor for a Hz-rate parameter: rclpy rejects values outside [0.1, 1000].

    The nodes divide by these rates (``1.0 / rate_hz``); a declarative range
    makes 0 (ZeroDivisionError) and negatives (busy-spin ``Event.wait``)
    unrepresentable at declare/set time instead of failing inside a loop.
    """
    return ParameterDescriptor(
        description=description,
        floating_point_range=[FloatingPointRange(from_value=0.1, to_value=1000.0, step=0.0)],
    )


# =============================================================================
# Termination vocabulary
# =============================================================================

# The .action files are the single source of these values: they are what
# non-Python clients read, so defining them again here in Python would be two
# places to keep in sync. Re-exported under short names because every node uses
# them and ``ManageEpisode.Result.TERMINATION_NODE_DEACTIVATED`` does not read
# well inline.
#
# ManageEpisode declares the widest reason set, so it is the natural home for
# all but ``completed``, which only a policy run can produce.
# test_termination_constants.py checks that each action declares every reason
# its own server can produce.
#
# The goal's terminal GoalStatus is the coarse view -- SUCCEEDED when the work
# reached a defined end, CANCELED when a client took it away, ABORTED when the
# server stopped it. These values say which of those, exactly.
TERMINATION_STOPPED = ManageEpisode.Result.TERMINATION_STOPPED
TERMINATION_TIMEOUT = ManageEpisode.Result.TERMINATION_TIMEOUT
TERMINATION_REWARD_THRESHOLD = ManageEpisode.Result.TERMINATION_REWARD_THRESHOLD
TERMINATION_COMPLETED = RunPolicy.Result.TERMINATION_COMPLETED
TERMINATION_CANCELLED = ManageEpisode.Result.TERMINATION_CANCELLED
TERMINATION_NODE_DEACTIVATED = ManageEpisode.Result.TERMINATION_NODE_DEACTIVATED
TERMINATION_ERROR = ManageEpisode.Result.TERMINATION_ERROR

# The ``outcome`` value set (ManageEpisode only): whether the robot did the
# task. Independent of how the episode ended.
OUTCOME_SUCCESS = ManageEpisode.Result.OUTCOME_SUCCESS
OUTCOME_FAILURE = ManageEpisode.Result.OUTCOME_FAILURE
OUTCOME_UNLABELED = ManageEpisode.Result.OUTCOME_UNLABELED

#: Reasons that must not report the work as having reached a defined end.
#: Everything else succeeds.
#:
#: ``node_deactivated`` is here because a lifecycle deactivate IS the server
#: choosing to stop a goal, which the ROS 2 action docs define as an abort.
#:
#: ``cancelled`` is here only as a fallback. The normal cancel path never
#: reaches it -- ``is_cancel_requested`` is checked first and wins -- so this
#: catches the anomaly where the reason was latched but the goal never entered
#: CANCELING. Succeeding there would report a clean finish for work somebody
#: asked to cancel.
_ABORT_REASONS = frozenset(
    {
        TERMINATION_ERROR,
        TERMINATION_NODE_DEACTIVATED,
        TERMINATION_CANCELLED,
    }
)


# =============================================================================
# Goal termination
# =============================================================================


def finish_goal(goal_handle, result) -> None:
    """Set an action goal's terminal state from ``result.termination_reason``.

    Legality first, then meaning. ``canceled()`` is legal only from CANCELING
    (``is_cancel_requested`` True), so that branch decides on its own -- and it
    is reachable from every cancel path, because the cancel services forward to
    the action server's own ``_action/cancel_goal`` rather than faking a stop.
    ``succeed()``/``abort()`` are legal from both EXECUTING and CANCELING, so a
    cancel racing in after the check cannot produce an illegal transition.

    ``result`` must carry a ``termination_reason`` field, already set by the
    work loop (true for all three rosetta action results). This function never
    writes it: the loop is its single writer, so the terminal state and the
    reported reason cannot disagree.
    """
    if goal_handle.is_cancel_requested:
        goal_handle.canceled()
    elif result.termination_reason in _ABORT_REASONS:
        goal_handle.abort()
    else:
        goal_handle.succeed()


def request_cancel_all(cancel_client, *, warn: Callable[[str], None] | None = None) -> bool:
    """Ask an action server to cancel every goal it is currently running.

    ``CancelGoal.Request()`` is the action spec's cancel-all wildcard: an
    all-zero goal id with a zero stamp means "cancel all goals"
    (``rcl_action_process_cancel_request``). Nodes use this to give clients that
    can call services but not actions -- Foxglove, most of all -- a real cancel
    rather than a lookalike. It is the same request an action client sends, so
    the goal genuinely enters CANCELING and ends CANCELED.

    Fire-and-forget on purpose. The response is never awaited: doing so from
    inside a service callback would need the executor to service three more
    wakeups on the same node before the handler could return, which starves a
    thread-limited executor and hard-deadlocks a single-threaded one. The
    response carries nothing the caller needs anyway -- whether work was running
    is already known from the node's own ``busy`` flag.

    Args:
        cancel_client: Client for ``<action_name>/_action/cancel_goal``.
        warn: Optional callable invoked with a message when the server's cancel
            service is not reachable.

    Returns:
        True when the request was dispatched, False when the cancel service was
        not available.

    """
    if not cancel_client.service_is_ready():
        if warn is not None:
            warn("cancel service not available; nothing was cancelled")
        return False
    cancel_client.call_async(CancelGoal.Request())
    return True


# =============================================================================
# QoS introspection / conversion (bag topic metadata)
# =============================================================================


def extract_qos_numeric_values(q: QoSProfile) -> dict[str, int]:
    """Flatten a QoSProfile's policy enums to their raw RMW integer codes.

    rclpy exposes each policy as an enum whose ``.value`` is the underlying
    RMW numeric constant. rosbag2's QoS setters want those ints, so read
    them out here once.

    Args:
        q: Source rclpy QoS profile.

    Returns:
        Mapping with keys ``depth``, ``history``, ``reliability``,
        ``durability``, ``liveliness``. Every value is an int RMW policy code.

    """
    return {
        "depth": q.depth,
        "history": q.history.value,
        "reliability": q.reliability.value,
        "durability": q.durability.value,
        "liveliness": q.liveliness.value,
    }


# rosbag2_py._storage.Duration takes int32 seconds / uint32 nanoseconds.
# "Infinite"/unset durations discovered from real publishers come back as
# RMW_DURATION_INFINITE (int64 nanoseconds max), which overflows those
# bounds, so clamp to the max representable sentinel.
_MAX_DURATION_SEC = 2147483647
_MAX_DURATION_NSEC = 4294967295


def _rosbag2_duration(rclpy_duration) -> Rosbag2Duration:
    ns = int(getattr(rclpy_duration, "nanoseconds", 0) or 0)
    return Rosbag2Duration(
        min(ns // 1_000_000_000, _MAX_DURATION_SEC),
        min(ns % 1_000_000_000, _MAX_DURATION_NSEC),
    )


def qos_to_rosbag2(q: QoSProfile) -> Rosbag2QoS:
    """Convert an rclpy QoSProfile to a rosbag2_py QoS for TopicMetadata.

    The QoS setter methods require rmw_qos_*_policy_t enums, not raw ints.
    """
    vals = extract_qos_numeric_values(q)
    bag_qos = Rosbag2QoS(vals["depth"])
    bag_qos = bag_qos.history(rmw_qos_history_policy_t(vals["history"]))
    bag_qos = bag_qos.reliability(rmw_qos_reliability_policy_t(vals["reliability"]))
    bag_qos = bag_qos.durability(rmw_qos_durability_policy_t(vals["durability"]))
    bag_qos = bag_qos.liveliness(rmw_qos_liveliness_policy_t(vals["liveliness"]))
    bag_qos = bag_qos.deadline(_rosbag2_duration(q.deadline))
    bag_qos = bag_qos.lifespan(_rosbag2_duration(q.lifespan))
    return bag_qos.liveliness_lease_duration(_rosbag2_duration(q.liveliness_lease_duration))


# =============================================================================
# Timing Utilities
# =============================================================================


def wait_until(predicate: Callable[[], bool], timeout: float, poll: float = 0.1) -> bool:
    """Block until ``predicate()`` returns true or ``timeout`` seconds elapse.

    Used by lifecycle ``on_deactivate`` callbacks to give in-progress work a
    bounded window to wind down before the transition proceeds.

    Args:
        predicate: Condition polled once per interval.
        timeout: Maximum seconds to wait.
        poll: Seconds slept between checks.

    Returns:
        The final ``predicate()`` value. True if it became true in time,
        False on timeout.

    """
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(poll)
    return predicate()


# =============================================================================
# Node entry point
# =============================================================================


def spin_lifecycle_node(node_factory: Callable[[], LifecycleNode], *, args=None) -> int:
    """Run a lifecycle node until interrupted, driving the shutdown transition on exit.

    ``destroy_node()`` never runs ``on_shutdown``. Only the lifecycle state
    machine does. So ``trigger_shutdown()`` is called explicitly to let the
    node finalize resources (e.g. close an open bag writer, send a safety
    action) on Ctrl+C. rclpy's default SIGINT handler would shut the context
    down before user code can react, making that transition impossible exactly
    when it matters. Signal handling is disabled and KeyboardInterrupt is the
    shutdown trigger instead.
    """
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = node_factory()
    # 4 threads: enough for a blocking execute callback + feedback/service
    # callbacks alongside it. No caller has needed to override it.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        # spin_once loop, not spin(): without rclpy's signal handlers a
        # blocking rcl_wait never wakes on SIGINT, so return to Python
        # regularly to let KeyboardInterrupt be delivered.
        while True:
            executor.spin_once(timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.trigger_shutdown()
        except Exception as e:
            node.get_logger().warning(f"Shutdown transition not driven: {e}")
        node.destroy_node()
        rclpy.try_shutdown()
    return 0
