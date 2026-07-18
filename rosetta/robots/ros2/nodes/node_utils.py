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

Node-side concerns: rcl-legal action termination (finish_goal), QoS
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
# Goal termination
# =============================================================================


def finish_goal(
    goal_handle,
    result,
    *,
    service_cancelled: bool = False,
    success_message: str | None = None,
) -> None:
    """Set an action goal's terminal state using only rcl-legal transitions.

    ``canceled()`` is legal only from CANCELING (``is_cancel_requested``
    True); a service-initiated cancel leaves the goal EXECUTING and must
    ``abort()`` instead. ``succeed()``/``abort()`` are legal from both
    EXECUTING and CANCELING, so a cancel racing in after these checks cannot
    produce an illegal transition. ``result`` must carry ``success`` and
    ``message`` fields (true for all rosetta action results).
    """
    if goal_handle.is_cancel_requested:
        result.success = False
        result.message = "Cancelled"
        goal_handle.canceled()
    elif service_cancelled:
        result.success = False
        result.message = "Cancelled via service"
        goal_handle.abort()
    elif result.success:
        if success_message is not None:
            result.message = success_message
        goal_handle.succeed()
    else:
        goal_handle.abort()


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
