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

"""Send and cancel goals on other action servers from inside a node's own work.

For apps that orchestrate action servers as children, for example a manager
that runs the episode recorder and a policy runner for each episode. Both
functions block the calling thread while polling futures, so call them from
work running under ``goal_work`` (an action execute callback or a
service-started thread) on a multi-threaded executor, never from a callback
the executor must return from to complete them.
"""

from __future__ import annotations

from rosetta.robots.ros2.node_utils import wait_until

__all__ = [
    "cancel_goal",
    "send_goal",
]

_POLL_SEC = 0.01


def send_goal(
    client,
    goal,
    *,
    name: str,
    logger,
    feedback_callback=None,
    server_timeout: float = 10.0,
    response_timeout: float = 10.0,
):
    """Send ``goal`` through ``client`` and return the accepted goal handle, or None.

    Returns None, after logging an error naming ``name``, when the server is
    not available, the response times out, or the goal is rejected.
    """
    if not client.wait_for_server(timeout_sec=server_timeout):
        logger.error(f"{name} action server not available")
        return None
    future = client.send_goal_async(goal, feedback_callback=feedback_callback)
    if not wait_until(future.done, response_timeout, poll=_POLL_SEC):
        logger.error(f"{name} goal send timed out")
        return None
    handle = future.result()
    if not handle.accepted:
        logger.error(f"{name} goal rejected")
        return None
    logger.info(f"{name} started")
    return handle


def cancel_goal(
    handle,
    *,
    name: str,
    logger,
    wait_result: bool = False,
    cancel_timeout: float = 5.0,
    result_timeout: float = 10.0,
):
    """Cancel a goal, best effort. ``handle`` may be None, which is a no-op.

    With ``wait_result``, also wait until the goal has finished and return its
    result message (None on timeout), so the server's work slot is free before
    the next goal. A cancel timeout only warns, since the goal may still end.
    Failures are logged, never raised.
    """
    if handle is None:
        return None
    try:
        if not wait_until(handle.cancel_goal_async().done, cancel_timeout, poll=_POLL_SEC):
            logger.warning(f"{name} cancel timed out")
        if not wait_result:
            logger.info(f"{name} cancelled")
            return None
        result_future = handle.get_result_async()
        if not wait_until(result_future.done, result_timeout, poll=_POLL_SEC):
            logger.warning(f"{name} result timed out after cancel")
            return None
        logger.info(f"{name} cancelled")
        return result_future.result().result
    except Exception as e:
        logger.warning(f"Failed to cancel {name}: {e}")
        return None
