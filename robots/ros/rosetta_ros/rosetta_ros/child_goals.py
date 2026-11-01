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
that runs the episode recorder and a policy runner for each episode. Waiting
polls futures that only the node's executor completes, so call these from a
work thread (a goal run by ``create_work_action_server``, or a
service-started thread), never from a callback, and don't wait while a
lifecycle transition holds the executor.
"""

from __future__ import annotations

from collections.abc import Callable

from rosetta_ros.node_utils import wait_until

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
    result_timeout: float = 10.0,
    abandon: Callable[[], bool] | None = None,
):
    """Cancel a goal, best effort. ``handle`` may be None, which is a no-op.

    The cancel request is sent at once, and a server that refuses it is
    reported (with a warning) when its response arrives. Without
    ``wait_result`` that is all. With it, wait until the goal has finished and
    return its result message (None on timeout), so the server's work slot is
    free before the next goal. ``abandon`` ends that wait early, returning
    None, as soon as it returns true: pass one that turns true when a
    lifecycle transition starts, since the transition holds the executor that
    completes the wait. Failures are logged, never raised.
    """
    if handle is None:
        return None
    try:
        handle.cancel_goal_async().add_done_callback(lambda future: _report_refusal(future, name, logger))
        if not wait_result:
            logger.info(f"{name} cancel requested")
            return None
        result_future = handle.get_result_async()
        stop = result_future.done if abandon is None else (lambda: result_future.done() or abandon())
        wait_until(stop, result_timeout, poll=_POLL_SEC)
        if not result_future.done():
            if abandon is not None and abandon():
                logger.warning(f"{name} result not awaited: the node is deactivating")
            else:
                logger.warning(f"{name} result timed out after cancel")
            return None
        logger.info(f"{name} cancelled")
        return result_future.result().result
    except Exception as e:
        logger.warning(f"Failed to cancel {name}: {e}")
        return None


def _report_refusal(future, name: str, logger) -> None:
    """Warn when a child's server answered a cancel request without cancelling anything."""
    try:
        response = future.result()
    except Exception as e:
        logger.warning(f"{name} cancel failed: {e}")
        return
    if response is not None and not response.goals_canceling:
        logger.warning(f"{name} refused the cancel (return code {response.return_code}); it may still be running")
