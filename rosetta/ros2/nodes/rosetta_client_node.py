#!/usr/bin/env python
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
RosettaClientNode: backend-agnostic ROS2 action server for policy inference.

The node owns the contract, the :class:`TopicBridge` (observation/action ROS2
plumbing), and the ``run_policy`` action lifecycle. The actual policy execution
is delegated to a :class:`~rosetta.backends.protocols.PolicyRunner` resolved by
name from the ``backend`` parameter (e.g. ``lerobot``, ``vla_foundry``). The node
imports no specific policy framework.

Usage:
    ros2 launch rosetta rosetta_client_launch.py

    ros2 action send_goal /rosetta_client/run_policy \
        rosetta_interfaces/action/RunPolicy "{prompt: 'pick up cube'}" --feedback
"""

from __future__ import annotations

import sys
import threading

from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rosetta_interfaces.action import RunPolicy

from rosetta.backends.protocols import load_policy_runner, PolicyRunner, RunnerResult
from rosetta.core.contract import load_contract
from rosetta.core.contract_utils import (
    iter_action_specs,
    iter_observation_specs,
    iter_reward_as_action_specs,
)
from rosetta.ros2.ros2_utils import is_jazzy_or_newer, wait_until
from rosetta.ros2.topic_bridge import TopicBridge


class RosettaClientNode(LifecycleNode):
    """Backend-agnostic ROS2 Lifecycle Action Server for policy inference."""

    def __init__(self):
        """Initialize the rosetta client node and declare shared parameters."""
        if is_jazzy_or_newer():
            super().__init__('rosetta_client', enable_logger_service=True)
        else:
            super().__init__('rosetta_client')

        # --- Shared (backend-agnostic) parameters ---
        self.declare_parameter(
            'contract_path',
            '',
            ParameterDescriptor(description='Path to contract YAML file', read_only=True),
        )
        self.declare_parameter(
            'backend',
            'lerobot',
            ParameterDescriptor(
                description='Policy backend to use (lerobot, vla_foundry, ...)',
                read_only=True,
            ),
        )
        self.declare_parameter(
            'is_classifier',
            False,
            ParameterDescriptor(
                description='Publish the reward section as the action output',
                read_only=True,
            ),
        )
        self.declare_parameter(
            'feedback_rate_hz',
            2.0,
            ParameterDescriptor(description='Rate for publishing action feedback'),
        )

        # State (resources created in lifecycle callbacks)
        self._contract = None
        self._bridge: TopicBridge | None = None
        self._runner: PolicyRunner | None = None
        self._action_server: ActionServer | None = None
        self._accepting_goals = False
        self._active_goal = None
        self._stop_event: threading.Event | None = None

        self.get_logger().info('Node created (unconfigured)')

    # -------------------- Lifecycle callbacks --------------------

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Load contract, build the bridge, resolve the backend runner."""
        contract_path = self.get_parameter('contract_path').value
        if not contract_path:
            self.get_logger().error('contract_path parameter required')
            return TransitionCallbackReturn.FAILURE

        try:
            self._contract = load_contract(contract_path)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'Failed to load contract: {e}')
            return TransitionCallbackReturn.FAILURE

        is_classifier = self.get_parameter('is_classifier').value
        obs_specs = list(iter_observation_specs(self._contract))
        if is_classifier:
            act_specs = list(iter_reward_as_action_specs(self._contract))
        else:
            act_specs = list(iter_action_specs(self._contract))

        # Build the backend-neutral topic bridge on this node.
        self._bridge = TopicBridge(obs_specs, act_specs, self._contract.fps)
        self._bridge.setup(self)

        # Resolve and prepare the backend policy runner.
        backend = self.get_parameter('backend').value
        try:
            self._runner = load_policy_runner(backend)
            self._runner.setup(self, self._contract)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"Failed to load backend '{backend}': {e}")
            return TransitionCallbackReturn.FAILURE

        self._action_server = ActionServer(
            self,
            RunPolicy,
            'run_policy',
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
            callback_group=ReentrantCallbackGroup(),
        )

        self.get_logger().info(f"Configured: contract={contract_path}, backend={backend}")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Activate lifecycle publishers and accept goals."""
        self._accepting_goals = True
        self.get_logger().info('Activated and ready for policy execution')
        return super().on_activate(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Stop accepting goals, cancel in-progress execution, send safety action."""
        self._accepting_goals = False

        if self._bridge is not None:
            self._bridge.send_safety_action()

        # Cancel any in-progress goal
        if self._stop_event is not None:
            self._stop_event.set()
        if self._runner is not None:
            self._runner.request_stop()

        if not wait_until(lambda: self._active_goal is None, timeout=5.0):
            self.get_logger().warning('Goal did not complete within timeout')

        self.get_logger().info('Deactivated')
        return super().on_deactivate(state)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Release resources and destroy the action server."""
        self._teardown_runner()

        if self._bridge is not None:
            self._bridge.teardown()
            self._bridge = None
        self._contract = None

        if self._action_server is not None:
            self.destroy_action_server(self._action_server)
            self._action_server = None

        self.get_logger().info('Cleaned up')
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Clean up resources before destruction."""
        self._accepting_goals = False
        self._teardown_runner()

        if self._bridge is not None:
            self._bridge.teardown()
            self._bridge = None

        if self._action_server is not None:
            self.destroy_action_server(self._action_server)
            self._action_server = None

        self.get_logger().info('Shutdown complete')
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Handle errors by cleaning up resources."""
        self.get_logger().error(f'Error occurred in state: {state.label}')
        try:
            self._accepting_goals = False
            if self._stop_event is not None:
                self._stop_event.set()
            if self._runner is not None:
                self._runner.request_stop()
            self._teardown_runner()
            if self._bridge is not None:
                self._bridge.teardown()
                self._bridge = None
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'Error during error handling: {e}')
        return TransitionCallbackReturn.SUCCESS

    def _teardown_runner(self) -> None:
        if self._runner is not None:
            try:
                self._runner.teardown()
            except Exception as e:  # noqa: BLE001
                self.get_logger().warning(f'Runner teardown error: {e}')

    # -------------------- Action callbacks --------------------

    def _on_goal(self, _goal_request) -> GoalResponse:
        """Accept or reject a client request to begin an action."""
        self.get_logger().info('Received goal request')
        if not self._accepting_goals:
            self.get_logger().warning('Rejected: node not active')
            return GoalResponse.REJECT
        if self._active_goal is not None:
            self.get_logger().warning('Rejected: already running')
            return GoalResponse.REJECT
        self.get_logger().info('Goal accepted')
        return GoalResponse.ACCEPT

    def _on_cancel(self, _goal_handle) -> CancelResponse:
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info('Received cancel request')
        if self._stop_event is not None:
            self._stop_event.set()
        if self._runner is not None:
            self._runner.request_stop()
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle) -> RunPolicy.Result:
        """Execute policy inference by delegating to the backend runner."""
        self._active_goal = goal_handle
        self._stop_event = threading.Event()
        task = goal_handle.request.prompt
        result = RunPolicy.Result()

        self.get_logger().info(f"Starting: task='{task}' backend={self.get_parameter('backend').value}")

        feedback_stop = threading.Event()
        feedback_thread = threading.Thread(
            target=self._feedback_loop,
            args=(goal_handle, feedback_stop),
            daemon=True,
        )
        feedback_thread.start()

        try:
            run_result: RunnerResult = self._runner.run(
                self._bridge, task=task, stop_event=self._stop_event
            )
            result.success = run_result.success
            result.message = run_result.message
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'Error: {e}')
            result.success = False
            result.message = str(e)
        finally:
            feedback_stop.set()
            feedback_thread.join(timeout=1.0)

        return self._finish(goal_handle, result)

    def _feedback_loop(self, goal_handle, stop_event: threading.Event) -> None:
        """Publish runner feedback at the configured rate."""
        interval = 1.0 / self.get_parameter('feedback_rate_hz').value
        while not stop_event.wait(interval):
            if goal_handle.is_cancel_requested:
                break
            snap = self._runner.feedback()
            feedback = RunPolicy.Feedback()
            feedback.queue_depth = snap.queue_depth
            feedback.published_actions = snap.published_actions
            feedback.status = snap.status
            goal_handle.publish_feedback(feedback)

    def _finish(self, goal_handle, result: RunPolicy.Result) -> RunPolicy.Result:
        """Clean up and set goal status."""
        self._active_goal = None
        self._stop_event = None

        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        elif result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()

        self.get_logger().info(f'Finished: {result.message}')
        return result


def main(args=None):
    """Run the rosetta client node."""
    rclpy.init(args=args)
    node = RosettaClientNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

    return 0


if __name__ == '__main__':
    sys.exit(main())
