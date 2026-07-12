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
PolicyRunnerNode: framework-agnostic ROS2 action server for policy inference.

The node owns the contract, the :class:`TopicBridge` (observation/action ROS2
plumbing), and the ``run_policy`` action lifecycle. The actual policy execution
is delegated to a :class:`~rosetta.policies.PolicyRunner` resolved by
name from the ``framework`` parameter (e.g. ``lerobot``, ``vla_foundry``). The node
imports no specific policy framework.

The action is served under the relative name ``run_policy``; the launch-file
namespace determines the fully qualified name (default launch: ``/run_policy``;
the HIL launch runs two clients as ``/robot_policy/run_policy`` and
``/reward_classifier/run_policy``).

Usage:
    ros2 launch rosetta policy_runner_launch.py

    ros2 action send_goal /run_policy \
        rosetta_interfaces/action/RunPolicy "{prompt: 'pick up cube'}" --feedback
"""

from __future__ import annotations

import json
import sys
import threading

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rosetta_interfaces.action import RunPolicy

from rosetta.contract.schema import load_contract
from rosetta.contract.sidecar import resolve_repo_file
from rosetta.contract.specs import (
    iter_action_specs,
    iter_observation_specs,
    iter_reward_as_action_specs,
)
from rosetta.policies import PolicyRunner, RunnerResult, load_policy_runner
from rosetta.robots.ros2.nodes.node_utils import BusyGuard, finish_goal, is_jazzy_or_newer, wait_until
from rosetta.robots.ros2.topic_bridge import TopicBridge


class PolicyRunnerNode(LifecycleNode):
    """Backend-agnostic ROS2 Lifecycle Action Server for policy inference."""

    def __init__(self):
        """Initialize the rosetta client node and declare shared parameters."""
        if is_jazzy_or_newer():
            super().__init__("policy_runner", enable_logger_service=True)
        else:
            super().__init__("policy_runner")

        # --- Shared (framework-agnostic) parameters ---
        self.declare_parameter(
            "contract_path",
            "",
            ParameterDescriptor(description="Path to contract YAML file", read_only=True),
        )
        self.declare_parameter(
            "pretrained_name_or_path",
            "",
            ParameterDescriptor(
                description="Path or HF repo ID of the trained policy. Declared here (framework-agnostic) "
                "so it's available before contract_path is resolved -- if contract_path is empty, this is "
                "used to fall back to the contract that trained the model. Framework adapters read the same "
                "parameter for their own checkpoint loading; their own declare calls become no-ops.",
                read_only=True,
            ),
        )
        self.declare_parameter(
            "framework",
            "lerobot",
            ParameterDescriptor(
                description="Policy framework to use (lerobot, vla_foundry, ...)",
                read_only=True,
            ),
        )
        self.declare_parameter(
            "is_classifier",
            False,
            ParameterDescriptor(
                description="Publish the reward section as the action output",
                read_only=True,
            ),
        )
        self.declare_parameter(
            "feedback_rate_hz",
            2.0,
            ParameterDescriptor(description="Rate for publishing action feedback"),
        )

        # State (resources created in lifecycle callbacks)
        self._contract = None
        self._bridge: TopicBridge | None = None
        self._runner: PolicyRunner | None = None
        self._action_server: ActionServer | None = None
        self._accepting_goals = False
        # Accept-time guard: claimed in _on_goal (atomic check-and-set) so two
        # concurrent goal requests can't both be accepted under the
        # MultiThreadedExecutor before _execute runs.
        self._busy = BusyGuard()
        self._stop_event: threading.Event | None = None

        self.get_logger().info("Node created (unconfigured)")

    # -------------------- Lifecycle callbacks --------------------

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Load contract, build the bridge, resolve the framework runner."""
        contract_path = self.get_parameter("contract_path").value
        if not contract_path:
            contract_path = self._resolve_fallback_contract_path()
            if not contract_path:
                self.get_logger().error("contract_path parameter required")
                return TransitionCallbackReturn.FAILURE
            self.get_logger().info(f"Using model-embedded contract (via dataset chain): {contract_path}")

        try:
            self._contract = load_contract(contract_path)
        except Exception as e:
            self.get_logger().error(f"Failed to load contract: {e}")
            return TransitionCallbackReturn.FAILURE

        is_classifier = self.get_parameter("is_classifier").value
        obs_specs = list(iter_observation_specs(self._contract))
        if is_classifier:
            act_specs = list(iter_reward_as_action_specs(self._contract))
        else:
            act_specs = list(iter_action_specs(self._contract))

        # Build the framework-neutral topic bridge on this node.
        self._bridge = TopicBridge(obs_specs, act_specs, self._contract.fps)
        self._bridge.setup(self)

        # Resolve and prepare the framework policy runner.
        framework = self.get_parameter("framework").value
        try:
            self._runner = load_policy_runner(framework)
            self._runner.setup(self, self._contract)
        except Exception as e:
            self.get_logger().error(f"Failed to load framework '{framework}': {e}")
            return TransitionCallbackReturn.FAILURE

        self._action_server = ActionServer(
            self,
            RunPolicy,
            "run_policy",
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
            callback_group=ReentrantCallbackGroup(),
        )

        self.get_logger().info(f"Configured: contract={contract_path}, framework={framework}")
        return TransitionCallbackReturn.SUCCESS

    def _resolve_fallback_contract_path(self) -> str:
        """Chase pretrained_name_or_path -> train_config.json -> dataset -> its contract sidecar.

        Best-effort: any missing link (no checkpoint given, no train_config.json,
        no dataset reference, dataset unreachable, no contract sidecar on the
        dataset) resolves to "" and the caller falls back to today's required-
        contract_path error. Never raises.
        """
        pretrained = self.get_parameter("pretrained_name_or_path").value
        if not pretrained:
            return ""

        train_config_path = resolve_repo_file(pretrained, "train_config.json", repo_type="model")
        if train_config_path is None:
            return ""

        try:
            train_config = json.loads(train_config_path.read_text())
            dataset_repo_id = train_config["dataset"]["repo_id"]
            dataset_root = train_config["dataset"].get("root")
        except Exception as e:
            self.get_logger().warning(f"Failed to read dataset reference from {train_config_path}: {e}")
            return ""

        contract_path = resolve_repo_file(
            dataset_root or dataset_repo_id, "meta/rosetta_contract.yaml", repo_type="dataset"
        )
        return str(contract_path) if contract_path is not None else ""

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Activate lifecycle publishers and accept goals."""
        self._accepting_goals = True
        self.get_logger().info("Activated and ready for policy execution")
        return super().on_activate(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Stop the runner, wait for the goal to end, then send the safety action."""
        self._stop_and_secure()
        self.get_logger().info("Deactivated")
        return super().on_deactivate(state)

    def _stop_and_secure(self, wait_timeout: float = 5.0) -> None:
        """Stop goal acceptance and the runner, then send the safety action.

        Order matters: the safety action must be the LAST command on the wire.
        Sending it before stopping the runner let the still-running policy loop
        publish at least one more action over it, leaving a stale policy
        command as the robot's final input. Runs before super().on_deactivate()
        so the lifecycle publishers are still active for the safety send.
        """
        self._accepting_goals = False

        # 1. Stop the runner first so no policy action can land after safety.
        if self._stop_event is not None:
            self._stop_event.set()
        if self._runner is not None:
            self._runner.request_stop()

        # 2. Bounded wait for the in-progress goal to finish.
        if not wait_until(lambda: not self._busy.busy, timeout=wait_timeout):
            self.get_logger().warning("Goal did not stop within timeout; sending safety action anyway")

        # 3. Safety action last.
        if self._bridge is not None:
            self._bridge.send_safety_action()

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

        self.get_logger().info("Cleaned up")
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

        self.get_logger().info("Shutdown complete")
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Handle errors by cleaning up resources."""
        self.get_logger().error(f"Error occurred in state: {state.label}")
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
        except Exception as e:
            self.get_logger().error(f"Error during error handling: {e}")
        return TransitionCallbackReturn.SUCCESS

    def _teardown_runner(self) -> None:
        if self._runner is not None:
            try:
                self._runner.teardown()
            except Exception as e:
                self.get_logger().warning(f"Runner teardown error: {e}")

    # -------------------- Action callbacks --------------------

    def _on_goal(self, _goal_request) -> GoalResponse:
        """Accept or reject a client request to begin an action."""
        self.get_logger().info("Received goal request")
        if not self._accepting_goals:
            self.get_logger().warning("Rejected: node not active")
            return GoalResponse.REJECT
        if not self._busy.try_acquire():
            self.get_logger().warning("Rejected: already running")
            return GoalResponse.REJECT
        self.get_logger().info("Goal accepted")
        return GoalResponse.ACCEPT

    def _on_cancel(self, _goal_handle) -> CancelResponse:
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info("Received cancel request")
        if self._stop_event is not None:
            self._stop_event.set()
        if self._runner is not None:
            self._runner.request_stop()
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle) -> RunPolicy.Result:
        """Execute policy inference by delegating to the framework runner."""
        self._stop_event = threading.Event()
        task = goal_handle.request.prompt
        result = RunPolicy.Result()

        self.get_logger().info(f"Starting: task='{task}' framework={self.get_parameter('framework').value}")

        feedback_stop = threading.Event()
        feedback_thread = threading.Thread(
            target=self._feedback_loop,
            args=(goal_handle, feedback_stop),
            daemon=True,
        )
        feedback_thread.start()

        try:
            try:
                run_result: RunnerResult = self._runner.run(self._bridge, task=task, stop_event=self._stop_event)
                result.success = run_result.success
                result.message = run_result.message
            except Exception as e:
                self.get_logger().error(f"Error: {e}")
                result.success = False
                result.message = str(e)
            finally:
                feedback_stop.set()
                feedback_thread.join(timeout=1.0)

            finish_goal(goal_handle, result)
            self.get_logger().info(f"Finished: {result.message}")
            return result
        finally:
            # Release on EVERY exit path — a leaked guard would reject all
            # subsequent goals until node restart.
            self._stop_event = None
            self._busy.release()

    def _feedback_loop(self, goal_handle, stop_event: threading.Event) -> None:
        """Publish runner feedback at the configured rate."""
        interval = 1.0 / self.get_parameter("feedback_rate_hz").value
        while not stop_event.wait(interval):
            if goal_handle.is_cancel_requested:
                break
            snap = self._runner.feedback()
            feedback = RunPolicy.Feedback()
            feedback.queue_depth = snap.queue_depth
            feedback.published_actions = snap.published_actions
            feedback.status = snap.status
            goal_handle.publish_feedback(feedback)


def main(args=None):
    """Run the rosetta client node."""
    rclpy.init(args=args)
    node = PolicyRunnerNode()

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


if __name__ == "__main__":
    sys.exit(main())
