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

Lifecycle transitions (fail-loud configure, stop/wait/safety ordering on
deactivate AND shutdown, one teardown path) come from
:class:`RosettaLifecycleNode`; this class only fills in the ``_setup()`` /
``_teardown()`` / ``_signal_stop()`` / ``_send_safety_action()`` hooks and the
action callbacks.

The action is served under the relative name ``run_policy``. The launch-file
namespace sets the fully qualified name. The default launch gives
``/run_policy``. The HIL launch runs two clients as ``/robot_policy/run_policy``
and ``/reward_classifier/run_policy``.

Usage:
    ros2 launch rosetta policy_runner_launch.py

    ros2 action send_goal /run_policy \
        rosetta_interfaces/action/RunPolicy "{prompt: 'pick up cube'}" --feedback
"""

from __future__ import annotations

import sys
import threading
import traceback

from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rosetta_interfaces.action import RunPolicy

from rosetta.contract.schema import load_contract
from rosetta.contract.sidecar import find_contract_for_pretrained, scan_inline_codec_paths
from rosetta.contract.specs import (
    iter_action_specs,
    iter_observation_specs,
    iter_reward_as_action_specs,
)
from rosetta.policies import PolicyRunner, RunnerResult, load_policy_runner
from rosetta.robots.ros2.nodes.node_utils import (
    finish_goal,
    positive_rate_descriptor,
    spin_lifecycle_node,
)
from rosetta.robots.ros2.rosetta_lifecycle_node import RosettaLifecycleNode
from rosetta.robots.ros2.topic_bridge import TopicBridge


class PolicyRunnerNode(RosettaLifecycleNode):
    """Framework-agnostic lifecycle action server for policy inference."""

    def __init__(self, **kwargs):
        """Declare the shared (framework-agnostic) parameters."""
        super().__init__("policy_runner", **kwargs)

        self.declare_parameter(
            "contract_path",
            "",
            ParameterDescriptor(description="Path to contract YAML file", read_only=True),
        )
        # This default, and is_classifier's below, are duplicated as inert
        # standalone-use fallbacks in each framework adapter's own setup()
        # (e.g. lerobot_rosetta/policy_runner.py). Keep both in sync.
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
            positive_rate_descriptor("Rate for publishing action feedback"),
        )

        # State (resources created in _setup)
        self._contract = None
        self._bridge: TopicBridge | None = None
        self._runner: PolicyRunner | None = None
        self._action_server: ActionServer | None = None

        self.get_logger().info("Node created (unconfigured)")

    # -------------------- Lifecycle hooks --------------------

    def _setup(self) -> None:
        """Load the contract, build the bridge, resolve the framework runner.

        Any failure raises. The base logs the traceback and routes through
        on_error, which tears down whatever this method partially built.
        """
        contract_path = self.get_parameter("contract_path").value
        if not contract_path:
            pretrained = self.get_parameter("pretrained_name_or_path").value
            fallback = find_contract_for_pretrained(pretrained, warn=self.get_logger().warning)
            if fallback is None:
                if pretrained:
                    raise ValueError(
                        f"contract_path parameter required: no contract sidecar found via "
                        f"pretrained_name_or_path={pretrained!r} (see warnings above)"
                    )
                raise ValueError("contract_path parameter required")
            contract_path = str(fallback)
            self.get_logger().info(f"Using model-embedded contract (via dataset chain): {contract_path}")
            # A hub-resolved contract is third-party config, and loading it
            # imports the code its inline codec paths name. Surface each one
            # BEFORE load_contract runs those imports.
            for role, path in scan_inline_codec_paths(contract_path):
                self.get_logger().warning(
                    f"Hub-resolved contract declares an inline {role} '{path}': loading this "
                    f"contract imports and executes that installed code. Pass an explicit "
                    f"contract_path to use a locally reviewed contract."
                )

        self._contract = load_contract(contract_path)

        is_classifier = self.get_parameter("is_classifier").value
        obs_specs = list(iter_observation_specs(self._contract))
        if is_classifier:
            act_specs = list(iter_reward_as_action_specs(self._contract))
        else:
            act_specs = list(iter_action_specs(self._contract))

        self._bridge = TopicBridge(obs_specs, act_specs, self._contract.fps)
        self._bridge.setup(self)

        # Assign self._runner before calling setup() on it: a setup() that
        # raises partway leaves a half-initialized runner that on_error's
        # _teardown() must still reach in order to release it.
        framework = self.get_parameter("framework").value
        self._runner = load_policy_runner(framework)
        self._runner.setup(self, self._contract)

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

    def _teardown(self) -> None:
        """Destroy everything _setup() created. Tolerates partial state."""
        if self._action_server is not None:
            self._action_server.destroy()
            self._action_server = None
        self._teardown_runner()
        if self._bridge is not None:
            self._bridge.teardown()
            self._bridge = None
        self._contract = None

    def _signal_stop(self) -> None:
        """Set the stop event, then also call the runner's ``request_stop()``.

        Event first, both always. The event is the cooperative signal every
        control loop polls. ``request_stop()`` unblocks a run stuck in I/O so
        it observes the event promptly. See :meth:`PolicyRunner.request_stop`.
        Safe to call with no goal active.
        """
        super()._signal_stop()
        runner = self._runner
        if runner is not None:
            runner.request_stop()

    def _send_safety_action(self) -> None:
        bridge = self._bridge
        if bridge is not None:
            bridge.send_safety_action()

    def _teardown_runner(self) -> None:
        # Null self._runner first so a second call is a clean no-op even when
        # teardown() raises. The exception then propagates to the base's
        # guarded transition handler. A framework resource that failed to
        # release (e.g. a policy-server subprocess) is a real leak, so let it
        # surface instead of swallowing it here.
        runner, self._runner = self._runner, None
        if runner is not None:
            runner.teardown()

    # -------------------- Action callbacks --------------------
    # Goal/cancel callbacks come from the base: claim-or-reject, goal-id-routed cancel.

    def _execute(self, goal_handle) -> RunPolicy.Result:
        """Execute policy inference by delegating to the framework runner."""
        task = goal_handle.request.prompt
        result = RunPolicy.Result()

        self.get_logger().info(f"Starting: task={task!r} framework={self.get_parameter('framework').value}")

        with self._goal_work(goal_handle) as stop_event:
            feedback_stop = threading.Event()
            feedback_thread = threading.Thread(
                target=self._feedback_loop,
                args=(goal_handle, feedback_stop),
                daemon=True,
            )
            feedback_thread.start()

            try:
                run_result: RunnerResult = self._runner.run(self._bridge, task=task, stop_event=stop_event)
                result.success = run_result.success
                result.message = run_result.message
            except Exception as e:
                self.get_logger().error(f"Policy run failed:\n{traceback.format_exc()}")
                result.success = False
                result.message = f"{type(e).__name__}: {e}"
            finally:
                feedback_stop.set()
                feedback_thread.join(timeout=1.0)

            finish_goal(goal_handle, result)
            self.get_logger().info(f"Finished: {result.message}")
            return result

    def _feedback_loop(self, goal_handle, stop_event: threading.Event) -> None:
        """Publish runner feedback at the configured rate."""
        interval = 1.0 / self.get_parameter("feedback_rate_hz").value
        while not stop_event.wait(interval):
            if goal_handle.is_cancel_requested:
                break
            try:
                snap = self._runner.feedback()
                feedback = RunPolicy.Feedback()
                feedback.queue_depth = snap.queue_depth
                feedback.published_actions = snap.published_actions
                feedback.status = snap.status
                goal_handle.publish_feedback(feedback)
            except Exception:
                # A silent death of this daemon thread would end feedback
                # while the goal runs on. Log loudly, then stop publishing.
                self.get_logger().error(f"Feedback loop stopped:\n{traceback.format_exc()}")
                break


def main(args=None):
    """Run the policy runner node."""
    return spin_lifecycle_node(PolicyRunnerNode, args=args)


if __name__ == "__main__":
    sys.exit(main())
