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
name from the ``framework`` parameter (e.g. ``lerobot``). The node
imports no specific policy framework.

Lifecycle transitions (fail-loud configure, stop/wait/safety ordering on
deactivate AND shutdown, one teardown path) come from
:class:`RosettaLifecycleNode`; this class only fills in the ``_setup()`` /
``_teardown()`` / ``_unblock_stop()`` / ``_send_safety_action()`` hooks and the
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

from action_msgs.srv import CancelGoal
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from std_srvs.srv import Trigger

from rosetta.contract.schema import load_contract
from rosetta.contract.sidecar import find_contract_for_pretrained, scan_inline_codec_paths
from rosetta.contract.specs import (
    iter_action_specs,
    iter_observation_specs,
    iter_reward_as_action_specs,
)
from rosetta.policies import PolicyRunner, RunnerResult, load_policy_runner
from rosetta.robots.ros2.nodes.node_utils import (
    TERMINATION_COMPLETED,
    TERMINATION_ERROR,
    TERMINATION_TIMEOUT,
    finish_goal,
    positive_rate_descriptor,
    spin_lifecycle_node,
)
from rosetta.robots.ros2.rosetta_lifecycle_node import RosettaLifecycleNode
from rosetta.robots.ros2.topic_bridge import TopicBridge
from rosetta_interfaces.action import RunPolicy
from rosetta_interfaces.srv import StartPolicy


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
                description="Policy framework to use (e.g. lerobot)",
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
        self.declare_parameter(
            "default_prompt",
            "",
            ParameterDescriptor(
                description="Task prompt used when a goal or service call leaves prompt empty. "
                "Read at each run start. For a language-conditioned policy this should match the "
                "prompt its episodes were recorded with."
            ),
        )
        self.declare_parameter(
            "default_max_duration_s",
            0.0,
            ParameterDescriptor(
                description="Maximum policy run duration in seconds (0 or negative = run until stopped, no limit). "
                "Read at each run start, and used only when the goal does not set max_duration_s."
            ),
        )

        # State (resources created in _setup)
        self._contract = None
        self._bridge: TopicBridge | None = None
        self._runner: PolicyRunner | None = None
        self._action_server: ActionServer | None = None
        self._cancel_client = None
        self._start_service = None
        self._cancel_service = None
        self._cbg = ReentrantCallbackGroup()

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
            callback_group=self._cbg,
        )

        # Client to this server's own cancel service, so ~/cancel_policy can
        # perform a real cancel. Relative name, so it expands through the same
        # node-namespace rules as the action name above.
        self._cancel_client = self.create_client(
            CancelGoal,
            "run_policy/_action/cancel_goal",
            callback_group=self._cbg,
        )

        # Service wrappers for callers that can call services but not actions
        # (Foxglove). Node-private (~/) names, matching the recorder's
        # convention.
        self._start_service = self.create_service(
            StartPolicy,
            "~/start_policy",
            self._on_start_service,
            callback_group=self._cbg,
        )
        self._cancel_service = self.create_service(
            Trigger,
            "~/cancel_policy",
            self._on_cancel_service,
            callback_group=self._cbg,
        )

        self.get_logger().info(f"Configured: contract={contract_path}, framework={framework}")

    def _teardown(self) -> None:
        """Destroy everything _setup() created. Tolerates partial state."""
        if self._action_server is not None:
            self._action_server.destroy()
            self._action_server = None
        if self._cancel_client is not None:
            self.destroy_client(self._cancel_client)
            self._cancel_client = None
        for attr in ("_start_service", "_cancel_service"):
            service = getattr(self, attr)
            if service is not None:
                self.destroy_service(service)
                setattr(self, attr, None)
        self._teardown_runner()
        if self._bridge is not None:
            self._bridge.teardown()
            self._bridge = None
        self._contract = None

    def _unblock_stop(self) -> None:
        """Also call the runner's ``request_stop()`` after the stop event is set.

        The event is the cooperative signal every control loop polls;
        ``request_stop()`` unblocks a run stuck in I/O so it observes the event
        promptly. See :meth:`PolicyRunner.request_stop`. Safe to call with no
        goal active, and -- unlike the old ``_signal_stop`` override -- never
        called while holding the base's work gate, so a framework call that
        blocks cannot stall every other claim.
        """
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

    def _on_start_service(self, request, response):
        """Start policy inference without the ROS2 action protocol.

        For Foxglove extensions and other clients that cannot route to the
        hidden _action/* services. The run happens on a background thread, so
        this returns immediately -- a service that blocked for the length of a
        policy run would hold an executor thread indefinitely.

        Returns:
            The response with accepted False and a reason when the slot is
            already claimed, else True with "Policy started".

        """
        return self._handle_start_service(response, self._run, (request.prompt or "",), what="policy run")

    def _on_cancel_service(self, _request, response: Trigger.Response) -> Trigger.Response:
        """Cancel the active policy run from a plain Trigger service call.

        Forwards to this server's own cancel service so the goal ends CANCELED,
        exactly as a `ros2 action` cancel would. See
        :meth:`RosettaLifecycleNode._cancel_current_work`.
        """
        return self._handle_cancel_service(response, self._cancel_client, what="policy run")

    def _execute(self, goal_handle) -> RunPolicy.Result:
        """Action-goal wrapper around the shared run loop."""
        return self._run(
            goal_handle.request.prompt or "",
            goal_handle,
            max_duration=goal_handle.request.max_duration_s,
        )

    def _run(self, task: str, goal_handle=None, max_duration: float = 0.0) -> RunPolicy.Result:
        """The one run loop, shared by both start paths.

        ``goal_handle=None`` marks a service start. That path skips feedback
        publishing and the terminal goal transition; the run itself, the stop
        handling, and the reason mapping are identical by construction.

        Args:
            task: Task prompt the policy is conditioned on. Empty falls back to
                the ``default_prompt`` parameter.
            goal_handle: Action goal handle, or None for a service start.
            max_duration: Stop the run after this many seconds. 0 or negative
                falls back to the ``default_max_duration_s`` parameter, which is
                what the service-start path always uses -- the lightweight
                service start takes node defaults; the goal can override them.

        Returns:
            The populated RunPolicy.Result.

        """
        # Resolved here rather than at each entry point, so the action goal,
        # the service call, and any future caller all fall back the same way.
        task = task or self.get_parameter("default_prompt").value

        result = RunPolicy.Result()
        source = "action" if goal_handle is not None else "service"
        self.get_logger().info(f"Starting ({source}): task={task!r} framework={self.get_parameter('framework').value}")

        with self._goal_work(goal_handle) as stop_event:
            feedback_stop = threading.Event()
            feedback_thread = None
            if goal_handle is not None:
                feedback_thread = threading.Thread(
                    target=self._feedback_loop,
                    args=(goal_handle, feedback_stop),
                    daemon=True,
                )
                feedback_thread.start()

            # The run itself is one blocking call into the framework, so the
            # deadline is a timer rather than a loop check. Signalling names
            # the reason and, via _unblock_stop, asks the runner to stop --
            # the same path a cancel takes.
            if max_duration <= 0.0:
                max_duration = float(self.get_parameter("default_max_duration_s").value)
            deadline = None
            if max_duration > 0.0:
                deadline = threading.Timer(max_duration, self._signal_stop, args=(TERMINATION_TIMEOUT,))
                deadline.daemon = True
                deadline.start()

            try:
                run_result: RunnerResult = self._runner.run(self._bridge, task=task, stop_event=stop_event)
                if run_result.success:
                    # The runner ran clean. Whether that was a stop somebody
                    # asked for or the control loop finishing on its own is the
                    # node's to say -- the runner only knows it was told to
                    # stop, not by whom. Its message ("Stopped", "Completed")
                    # only restates that, so it does not travel.
                    result.termination_reason = self.stop_reason or TERMINATION_COMPLETED
                else:
                    result.termination_reason = TERMINATION_ERROR
                    result.message = run_result.message
            except Exception as e:
                self.get_logger().error(f"Policy run failed:\n{traceback.format_exc()}")
                result.termination_reason = TERMINATION_ERROR
                result.message = f"{type(e).__name__}: {e}"
            finally:
                if deadline is not None:
                    deadline.cancel()
                feedback_stop.set()
                if feedback_thread is not None:
                    feedback_thread.join(timeout=1.0)

            if goal_handle is not None:
                finish_goal(goal_handle, result)
            self.get_logger().info(f"Finished ({result.termination_reason}): {result.message}")
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
