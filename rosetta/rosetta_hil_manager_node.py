#!/usr/bin/env python3
# Copyright 2026 Brian Blankenau
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
RosettaHilManagerNode: Human-in-the-Loop episode orchestrator.

Coordinates robot policy inference, bag recording, reward classification, and
teleop muxing. Exposes a ManageEpisode action for unified episode control.

The node acts as an orchestrator:
- Action client to rosetta_client_node(s) for policy inference (RunPolicy)
- Action client to episode_recorder_node for bag recording (RecordEpisode)
- Muxes between policy output and teleop input for seamless human takeover
- Muxes between reward classifier output and human reward overrides
- Monitors episode termination conditions (timeout, human stop, reward threshold)

Usage:
    ros2 launch rosetta rosetta_hil_launch.py

    ros2 action send_goal /hil_manager/manage_episode \\
        rosetta_interfaces/action/ManageEpisode \\
        "{prompt: 'pick up cube', max_duration_s: 90.0}" --feedback
"""

from __future__ import annotations

import sys
import threading
import time

import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rcl_interfaces.msg import ParameterDescriptor
from rosidl_runtime_py.utilities import get_message

from std_msgs.msg import Int8
from std_srvs.srv import SetBool, Trigger

from rosetta.common.contract import load_contract
from rosetta.common.ros2_utils import qos_profile_from_dict

from rosetta_interfaces.action import ManageEpisode, RecordEpisode, RunPolicy
from rosetta_interfaces.srv import StartHILEpisode


# ---------- Helpers ----------


def _resolve_selector(obj, path: str):
    """Resolve a dotted selector, supporting numeric array indices.

    Unlike dot_get, this handles paths like "buttons.5" -> obj.buttons[5],
    which is needed for Joy message event mappings.
    """
    cur = obj
    for p in path.split("."):
        if p.isdigit() and hasattr(cur, "__getitem__"):
            cur = cur[int(p)]
        else:
            cur = getattr(cur, p)
    return cur


# ---------- Constants ----------

ACTION_CLIENT_TIMEOUT_SEC = 10.0
GOAL_RESPONSE_TIMEOUT_SEC = 10.0
CANCEL_TIMEOUT_SEC = 5.0
RESULT_TIMEOUT_SEC = 10.0
FUTURE_POLL_SEC = 0.01


def _wait_for_future(future, timeout_sec: float) -> bool:
    """Poll a future until done or timeout. Returns True if completed."""
    deadline = time.time() + timeout_sec
    while not future.done():
        if time.time() >= deadline:
            return False
        time.sleep(FUTURE_POLL_SEC)
    return True


class RosettaHilManagerNode(LifecycleNode):
    """ROS2 Lifecycle node that orchestrates HIL episodes.

    Coordinates robot policy, reward classifier, episode recorder, and teleop
    muxing through a single ManageEpisode action interface.
    """

    def __init__(self):
        super().__init__("hil_manager", enable_logger_service=True)

        # -------------------- Parameters --------------------
        self.declare_parameter(
            "contract_path", "",
            ParameterDescriptor(description="Path to HIL contract YAML file", read_only=True)
        )
        self.declare_parameter(
            "enable_reward_classifier", False,
            ParameterDescriptor(
                description="Enable reward classifier policy (requires separate contract + model)",
                read_only=True,
            )
        )
        self.declare_parameter(
            "policy_action_name", "/robot_policy/run_policy",
            ParameterDescriptor(description="Action name for robot policy client", read_only=True)
        )
        self.declare_parameter(
            "reward_classifier_action_name", "/reward_classifier/run_policy",
            ParameterDescriptor(
                description="Action name for reward classifier client", read_only=True
            )
        )
        self.declare_parameter(
            "recorder_action_name", "/record_episode",
            ParameterDescriptor(description="Action name for episode recorder client", read_only=True)
        )
        self.declare_parameter(
            "policy_remap_prefix", "/hil/policy",
            ParameterDescriptor(
                description="Topic prefix for remapped policy output", read_only=True
            )
        )
        self.declare_parameter(
            "reward_remap_prefix", "/hil/reward",
            ParameterDescriptor(
                description="Topic prefix for remapped reward classifier output", read_only=True
            )
        )
        self.declare_parameter(
            "human_reward_positive", 1.0,
            ParameterDescriptor(description="Reward value for human positive override")
        )
        self.declare_parameter(
            "human_reward_negative", 0.0,
            ParameterDescriptor(description="Reward value for human negative override")
        )
        self.declare_parameter(
            "feedback_rate_hz", 30.0,
            ParameterDescriptor(description="Rate for publishing ManageEpisode feedback")
        )

        # -------------------- State --------------------
        self._contract = None
        self._accepting_goals = False

        # Action clients
        self._policy_client: ActionClient | None = None
        self._reward_client: ActionClient | None = None
        self._recorder_client: ActionClient | None = None

        # Action server
        self._action_server: ActionServer | None = None

        # Mux state (guarded by _mux_lock)
        self._mux_lock = threading.Lock()
        self._control_source = "policy"  # "policy" or "teleop"
        self._current_reward = 0.0
        self._human_reward_override = False
        self._stop_requested = False
        self._start_requested = False

        # Subscriptions and publishers for muxing
        self._mux_subs: list = []
        self._command_publishers: dict[str, tuple] = {}  # original_topic -> (msg_cls, publisher)
        self._reward_publishers: dict[str, tuple] = {}   # original_topic -> (msg_cls, publisher)
        self._intervention_pub = None
        self._episode_service = None
        self._stop_service = None
        self._intervention_service = None
        self._reward_override_service = None
        self._clear_reward_service = None

        # Active episode goal handles for child actions
        self._policy_goal_handle = None
        self._reward_goal_handle = None
        self._recorder_goal_handle = None

        # Callback groups
        self._action_cbg = ReentrantCallbackGroup()
        self._sub_cbg = ReentrantCallbackGroup()

        self.get_logger().info("Node created (unconfigured)")

    # ====================================================================
    # Lifecycle callbacks
    # ====================================================================

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Load contract, create action clients, subscriptions, publishers, and action server."""
        contract_path = self.get_parameter("contract_path").value
        if not contract_path:
            self.get_logger().error("contract_path parameter required")
            return TransitionCallbackReturn.FAILURE

        try:
            self._contract = load_contract(contract_path)
        except Exception as e:
            self.get_logger().error(f"Failed to load contract: {e}")
            return TransitionCallbackReturn.FAILURE

        enable_reward = self.get_parameter("enable_reward_classifier").value
        policy_remap_prefix = self.get_parameter("policy_remap_prefix").value
        reward_remap_prefix = self.get_parameter("reward_remap_prefix").value

        # --- Action clients ---
        policy_action = self.get_parameter("policy_action_name").value
        self._policy_client = ActionClient(
            self, RunPolicy, policy_action, callback_group=self._action_cbg
        )

        recorder_action = self.get_parameter("recorder_action_name").value
        self._recorder_client = ActionClient(
            self, RecordEpisode, recorder_action, callback_group=self._action_cbg
        )

        if enable_reward:
            reward_action = self.get_parameter("reward_classifier_action_name").value
            self._reward_client = ActionClient(
                self, RunPolicy, reward_action, callback_group=self._action_cbg
            )

        # --- Mux: subscribe to remapped policy output, publish to real command topic ---
        for action_spec in self._contract.actions:
            original_topic = action_spec.publish_topic
            remapped_topic = policy_remap_prefix + original_topic
            msg_cls = get_message(action_spec.type)
            qos = qos_profile_from_dict(action_spec.publish_qos) or 10

            # Publisher to the real command topic
            pub = self.create_publisher(msg_cls, original_topic, qos)
            self._command_publishers[original_topic] = (msg_cls, pub)

            # Subscribe to the remapped policy output
            sub = self.create_subscription(
                msg_cls, remapped_topic,
                lambda msg, topic=original_topic: self._on_policy_output(msg, topic),
                qos, callback_group=self._sub_cbg,
            )
            self._mux_subs.append(sub)
            self.get_logger().info(f"Action mux: {remapped_topic} -> {original_topic}")

        # --- Mux: subscribe to teleop input, publish to real command topic ---
        if self._contract.teleop:
            for teleop_input in self._contract.teleop.inputs:
                msg_cls = get_message(teleop_input.type)
                qos = qos_profile_from_dict(teleop_input.qos) or 10
                sub = self.create_subscription(
                    msg_cls, teleop_input.topic,
                    self._on_teleop_input,
                    qos, callback_group=self._sub_cbg,
                )
                self._mux_subs.append(sub)
                self.get_logger().info(f"Teleop input: {teleop_input.topic}")

        # --- Reward publishers (always created for human reward buttons) ---
        if self._contract.rewards:
            for reward_spec in self._contract.rewards:
                original_topic = reward_spec.topic
                msg_cls = get_message(reward_spec.type)
                qos = qos_profile_from_dict(reward_spec.qos) or 10

                pub = self.create_publisher(msg_cls, original_topic, qos)
                self._reward_publishers[original_topic] = (msg_cls, pub)
                self.get_logger().info(f"Reward publisher: {original_topic}")

        # --- Mux: subscribe to remapped reward classifier output ---
        if enable_reward and self._contract.rewards:
            for reward_spec in self._contract.rewards:
                original_topic = reward_spec.topic
                remapped_topic = reward_remap_prefix + original_topic
                msg_cls = get_message(reward_spec.type)
                qos = qos_profile_from_dict(reward_spec.qos) or 10

                sub = self.create_subscription(
                    msg_cls, remapped_topic,
                    lambda msg, topic=original_topic: self._on_reward_classifier_output(msg, topic),
                    qos, callback_group=self._sub_cbg,
                )
                self._mux_subs.append(sub)
                self.get_logger().info(f"Reward mux: {remapped_topic} -> {original_topic}")

        # --- Subscribe to teleop events ---
        if self._contract.teleop and self._contract.teleop.events:
            events_spec = self._contract.teleop.events
            msg_cls = get_message(events_spec.msg_type)
            qos = qos_profile_from_dict(events_spec.qos) or 10
            sub = self.create_subscription(
                msg_cls, events_spec.topic,
                lambda msg: self._on_teleop_events(msg, events_spec),
                qos, callback_group=self._sub_cbg,
            )
            self._mux_subs.append(sub)
            self.get_logger().info(f"Teleop events: {events_spec.topic}")

        # --- HIL intervention publisher ---
        self._intervention_pub = self.create_publisher(Int8, "hil_intervention", 10)

        # --- Service wrapper (for callers that don't support actions) ---
        self._episode_service = self.create_service(
            StartHILEpisode, "start_episode", self._handle_start_episode,
            callback_group=self._action_cbg,
        )
        self._stop_service = self.create_service(
            Trigger, "stop_episode", self._handle_stop_episode,
            callback_group=self._action_cbg,
        )
        self._intervention_service = self.create_service(
            SetBool, "set_intervention", self._handle_set_intervention,
            callback_group=self._action_cbg,
        )
        self._reward_override_service = self.create_service(
            SetBool, "set_reward_override", self._handle_set_reward_override,
            callback_group=self._action_cbg,
        )
        self._clear_reward_service = self.create_service(
            Trigger, "clear_reward_override", self._handle_clear_reward_override,
            callback_group=self._action_cbg,
        )

        # --- Action server ---
        self._action_server = ActionServer(
            self,
            ManageEpisode,
            "manage_episode",
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
            callback_group=self._action_cbg,
        )

        self.get_logger().info(
            f"Configured: robot_type={self._contract.robot_type}, "
            f"reward_classifier={'enabled' if enable_reward else 'disabled'}, "
            f"actions={len(self._contract.actions)}, "
            f"teleop={'yes' if self._contract.teleop else 'no'}"
        )
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Enable goal acceptance."""
        self._accepting_goals = True
        self.get_logger().info("Activated and ready for HIL episodes")
        return super().on_activate(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Stop accepting goals and cancel any active episode."""
        self._accepting_goals = False

        # Signal active episode to stop
        with self._mux_lock:
            self._stop_requested = True

        # Wait for episode to complete
        timeout = 10.0
        start = time.time()
        while self._policy_goal_handle is not None and (time.time() - start) < timeout:
            time.sleep(0.1)

        self.get_logger().info("Deactivated")
        return super().on_deactivate(state)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Release resources."""
        self._destroy_resources()
        self.get_logger().info("Cleaned up")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Final cleanup before destruction."""
        self._accepting_goals = False
        with self._mux_lock:
            self._stop_requested = True
        self._destroy_resources()
        self.get_logger().info("Shutdown complete")
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Handle errors by cleaning up resources."""
        self.get_logger().error(f"Error occurred in state: {state.label}")
        try:
            self._accepting_goals = False
            with self._mux_lock:
                self._stop_requested = True
            self._destroy_resources()
        except Exception as e:
            self.get_logger().error(f"Error during cleanup: {e}")
        return TransitionCallbackReturn.SUCCESS

    def _destroy_resources(self) -> None:
        """Destroy subscriptions, publishers, and action clients/server."""
        for sub in self._mux_subs:
            self.destroy_subscription(sub)
        self._mux_subs = []

        for _, pub in self._command_publishers.values():
            self.destroy_publisher(pub)
        self._command_publishers = {}

        for _, pub in self._reward_publishers.values():
            self.destroy_publisher(pub)
        self._reward_publishers = {}

        if self._policy_client is not None:
            self._policy_client.destroy()
            self._policy_client = None

        if self._reward_client is not None:
            self._reward_client.destroy()
            self._reward_client = None

        if self._recorder_client is not None:
            self._recorder_client.destroy()
            self._recorder_client = None

        if self._action_server is not None:
            self.destroy_action_server(self._action_server)
            self._action_server = None

        if self._intervention_pub is not None:
            self.destroy_publisher(self._intervention_pub)
            self._intervention_pub = None

        if self._episode_service is not None:
            self.destroy_service(self._episode_service)
            self._episode_service = None

        if self._stop_service is not None:
            self.destroy_service(self._stop_service)
            self._stop_service = None

        if self._intervention_service is not None:
            self.destroy_service(self._intervention_service)
            self._intervention_service = None

        if self._reward_override_service is not None:
            self.destroy_service(self._reward_override_service)
            self._reward_override_service = None

        if self._clear_reward_service is not None:
            self.destroy_service(self._clear_reward_service)
            self._clear_reward_service = None

        self._contract = None

    # ====================================================================
    # Mux callbacks
    # ====================================================================

    def _on_policy_output(self, msg, original_topic: str) -> None:
        """Forward policy output to command topic when in policy mode."""
        with self._mux_lock:
            if self._control_source != "policy":
                return

        if original_topic in self._command_publishers:
            _, pub = self._command_publishers[original_topic]
            pub.publish(msg)

    def _on_teleop_input(self, msg) -> None:
        """Forward teleop input to all command topics when in teleop mode."""
        with self._mux_lock:
            if self._control_source != "teleop":
                return

        # Forward to all command publishers (teleop input maps to command output)
        for _, pub in self._command_publishers.values():
            pub.publish(msg)

    def _on_reward_classifier_output(self, msg, original_topic: str) -> None:
        """Forward reward classifier output when no human override is active."""
        with self._mux_lock:
            if self._human_reward_override:
                return
            self._current_reward = float(msg.data)

        if original_topic in self._reward_publishers:
            _, pub = self._reward_publishers[original_topic]
            pub.publish(msg)

    def _on_teleop_events(self, msg, events_spec) -> None:
        """Handle teleop event buttons (intervention, stop, reward override).

        Event names follow the LeRobot TeleopEvents convention:
          is_intervention  - toggle between policy and teleop control
          terminate_episode - stop the current episode
          rerecord_episode  - request episode restart
          success           - human positive reward override
          failure           - human negative reward override
        """
        self.get_logger().debug(f"Joy received: buttons={list(msg.buttons)}")
        for event_name, selector in events_spec.mappings.items():
            try:
                value = _resolve_selector(msg, selector)
            except (AttributeError, IndexError, ValueError) as e:
                self.get_logger().warning(f"Selector '{selector}' failed: {e}")
                continue

            if not bool(value):
                # Button not pressed - check for release events
                if event_name == "is_intervention":
                    with self._mux_lock:
                        if self._control_source == "teleop":
                            self._control_source = "policy"
                            self.get_logger().info("Mux: teleop -> policy (intervention released)")
                elif event_name in ("success", "failure"):
                    with self._mux_lock:
                        if self._human_reward_override:
                            self._human_reward_override = False
                            self.get_logger().info("Human reward override released")
                continue

            # Button pressed
            if event_name == "is_intervention":
                with self._mux_lock:
                    if self._control_source != "teleop":
                        self._control_source = "teleop"
                        self.get_logger().info("Mux: policy -> teleop (human intervention)")

            elif event_name == "terminate_episode":
                with self._mux_lock:
                    self._stop_requested = True
                self.get_logger().info("Stop requested by human overseer")

            elif event_name == "rerecord_episode":
                with self._mux_lock:
                    self._start_requested = True

            elif event_name == "success":
                reward_val = self.get_parameter("human_reward_positive").value
                with self._mux_lock:
                    self._current_reward = reward_val
                    self._human_reward_override = True
                self._publish_human_reward(reward_val)
                self.get_logger().info(f"Human reward override: {reward_val}")

            elif event_name == "failure":
                reward_val = self.get_parameter("human_reward_negative").value
                with self._mux_lock:
                    self._current_reward = reward_val
                    self._human_reward_override = True
                self._publish_human_reward(reward_val)
                self.get_logger().info(f"Human reward override: {reward_val}")

    def _publish_human_reward(self, reward_val: float) -> None:
        """Publish a human-overridden reward value to all reward topics."""
        for original_topic, (msg_cls, pub) in self._reward_publishers.items():
            msg = msg_cls()
            msg.data = reward_val
            pub.publish(msg)

    # ====================================================================
    # Action server callbacks
    # ====================================================================

    def _on_goal(self, _goal_request) -> GoalResponse:
        """Accept or reject a ManageEpisode goal."""
        self.get_logger().info("Received ManageEpisode goal request")
        if not self._accepting_goals:
            self.get_logger().warning("Rejected: node not active")
            return GoalResponse.REJECT
        if self._policy_goal_handle is not None:
            self.get_logger().warning("Rejected: episode already in progress")
            return GoalResponse.REJECT
        self.get_logger().info("Goal accepted")
        return GoalResponse.ACCEPT

    def _on_cancel(self, _goal_handle) -> CancelResponse:
        """Accept cancel request for ManageEpisode."""
        self.get_logger().info("Received cancel request")
        with self._mux_lock:
            self._stop_requested = True
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle) -> ManageEpisode.Result:
        """Execute a full HIL episode via the action interface."""
        prompt = goal_handle.request.prompt or ""
        max_duration = goal_handle.request.max_duration_s
        reward_threshold = goal_handle.request.success_reward_threshold

        fields, cancelled = self._run_episode(prompt, max_duration, reward_threshold)

        result = ManageEpisode.Result()
        result.success = fields["success"]
        result.message = fields["message"]
        result.termination_reason = fields["termination_reason"]
        result.final_reward = fields["final_reward"]
        result.bag_path = fields["bag_path"]
        result.messages_written = fields["messages_written"]

        if cancelled:
            goal_handle.canceled()
        elif result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()

        return result

    def _handle_start_episode(
        self, request: StartHILEpisode.Request, response: StartHILEpisode.Response
    ) -> StartHILEpisode.Response:
        """Service wrapper: run a full HIL episode and return the result."""
        if not self._accepting_goals:
            response.success = False
            response.message = "Node not active"
            return response
        if self._policy_goal_handle is not None:
            response.success = False
            response.message = "Episode already in progress"
            return response

        fields, _ = self._run_episode(
            request.prompt or "",
            request.max_duration_s,
            request.success_reward_threshold,
        )

        response.success = fields["success"]
        response.message = fields["message"]
        response.termination_reason = fields["termination_reason"]
        response.final_reward = fields["final_reward"]
        response.bag_path = fields["bag_path"]
        response.messages_written = fields["messages_written"]
        return response

    def _handle_stop_episode(self, _request, response: Trigger.Response) -> Trigger.Response:
        """Service: set stop_requested so the active episode exits its feedback loop."""
        with self._mux_lock:
            self._stop_requested = True
        response.success = True
        response.message = "Stop requested"
        self.get_logger().info("Stop requested via service")
        return response

    def _handle_set_intervention(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        """Service: switch mux between policy (False) and teleop (True)."""
        with self._mux_lock:
            self._control_source = "teleop" if request.data else "policy"
        response.success = True
        response.message = f"Control source: {'teleop' if request.data else 'policy'}"
        self.get_logger().info(response.message)
        return response

    def _handle_set_reward_override(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        """Service: apply a human reward override. True = positive, False = negative."""
        reward_val = (
            self.get_parameter("human_reward_positive").value
            if request.data
            else self.get_parameter("human_reward_negative").value
        )
        with self._mux_lock:
            self._current_reward = reward_val
            self._human_reward_override = True
        self._publish_human_reward(reward_val)
        response.success = True
        response.message = f"Reward override set to {reward_val}"
        self.get_logger().info(response.message)
        return response

    def _handle_clear_reward_override(
        self, _request, response: Trigger.Response
    ) -> Trigger.Response:
        """Service: release the human reward override."""
        with self._mux_lock:
            self._human_reward_override = False
        response.success = True
        response.message = "Reward override cleared"
        self.get_logger().info(response.message)
        return response

    def _run_episode(
        self, prompt: str, max_duration: float, reward_threshold: float
    ) -> tuple[dict, bool]:
        """Core HIL episode logic shared by the action and service interfaces.

        Returns:
            (fields dict, cancelled bool) where fields contains result values.
        """
        if max_duration <= 0.0:
            max_duration = self._contract.max_duration_s

        self.get_logger().info(
            f"Starting episode: prompt='{prompt}', max_duration={max_duration}s, "
            f"reward_threshold={reward_threshold}"
        )

        with self._mux_lock:
            self._control_source = "policy"
            self._current_reward = 0.0
            self._human_reward_override = False
            self._stop_requested = False
            self._start_requested = False

        enable_reward = self.get_parameter("enable_reward_classifier").value
        cancelled = False
        fields = {
            "success": False,
            "message": "",
            "termination_reason": "",
            "final_reward": 0.0,
            "bag_path": "",
            "messages_written": 0,
        }

        try:
            recorder_gh = self._start_recorder(prompt)
            if recorder_gh is None:
                fields["message"] = "Failed to start episode recorder"
                return fields, cancelled
            self._recorder_goal_handle = recorder_gh

            policy_gh = self._start_policy(prompt)
            if policy_gh is None:
                fields["message"] = "Failed to start robot policy"
                self._cancel_recorder()
                return fields, cancelled
            self._policy_goal_handle = policy_gh

            reward_gh = None
            if enable_reward and self._reward_client is not None:
                reward_gh = self._start_reward_classifier(prompt)
                if reward_gh is None:
                    self.get_logger().warning(
                        "Failed to start reward classifier, continuing without it"
                    )
                self._reward_goal_handle = reward_gh

            termination_reason = self._feedback_loop(None, max_duration, reward_threshold)
            cancelled = termination_reason == "cancelled"

            self.get_logger().info(f"Episode ending: {termination_reason}")
            self._cancel_policy()
            if reward_gh is not None:
                self._cancel_reward_classifier()

            recorder_result = self._stop_recorder()

            with self._mux_lock:
                final_reward = self._current_reward

            fields["termination_reason"] = termination_reason
            fields["final_reward"] = final_reward
            if recorder_result is not None:
                fields["bag_path"] = recorder_result.bag_path
                fields["messages_written"] = recorder_result.messages_written

            if not cancelled:
                fields["success"] = True
                fields["message"] = f"Episode completed: {termination_reason}"
            else:
                fields["message"] = "Cancelled"

        except Exception as e:
            self.get_logger().error(f"Episode error: {e}")
            fields["message"] = str(e)
            self._cancel_all_children()

        finally:
            self._policy_goal_handle = None
            self._reward_goal_handle = None
            self._recorder_goal_handle = None

        self.get_logger().info(f"Episode finished: {fields['message']}")
        return fields, cancelled

    # ====================================================================
    # Feedback loop
    # ====================================================================

    def _feedback_loop(
        self,
        goal_handle,
        max_duration: float,
        reward_threshold: float,
    ) -> str:
        """Run the episode feedback loop until a termination condition is met.

        Returns:
            Termination reason string.
        """
        feedback_interval = 1.0 / self.get_parameter("feedback_rate_hz").value
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time

            # Check termination conditions
            with self._mux_lock:
                stop = self._stop_requested
                reward = self._current_reward
                source = self._control_source

            if goal_handle is not None and goal_handle.is_cancel_requested:
                return "cancelled"

            if stop:
                return "human_stop"

            if elapsed >= max_duration:
                return "timeout"

            if reward_threshold > 0.0 and reward >= reward_threshold:
                return "reward_threshold"

            # Publish HIL intervention state (0=policy, 1=human)
            if self._intervention_pub is not None:
                intervention_msg = Int8()
                intervention_msg.data = 0 if source == "policy" else 1
                self._intervention_pub.publish(intervention_msg)

            # Publish feedback (only available via action interface)
            if goal_handle is not None:
                feedback = ManageEpisode.Feedback()
                feedback.elapsed_s = elapsed
                feedback.current_reward = reward
                feedback.control_source = source
                feedback.status = "running"
                feedback.messages_written = 0  # Updated from recorder feedback if available
                goal_handle.publish_feedback(feedback)

            time.sleep(feedback_interval)

    # ====================================================================
    # Child action helpers
    # ====================================================================

    def _start_recorder(self, prompt: str):
        """Send RecordEpisode goal and return the goal handle, or None on failure."""
        if not self._recorder_client.wait_for_server(timeout_sec=ACTION_CLIENT_TIMEOUT_SEC):
            self.get_logger().error("Episode recorder action server not available")
            return None

        goal = RecordEpisode.Goal()
        goal.prompt = prompt

        future = self._recorder_client.send_goal_async(goal)
        if not _wait_for_future(future, GOAL_RESPONSE_TIMEOUT_SEC):
            self.get_logger().error("Recorder goal send timed out")
            return None

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Recorder goal rejected")
            return None

        self.get_logger().info("Episode recorder started")
        return goal_handle

    def _start_policy(self, prompt: str):
        """Send RunPolicy goal and return the goal handle, or None on failure."""
        if not self._policy_client.wait_for_server(timeout_sec=ACTION_CLIENT_TIMEOUT_SEC):
            self.get_logger().error("Robot policy action server not available")
            return None

        goal = RunPolicy.Goal()
        goal.prompt = prompt

        future = self._policy_client.send_goal_async(goal)
        if not _wait_for_future(future, GOAL_RESPONSE_TIMEOUT_SEC):
            self.get_logger().error("Policy goal send timed out")
            return None

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Policy goal rejected")
            return None

        self.get_logger().info("Robot policy started")
        return goal_handle

    def _start_reward_classifier(self, prompt: str):
        """Send RunPolicy goal to reward classifier and return goal handle, or None."""
        if not self._reward_client.wait_for_server(timeout_sec=ACTION_CLIENT_TIMEOUT_SEC):
            self.get_logger().error("Reward classifier action server not available")
            return None

        goal = RunPolicy.Goal()
        goal.prompt = prompt

        future = self._reward_client.send_goal_async(goal)
        if not _wait_for_future(future, GOAL_RESPONSE_TIMEOUT_SEC):
            self.get_logger().error("Reward classifier goal send timed out")
            return None

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Reward classifier goal rejected")
            return None

        self.get_logger().info("Reward classifier started")
        return goal_handle

    def _cancel_recorder(self) -> None:
        """Cancel the active recorder goal without waiting for its result."""
        if self._recorder_goal_handle is None:
            return
        try:
            cancel_future = self._recorder_goal_handle.cancel_goal_async()
            _wait_for_future(cancel_future, CANCEL_TIMEOUT_SEC)
            self.get_logger().info("Episode recorder cancelled")
        except Exception as e:
            self.get_logger().warning(f"Failed to cancel recorder: {e}")

    def _cancel_policy(self) -> None:
        """Cancel the active robot policy goal and wait for it to finish."""
        if self._policy_goal_handle is None:
            return
        try:
            cancel_future = self._policy_goal_handle.cancel_goal_async()
            if not _wait_for_future(cancel_future, CANCEL_TIMEOUT_SEC):
                self.get_logger().warning("Policy cancel timed out")
                return
            # Wait for the goal to actually finish executing so _active_goal
            # is cleared before we send a new goal on the next episode.
            result_future = self._policy_goal_handle.get_result_async()
            if not _wait_for_future(result_future, RESULT_TIMEOUT_SEC):
                self.get_logger().warning("Policy result timed out after cancel")
            self.get_logger().info("Robot policy cancelled")
        except Exception as e:
            self.get_logger().warning(f"Failed to cancel policy: {e}")

    def _cancel_reward_classifier(self) -> None:
        """Cancel the active reward classifier goal and wait for it to finish."""
        if self._reward_goal_handle is None:
            return
        try:
            cancel_future = self._reward_goal_handle.cancel_goal_async()
            if not _wait_for_future(cancel_future, CANCEL_TIMEOUT_SEC):
                self.get_logger().warning("Reward classifier cancel timed out")
                return
            result_future = self._reward_goal_handle.get_result_async()
            if not _wait_for_future(result_future, RESULT_TIMEOUT_SEC):
                self.get_logger().warning("Reward classifier result timed out after cancel")
            self.get_logger().info("Reward classifier cancelled")
        except Exception as e:
            self.get_logger().warning(f"Failed to cancel reward classifier: {e}")

    def _stop_recorder(self):
        """Cancel the recorder and wait for its result (which includes bag_path)."""
        if self._recorder_goal_handle is None:
            return None
        try:
            # Cancel recording - this triggers bag finalization
            cancel_future = self._recorder_goal_handle.cancel_goal_async()
            _wait_for_future(cancel_future, CANCEL_TIMEOUT_SEC)

            # Get result (bag_path, messages_written)
            result_future = self._recorder_goal_handle.get_result_async()

            if _wait_for_future(result_future, RESULT_TIMEOUT_SEC):
                result = result_future.result().result
                self.get_logger().info(
                    f"Recorder stopped: {result.messages_written} messages -> {result.bag_path}"
                )
                return result
            else:
                self.get_logger().warning("Recorder result timed out")
                return None
        except Exception as e:
            self.get_logger().warning(f"Failed to stop recorder: {e}")
            return None

    def _cancel_all_children(self) -> None:
        """Best-effort cancel of all child action goals."""
        self._cancel_policy()
        self._cancel_reward_classifier()
        try:
            if self._recorder_goal_handle is not None:
                cancel_future = self._recorder_goal_handle.cancel_goal_async()
                _wait_for_future(cancel_future, CANCEL_TIMEOUT_SEC)
        except Exception as e:
            self.get_logger().warning(f"Failed to cancel recorder: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = RosettaHilManagerNode()

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
