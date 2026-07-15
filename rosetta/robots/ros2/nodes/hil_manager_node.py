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

r"""
HilManagerNode: Human-in-the-Loop episode orchestrator.

Coordinates robot policy inference, bag recording, reward classification, and
teleop muxing. Exposes a ManageEpisode action for unified episode control.

The node acts as an orchestrator:
- Action client to policy_runner_node(s) for policy inference (RunPolicy)
- Action client to episode_recorder_node for bag recording (RecordEpisode)
- Muxes between policy output and teleop input for seamless human takeover
- Muxes between reward classifier output and human reward overrides
- Monitors episode termination conditions (timeout, human stop, reward threshold)

Usage:
    ros2 launch rosetta hil_launch.py

    ros2 action send_goal /manage_episode \\
        rosetta_interfaces/action/ManageEpisode \\
        "{prompt: 'pick up cube', max_duration_s: 90.0}" --feedback
"""

from __future__ import annotations

import sys
import threading
import time

from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rosetta_interfaces.action import ManageEpisode, RecordEpisode, RunPolicy
from rosetta_interfaces.srv import StartHILEpisode
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import Int8
from std_srvs.srv import SetBool, Trigger

from rosetta.contract.schema import load_contract
from rosetta.contract.specs import (
    iter_action_specs,
    iter_observation_specs,
    iter_teleop_feedback_specs,
    iter_teleop_input_specs,
)
from rosetta.frames.codecs import decode_value, encode_value
from rosetta.robots.ros2.nodes.node_utils import (
    finish_goal,
    positive_rate_descriptor,
    spin_lifecycle_node,
    wait_until,
)
from rosetta.robots.ros2.ros2_utils import qos_profile_from_dict
from rosetta.robots.ros2.rosetta_lifecycle_node import RosettaLifecycleNode

# ---------- Helpers ----------


def _resolve_selector(obj, path: str):
    """
    Resolve a dotted selector, supporting numeric array indices.

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
    return wait_until(future.done, timeout_sec, poll=FUTURE_POLL_SEC)


class HilManagerNode(RosettaLifecycleNode):
    """
    ROS2 Lifecycle node that orchestrates HIL episodes.

    Coordinates robot policy, reward classifier, episode recorder, and teleop
    muxing through a single ManageEpisode action interface.
    """

    # An episode winds down three child goals (policy, reward, recorder),
    # each with its own cancel/result timeouts — give it longer than the
    # default before deactivate/shutdown proceeds without it.
    STOP_WORK_TIMEOUT_SEC = 10.0

    def __init__(self, **kwargs):
        super().__init__("hil_manager", **kwargs)

        # -------------------- Parameters --------------------
        self.declare_parameter(
            "contract_path",
            "",
            ParameterDescriptor(description="Path to HIL contract YAML file", read_only=True),
        )
        self.declare_parameter(
            "enable_reward_classifier",
            False,
            ParameterDescriptor(
                description="Enable reward classifier policy (requires separate contract + model)",
                read_only=True,
            ),
        )
        self.declare_parameter(
            "enable_recording",
            True,
            ParameterDescriptor(
                description="Start the episode recorder for each episode. Everything else "
                "(policy, mux, teleop events, reward labeling, intervention) runs "
                "identically when false -- nothing is written to a bag.",
                read_only=True,
            ),
        )
        self.declare_parameter(
            "manage_policy_lifecycle",
            True,
            ParameterDescriptor(
                description="Send a RunPolicy goal to policy_runner_node at episode start and "
                "cancel it at episode end. When false, hil_manager never sends or cancels a "
                "policy goal at all -- it assumes a policy is already running, started some "
                "other way, and just mux/records/labels around it.",
                read_only=True,
            ),
        )
        self.declare_parameter(
            "default_episode_prompt",
            "",
            ParameterDescriptor(
                description="Task prompt used when an episode is started via the "
                "start_episode teleop event -- a button press carries no text input."
            ),
        )
        self.declare_parameter(
            "policy_action_name",
            "/robot_policy/run_policy",
            ParameterDescriptor(description="Action name for robot policy client", read_only=True),
        )
        self.declare_parameter(
            "reward_classifier_action_name",
            "/reward_classifier/run_policy",
            ParameterDescriptor(description="Action name for reward classifier client", read_only=True),
        )
        self.declare_parameter(
            "recorder_action_name",
            "/record_episode",
            ParameterDescriptor(description="Action name for episode recorder client", read_only=True),
        )
        self.declare_parameter(
            "policy_remap_prefix",
            "/hil/policy",
            ParameterDescriptor(description="Topic prefix for remapped policy output", read_only=True),
        )
        self.declare_parameter(
            "reward_remap_prefix",
            "/hil/reward",
            ParameterDescriptor(description="Topic prefix for remapped reward classifier output", read_only=True),
        )
        self.declare_parameter(
            "human_reward_positive",
            1.0,
            ParameterDescriptor(description="Reward value for human positive override"),
        )
        self.declare_parameter(
            "human_reward_negative",
            -1.0,
            ParameterDescriptor(
                description="Reward value for human negative override -- a distinct "
                "sentinel, not 0.0, so an explicit failure is distinguishable in the "
                "recorded reward stream from an episode nobody labeled."
            ),
        )
        self.declare_parameter(
            "feedback_rate_hz",
            30.0,
            positive_rate_descriptor("Rate for publishing ManageEpisode feedback"),
        )
        self.declare_parameter(
            "default_max_duration",
            0.0,
            ParameterDescriptor(
                description="Default max episode duration in seconds, used when the "
                "request does not specify max_duration_s "
                "(0 or negative = run until stopped, no limit)"
            ),
        )

        # -------------------- State --------------------
        # One-episode-at-a-time comes from the base work gate (busy guard +
        # per-claim stop event), claimed in _on_goal / _handle_start_episode /
        # _start_episode_from_button.
        self._contract = None

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
        # Last observed pressed state per teleop event, for edge detection
        # (guarded by _mux_lock). Deliberately NOT reset per episode: a button
        # held across an episode boundary is not a new press.
        self._prev_event_pressed: dict[str, bool] = {}

        # Subscriptions and publishers for muxing
        self._mux_subs: list = []
        self._command_publishers: dict[str, tuple] = {}  # original_topic -> (msg_cls, publisher)
        self._reward_publishers: dict[str, tuple] = {}  # original_topic -> (msg_cls, publisher)
        self._teleop_feedback_publishers: dict[str, tuple] = {}  # origin topic -> (msg_cls, publisher)
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

    def _setup(self) -> None:
        """Load contract, create action clients, subscriptions, publishers, and action server.

        Every failure raises: the base logs the traceback and routes through
        on_error, which tears down whatever was partially built here.
        """
        contract_path = self.get_parameter("contract_path").value
        if not contract_path:
            raise ValueError("contract_path parameter required")

        self._contract = load_contract(contract_path)

        enable_reward = self.get_parameter("enable_reward_classifier").value
        policy_remap_prefix = self.get_parameter("policy_remap_prefix").value
        reward_remap_prefix = self.get_parameter("reward_remap_prefix").value

        # --- Action clients ---
        policy_action = self.get_parameter("policy_action_name").value
        self._policy_client = ActionClient(self, RunPolicy, policy_action, callback_group=self._action_cbg)

        recorder_action = self.get_parameter("recorder_action_name").value
        self._recorder_client = ActionClient(self, RecordEpisode, recorder_action, callback_group=self._action_cbg)

        if enable_reward:
            reward_action = self.get_parameter("reward_classifier_action_name").value
            self._reward_client = ActionClient(self, RunPolicy, reward_action, callback_group=self._action_cbg)

        # --- Mux: subscribe to remapped policy output, publish to real command topic ---
        for entry in self._contract.actions:
            for src in entry.sources:
                original_topic = src.channel.topic
                remapped_topic = policy_remap_prefix + original_topic
                msg_cls = get_message(src.channel.type)
                qos = qos_profile_from_dict(src.channel.qos)

                # Publisher to the real command topic
                pub = self.create_publisher(msg_cls, original_topic, qos)
                self._command_publishers[original_topic] = (msg_cls, pub)

                # Subscribe to the remapped policy output
                sub = self.create_subscription(
                    msg_cls,
                    remapped_topic,
                    lambda msg, topic=original_topic: self._on_policy_output(msg, topic),
                    qos,
                    callback_group=self._sub_cbg,
                )
                self._mux_subs.append(sub)
                self.get_logger().info(f"Action mux: {remapped_topic} -> {original_topic}")

        # --- Mux: subscribe to teleop input, decode+encode+publish onto each
        # entry's target action topic. Teleop input's select/apply may differ
        # from the action's (a different leader topic, field layout, or
        # units), so a raw republish would be wrong whenever they do; going
        # through the same decode(teleop) -> encode(action) machinery as
        # everything else in the contract is correct regardless. ---
        if self._contract.teleop and self._contract.teleop.input:
            action_specs_by_topic = {s.source.channel.topic: s for s in iter_action_specs(self._contract)}
            teleop_input_pairs = zip(self._contract.teleop.input, iter_teleop_input_specs(self._contract), strict=True)
            for tis, teleop_spec in teleop_input_pairs:
                action_spec = action_specs_by_topic[tis.target]  # schema.py validated target at load time
                msg_cls = get_message(teleop_spec.source.channel.type)
                qos = qos_profile_from_dict(teleop_spec.source.channel.qos)
                sub = self.create_subscription(
                    msg_cls,
                    teleop_spec.source.channel.topic,
                    lambda msg, t_spec=teleop_spec, a_spec=action_spec, target=tis.target: self._on_teleop_input(
                        msg, t_spec, a_spec, target
                    ),
                    qos,
                    callback_group=self._sub_cbg,
                )
                self._mux_subs.append(sub)
                self.get_logger().info(f"Teleop input: {teleop_spec.source.channel.topic} -> {tis.target}")

        # --- Teleop feedback: subscribe to each entry's origin observation
        # topic, decode+encode+publish to the human device. Runs regardless
        # of mux state (policy or teleop) -- it's informational for whoever
        # is holding the device, not a control input. ---
        if self._contract.teleop and self._contract.teleop.feedback:
            observation_specs_by_topic = {s.source.channel.topic: s for s in iter_observation_specs(self._contract)}
            for tfs, feedback_spec in zip(
                self._contract.teleop.feedback, iter_teleop_feedback_specs(self._contract), strict=True
            ):
                obs_spec = observation_specs_by_topic[tfs.origin]  # schema.py validated origin at load time
                fb_msg_cls = get_message(feedback_spec.source.channel.type)
                fb_qos = qos_profile_from_dict(feedback_spec.source.channel.qos)
                fb_pub = self.create_publisher(fb_msg_cls, feedback_spec.source.channel.topic, fb_qos)
                self._teleop_feedback_publishers[tfs.origin] = (fb_msg_cls, fb_pub)

                origin_msg_cls = get_message(obs_spec.source.channel.type)
                origin_qos = qos_profile_from_dict(obs_spec.source.channel.qos)
                sub = self.create_subscription(
                    origin_msg_cls,
                    tfs.origin,
                    lambda msg, o_spec=obs_spec, f_spec=feedback_spec, pub=fb_pub: self._on_teleop_feedback_origin(
                        msg, o_spec, f_spec, pub
                    ),
                    origin_qos,
                    callback_group=self._sub_cbg,
                )
                self._mux_subs.append(sub)
                self.get_logger().info(f"Teleop feedback: {tfs.origin} -> {feedback_spec.source.channel.topic}")

        # --- Reward publishers (always created for human reward buttons) ---
        reward_channels = [src.channel for entry in self._contract.rewards for src in entry.sources]
        for ch in reward_channels:
            pub = self.create_publisher(get_message(ch.type), ch.topic, qos_profile_from_dict(ch.qos))
            self._reward_publishers[ch.topic] = (get_message(ch.type), pub)
            self.get_logger().info(f"Reward publisher: {ch.topic}")

        # --- Mux: subscribe to remapped reward classifier output ---
        if enable_reward:
            for ch in reward_channels:
                remapped_topic = reward_remap_prefix + ch.topic
                sub = self.create_subscription(
                    get_message(ch.type),
                    remapped_topic,
                    lambda msg, topic=ch.topic: self._on_reward_classifier_output(msg, topic),
                    qos_profile_from_dict(ch.qos),
                    callback_group=self._sub_cbg,
                )
                self._mux_subs.append(sub)
                self.get_logger().info(f"Reward mux: {remapped_topic} -> {ch.topic}")

        # --- Subscribe to teleop events ---
        if self._contract.teleop and self._contract.teleop.events:
            events_spec = self._contract.teleop.events
            msg_cls = get_message(events_spec.channel.type)
            qos = qos_profile_from_dict(events_spec.channel.qos)
            sub = self.create_subscription(
                msg_cls,
                events_spec.channel.topic,
                lambda msg: self._on_teleop_events(msg, events_spec),
                qos,
                callback_group=self._sub_cbg,
            )
            self._mux_subs.append(sub)
            self.get_logger().info(f"Teleop events: {events_spec.channel.topic}")

        # --- HIL intervention publisher ---
        # A physical leader arm that wants to torque-track the command topic
        # (so101_ros2_node's hil_teacher mode) subscribes to this directly
        # instead of hil_manager calling out to an arm-specific service --
        # keeps this node hardware-agnostic.
        self._intervention_pub = self.create_publisher(Int8, "hil_intervention", 10)

        # --- Service wrapper (for callers that don't support actions) ---
        self._episode_service = self.create_service(
            StartHILEpisode,
            "start_episode",
            self._handle_start_episode,
            callback_group=self._action_cbg,
        )
        self._stop_service = self.create_service(
            Trigger,
            "stop_episode",
            self._handle_stop_episode,
            callback_group=self._action_cbg,
        )
        self._intervention_service = self.create_service(
            SetBool,
            "set_intervention",
            self._handle_set_intervention,
            callback_group=self._action_cbg,
        )
        self._reward_override_service = self.create_service(
            SetBool,
            "set_reward_override",
            self._handle_set_reward_override,
            callback_group=self._action_cbg,
        )
        self._clear_reward_service = self.create_service(
            Trigger,
            "clear_reward_override",
            self._handle_clear_reward_override,
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

    def _teardown(self) -> None:
        """Destroy everything _setup() created. Tolerates partial state."""
        for sub in self._mux_subs:
            self.destroy_subscription(sub)
        self._mux_subs = []

        for _, pub in self._command_publishers.values():
            self.destroy_publisher(pub)
        self._command_publishers = {}

        for _, pub in self._reward_publishers.values():
            self.destroy_publisher(pub)
        self._reward_publishers = {}

        for _, pub in self._teleop_feedback_publishers.values():
            self.destroy_publisher(pub)
        self._teleop_feedback_publishers = {}

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
            self._action_server.destroy()
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

    def _on_teleop_input(self, msg, teleop_spec, action_spec, target_topic: str) -> None:
        """Decode a teleop input message and republish it as the executed action.

        decode(teleop_spec) -> encode(action_spec) mirrors the build/serve
        pipeline used everywhere else in the contract: teleop input's
        select/apply may differ from the target action's (a different leader
        topic, field layout, or units), so a raw republish would be wrong
        whenever they do. Going through the same decode/encode machinery as
        every other spec is correct regardless of whether they happen to match.
        """
        with self._mux_lock:
            if self._control_source != "teleop":
                return

        try:
            decoded = decode_value(msg, teleop_spec)
            ros_msg = encode_value(decoded, action_spec)
        except Exception as e:
            self.get_logger().error(
                f"Teleop input decode/encode failed for target '{target_topic}': {e}", throttle_duration_sec=1.0
            )
            return

        if target_topic in self._command_publishers:
            _, pub = self._command_publishers[target_topic]
            pub.publish(ros_msg)

    def _on_teleop_feedback_origin(self, msg, obs_spec, feedback_spec, pub) -> None:
        """Decode an observation message and forward it to the human device as feedback."""
        try:
            decoded = decode_value(msg, obs_spec)
            ros_msg = encode_value(decoded, feedback_spec)
        except Exception as e:
            self.get_logger().error(f"Teleop feedback encode failed: {e}", throttle_duration_sec=1.0)
            return
        pub.publish(ros_msg)

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
        """
        Handle teleop event buttons (intervention, stop, reward override).

        Six independent signals, matched to what the contract's `select` maps
        them to (they need not be on distinct physical buttons, but usually
        should be so each can fire independently):
          is_intervention    - hold for teleop control (edge-triggered mux)
          start_episode      - start a new episode, using default_episode_prompt
                                as the task description (a button carries no
                                text input). No-op if one is already running.
          success / failure  - label the reward override, episode keeps
                                running (latched: stays in effect until the
                                other button, the clear_reward_override
                                service, or the next episode's state reset --
                                one press marks every subsequent frame until
                                you say otherwise, not just while held)
          end_success        - end the episode now, reward = success value
          end_failure        - end the episode now, reward = failure value
                                (a distinct sentinel from the neutral 0.0 an
                                unlabeled end leaves behind -- see
                                human_reward_negative -- so a real failure is
                                filterable from "nobody labeled this one")

        Every manual stop goes through end_success/end_failure -- there is no
        "just end, no label" option. An episode ended for an unrelated reason
        (robot fault, cutting a test short) still gets a reward value; ignore
        it at train time rather than trying to represent "no claim made".

        Nothing here ever touches the recorded bag file -- episodes are never
        deleted by a button. A bad take is filtered later from the recorded
        reward stream, not discarded at record time.

        All handling is EDGE-triggered on press/release transitions. Every
        Joy message reports every button's level, so level-triggered handling
        made unpressed buttons act continuously: an unpressed `failure` button
        cleared the override that `success` had just set (the override never
        stuck), and an unpressed `is_intervention` forced the mux back to
        policy on every message, fighting the set_intervention service.

        A success/failure RELEASE is a no-op: the override latches until the
        other button, the clear_reward_override service, or the next
        episode's state reset.
        """
        self.get_logger().debug(f"Joy received: buttons={list(msg.buttons)}")
        for event_name, selector in events_spec.select.items():
            try:
                pressed = bool(_resolve_selector(msg, selector))
            except (AttributeError, IndexError, ValueError) as e:
                self.get_logger().warning(f"Selector '{selector}' failed: {e}")
                continue

            with self._mux_lock:
                prev = self._prev_event_pressed.get(event_name, False)
                self._prev_event_pressed[event_name] = pressed
            if pressed == prev:
                continue

            if pressed:
                self._on_event_press(event_name)
            else:
                self._on_event_release(event_name)

    def _on_event_press(self, event_name: str) -> None:
        """Act on a press edge of one teleop event button."""
        if event_name == "is_intervention":
            with self._mux_lock:
                self._control_source = "teleop"
            self.get_logger().info("Mux: policy -> teleop (human intervention)")

        elif event_name == "start_episode":
            self._start_episode_from_button()

        elif event_name in ("success", "failure"):
            param = "human_reward_positive" if event_name == "success" else "human_reward_negative"
            reward_val = self.get_parameter(param).value
            with self._mux_lock:
                self._current_reward = reward_val
                self._human_reward_override = True
            self._publish_human_reward(reward_val)
            self.get_logger().info(f"Human reward override: {reward_val}")

        elif event_name in ("end_success", "end_failure"):
            param = "human_reward_positive" if event_name == "end_success" else "human_reward_negative"
            reward_val = self.get_parameter(param).value
            with self._mux_lock:
                self._current_reward = reward_val
                self._human_reward_override = True
            self._signal_stop()
            self._publish_human_reward(reward_val)
            self.get_logger().info(f"Episode ending, reward={reward_val} ({event_name})")

    def _on_event_release(self, event_name: str) -> None:
        """Act on a release edge of one teleop event button."""
        if event_name == "is_intervention":
            with self._mux_lock:
                self._control_source = "policy"
            self.get_logger().info("Mux: teleop -> policy (intervention released)")
        # success/failure releases latch (see _on_teleop_events docstring);
        # end_success/end_failure are one-shot flags consumed by the
        # feedback loop; their release is a no-op.

    def _publish_human_reward(self, reward_val: float) -> None:
        """Publish a human-overridden reward value to all reward topics."""
        for _original_topic, (msg_cls, pub) in self._reward_publishers.items():
            msg = msg_cls()
            msg.data = reward_val
            pub.publish(msg)

    def _start_episode_from_button(self) -> None:
        """Start a new episode from the start_episode teleop event.

        Mirrors _handle_start_episode (the StartHILEpisode service path):
        same work-gate claim, same background thread via _run_episode_detached.
        There's no caller to report accept/reject back to, so a failed claim
        just logs (via the base) and no-ops -- a button press is
        fire-and-forget.
        """
        if not self._try_claim_work():
            return

        prompt = self.get_parameter("default_episode_prompt").value
        threading.Thread(
            target=self._run_episode_detached,
            args=(prompt, 0.0, 0.0),
            daemon=True,
        ).start()
        self.get_logger().info(f"Episode started via button: prompt='{prompt}'")

    # ====================================================================
    # Action server callbacks
    # ====================================================================

    def _on_goal(self, _goal_request) -> GoalResponse:
        """Accept or reject a ManageEpisode goal."""
        return GoalResponse.ACCEPT if self._try_claim_work() else GoalResponse.REJECT

    def _on_cancel(self, _goal_handle) -> CancelResponse:
        """Accept cancel request for ManageEpisode."""
        self.get_logger().info("Received cancel request")
        self._signal_stop()
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle) -> ManageEpisode.Result:
        """Execute a full HIL episode via the action interface."""
        prompt = goal_handle.request.prompt or ""
        max_duration = goal_handle.request.max_duration_s
        reward_threshold = goal_handle.request.success_reward_threshold

        try:
            fields, _cancelled = self._run_episode(prompt, max_duration, reward_threshold, goal_handle=goal_handle)

            result = ManageEpisode.Result()
            result.success = fields["success"]
            result.message = fields["message"]
            result.termination_reason = fields["termination_reason"]
            result.final_reward = fields["final_reward"]
            result.bag_path = fields["bag_path"]
            result.messages_written = fields["messages_written"]

            # finish_goal re-checks is_cancel_requested at terminal time, so a
            # cancel that raced in after the feedback loop exited still ends
            # CANCELED rather than SUCCEEDED (succeed() from CANCELING is
            # rcl-legal but would report the wrong terminal state).
            finish_goal(goal_handle, result)
            return result
        finally:
            # Release on EVERY exit path — a leaked guard would reject all
            # subsequent episodes until node restart.
            self._busy.release()

    def _handle_start_episode(
        self, request: StartHILEpisode.Request, response: StartHILEpisode.Response
    ) -> StartHILEpisode.Response:
        """Service: start a HIL episode and return immediately.

        The episode runs on a background thread — the old synchronous form
        held an executor thread for the whole episode (unbounded with
        max_duration_s=0), one concurrent call away from starving the
        executor. Monitor via ManageEpisode feedback, stop via stop_episode;
        results land in the node log.
        """
        # Claim the slot before returning so a concurrent start (service or
        # action goal) is rejected; the episode thread releases it.
        if not self._try_claim_work():
            response.accepted = False
            response.message = "Episode already in progress" if self._accepting_work else "Node not active"
            return response

        threading.Thread(
            target=self._run_episode_detached,
            args=(
                request.prompt or "",
                request.max_duration_s,
                request.success_reward_threshold,
            ),
            daemon=True,
        ).start()

        response.accepted = True
        response.message = "Episode started"
        return response

    def _run_episode_detached(self, prompt: str, max_duration: float, reward_threshold: float) -> None:
        """Thread target for the service path: run, log the result, release."""
        try:
            fields, _ = self._run_episode(prompt, max_duration, reward_threshold)
            self.get_logger().info(
                f"Episode finished ({fields['termination_reason'] or 'error'}): "
                f"{fields['message']} bag={fields['bag_path']!r} "
                f"reward={fields['final_reward']} msgs={fields['messages_written']}"
            )
        finally:
            # Mirror of the action path's finally — a leaked guard would
            # reject all subsequent episodes until node restart.
            self._busy.release()

    def _handle_stop_episode(self, _request, response: Trigger.Response) -> Trigger.Response:
        """Service: signal the active episode to exit its feedback loop."""
        self._signal_stop()
        response.success = True
        response.message = "Stop requested"
        self.get_logger().info("Stop requested via service")
        return response

    def _handle_set_intervention(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        """Service: switch mux between policy (False) and teleop (True)."""
        with self._mux_lock:
            self._control_source = "teleop" if request.data else "policy"
        response.success = True
        response.message = f"Control source: {'teleop' if request.data else 'policy'}"
        self.get_logger().info(response.message)
        return response

    def _handle_set_reward_override(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
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

    def _handle_clear_reward_override(self, _request, response: Trigger.Response) -> Trigger.Response:
        """Service: release the human reward override."""
        with self._mux_lock:
            self._human_reward_override = False
        response.success = True
        response.message = "Reward override cleared"
        self.get_logger().info(response.message)
        return response

    def _run_episode(
        self,
        prompt: str,
        max_duration: float,
        reward_threshold: float,
        goal_handle=None,
    ) -> tuple[dict, bool]:
        """
        Core HIL episode logic shared by the action and service interfaces.

        ``goal_handle`` is the ManageEpisode server goal when driven via the
        action interface (enables live feedback and cancel detection in the
        feedback loop); the service interface passes None.

        Returns
        -------
            (fields dict, cancelled bool) where fields contains result values.

        """
        stop_event = self._stop_event  # this episode's event, armed at claim time

        if max_duration <= 0.0:
            max_duration = self.get_parameter("default_max_duration").value

        self.get_logger().info(
            f"Starting episode: prompt={prompt!r}, max_duration={max_duration}s, reward_threshold={reward_threshold}"
        )

        with self._mux_lock:
            self._control_source = "policy"
            self._current_reward = 0.0
            self._human_reward_override = False

        enable_reward = self.get_parameter("enable_reward_classifier").value
        enable_recording = self.get_parameter("enable_recording").value
        manage_policy = self.get_parameter("manage_policy_lifecycle").value
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
            if enable_recording:
                recorder_gh = self._send_child_goal(
                    self._recorder_client, RecordEpisode.Goal(prompt=prompt), "Episode recorder"
                )
                if recorder_gh is None:
                    fields["message"] = "Failed to start episode recorder"
                    return fields, cancelled
                self._recorder_goal_handle = recorder_gh
            else:
                self.get_logger().info("Recording disabled (enable_recording=false) -- nothing will be saved")

            if manage_policy:
                policy_gh = self._send_child_goal(self._policy_client, RunPolicy.Goal(prompt=prompt), "Robot policy")
                if policy_gh is None:
                    fields["message"] = "Failed to start robot policy"
                    self._cancel_child(self._recorder_goal_handle, "Episode recorder")
                    return fields, cancelled
                self._policy_goal_handle = policy_gh
            else:
                self.get_logger().info(
                    "manage_policy_lifecycle=false -- assuming a policy is already running "
                    "externally; not sending or cancelling a policy goal"
                )

            reward_gh = None
            if enable_reward and self._reward_client is not None:
                reward_gh = self._send_child_goal(
                    self._reward_client, RunPolicy.Goal(prompt=prompt), "Reward classifier"
                )
                if reward_gh is None:
                    self.get_logger().warning("Failed to start reward classifier, continuing without it")
                self._reward_goal_handle = reward_gh

            termination_reason = self._feedback_loop(goal_handle, max_duration, reward_threshold, stop_event)
            cancelled = termination_reason == "cancelled"

            self.get_logger().info(f"Episode ending: {termination_reason}")
            # wait_result: the child's active-goal slot must be free before
            # the next episode sends a new goal. Skipped entirely when
            # manage_policy_lifecycle=false -- _policy_goal_handle is None in
            # that case anyway (never sent), but the explicit guard documents
            # that this is intentional, not an oversight.
            if manage_policy:
                self._cancel_child(self._policy_goal_handle, "Robot policy", wait_result=True)
            if reward_gh is not None:
                self._cancel_child(self._reward_goal_handle, "Reward classifier", wait_result=True)

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
        stop_event: threading.Event,
    ) -> str:
        """
        Run the episode feedback loop until a termination condition is met.

        ``stop_event`` is this episode's claim-time event — set by cancel,
        the stop_episode service, an end_success/end_failure press, or
        deactivate/shutdown.

        Returns:
            Termination reason string.

        """
        feedback_interval = 1.0 / self.get_parameter("feedback_rate_hz").value
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time

            # Check termination conditions
            stop = stop_event.is_set()
            with self._mux_lock:
                reward = self._current_reward
                source = self._control_source

            if goal_handle is not None and goal_handle.is_cancel_requested:
                return "cancelled"

            if stop:
                return "human_stop"

            if max_duration > 0.0 and elapsed >= max_duration:
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
                feedback.messages_written = 0  # recorder feedback is not wired through
                goal_handle.publish_feedback(feedback)

            time.sleep(feedback_interval)

    # ====================================================================
    # Child action helpers
    # ====================================================================

    def _send_child_goal(self, client, goal, name: str):
        """Send a goal to a child action server; return the handle or None.

        One implementation for all three children (recorder, policy, reward
        classifier) — they only differ in client, goal type, and log name.
        """
        if not client.wait_for_server(timeout_sec=ACTION_CLIENT_TIMEOUT_SEC):
            self.get_logger().error(f"{name} action server not available")
            return None

        future = client.send_goal_async(goal)
        if not _wait_for_future(future, GOAL_RESPONSE_TIMEOUT_SEC):
            self.get_logger().error(f"{name} goal send timed out")
            return None

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f"{name} goal rejected")
            return None

        self.get_logger().info(f"{name} started")
        return goal_handle

    def _cancel_child(self, goal_handle, name: str, wait_result: bool = False):
        """Best-effort cancel of a child goal; None-safe.

        ``wait_result=True`` additionally waits for the goal to finish
        executing — so the child's active-goal slot is free before the next
        episode sends a new goal — and returns its result (None on timeout).
        A cancel timeout is only a warning: the goal may still terminate, so
        the result wait proceeds regardless.
        """
        if goal_handle is None:
            return None
        try:
            cancel_future = goal_handle.cancel_goal_async()
            if not _wait_for_future(cancel_future, CANCEL_TIMEOUT_SEC):
                self.get_logger().warning(f"{name} cancel timed out")
            if not wait_result:
                self.get_logger().info(f"{name} cancelled")
                return None

            result_future = goal_handle.get_result_async()
            if not _wait_for_future(result_future, RESULT_TIMEOUT_SEC):
                self.get_logger().warning(f"{name} result timed out after cancel")
                return None
            self.get_logger().info(f"{name} cancelled")
            return result_future.result().result
        except Exception as e:
            self.get_logger().warning(f"Failed to cancel {name}: {e}")
            return None

    def _stop_recorder(self):
        """Cancel the recorder and wait for its result (which includes bag_path)."""
        result = self._cancel_child(self._recorder_goal_handle, "Episode recorder", wait_result=True)
        if result is not None:
            self.get_logger().info(f"Recorder stopped: {result.messages_written} messages -> {result.bag_path}")
        return result

    def _cancel_all_children(self) -> None:
        """Best-effort cancel of all child action goals."""
        self._cancel_child(self._policy_goal_handle, "Robot policy", wait_result=True)
        self._cancel_child(self._reward_goal_handle, "Reward classifier", wait_result=True)
        self._cancel_child(self._recorder_goal_handle, "Episode recorder")


def main(args=None):
    """Run the human-in-the-loop manager node."""
    return spin_lifecycle_node(HilManagerNode, args=args)


if __name__ == "__main__":
    sys.exit(main())
