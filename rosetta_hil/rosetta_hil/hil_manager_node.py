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
    ros2 launch rosetta_hil hil_launch.py

    ros2 action send_goal /manage_episode \\
        rosetta_hil_interfaces/action/ManageEpisode \\
        "{prompt: 'pick up cube', max_duration_s: 90.0}" --feedback
"""

from __future__ import annotations

import sys
import threading
import time

from action_msgs.srv import CancelGoal
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionClient, ActionServer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
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
from rosetta.robots.ros2.child_goals import cancel_goal, send_goal
from rosetta.robots.ros2.field_access import resolve_indexed
from rosetta.robots.ros2.node_utils import (
    RESULT_QOS,
    STATUS_QOS,
    TERMINATION_ERROR,
    TERMINATION_STOPPED,
    copy_fields,
    finish_goal,
    positive_rate_descriptor,
    spin_lifecycle_node,
)
from rosetta.robots.ros2.rclpy_utils import qos_profile_from_dict
from rosetta.robots.ros2.rosetta_lifecycle_node import RosettaLifecycleNode
from rosetta_hil.hil_episode import (
    OUTCOME_UNLABELED,
    Control,
    EdgeDetector,
    EpisodeState,
    resolve_outcome,
    termination_reason,
)
from rosetta_hil_interfaces.action import ManageEpisode
from rosetta_hil_interfaces.msg import ManageEpisodeResult, ManageEpisodeStatus
from rosetta_hil_interfaces.srv import StartHILEpisode
from rosetta_interfaces.action import RecordEpisode, RunPolicy


class HilManagerNode(RosettaLifecycleNode):
    """Lifecycle node that runs one HIL episode at a time behind ManageEpisode."""

    # Longer than the base default because episode teardown cancels up to
    # three child goals. 10 s covers children that answer promptly. A child
    # that uses its full cancel and result timeouts (about 45 s across all
    # three) is logged and not waited for: deactivate warns and runs the
    # safety action, the episode thread keeps winding the children down, and
    # cleanup is refused until _busy releases.
    STOP_WORK_TIMEOUT_SEC = 10.0

    def __init__(self, **kwargs):
        super().__init__("hil_manager", **kwargs)

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
            "default_prompt",
            "",
            ParameterDescriptor(
                description="Task prompt used when a goal, service call, or the start_episode "
                "teleop event leaves prompt empty -- a button press carries no text input. "
                "Read at each episode start."
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
            "default_max_duration_s",
            0.0,
            ParameterDescriptor(
                description="Default max episode duration in seconds, used when the "
                "request does not specify max_duration_s "
                "(0 or negative = run until stopped, no limit)"
            ),
        )

        # One episode at a time is enforced by the base work gate, claimed in
        # on_goal, _on_start_episode and _start_episode_from_button.
        self._contract = None

        self._policy_client: ActionClient | None = None
        self._reward_client: ActionClient | None = None
        self._recorder_client: ActionClient | None = None

        self._action_server: ActionServer | None = None

        self._episode = EpisodeState()
        self._edges = EdgeDetector()
        self._bad_selectors: set[str] = set()  # warned about once each

        self._mux_subs: list = []
        self._command_publishers: dict[str, tuple] = {}  # original_topic -> (msg_cls, publisher)
        self._reward_publishers: dict[str, tuple] = {}  # original_topic -> (msg_cls, publisher)
        self._teleop_feedback_publishers: dict[str, tuple] = {}  # origin topic -> (msg_cls, publisher)
        self._intervention_pub = None
        self._status_pub = None
        self._result_pub = None
        self._cancel_client = None
        self._episode_service = None
        self._end_service = None
        self._cancel_episode_service = None
        self._intervention_service = None
        self._reward_override_service = None
        self._clear_reward_service = None

        # Child goal handles for the running episode.
        self._policy_goal_handle = None
        self._reward_goal_handle = None
        self._recorder_goal_handle = None
        # Live recorder message count, fed by _on_recorder_feedback.
        self._recorder_messages_written = 0

        # Callback groups. Teleop events get their own mutually-exclusive
        # group: edge detection is order-sensitive (a press processed after
        # its release fires a spurious one-shot), and the reentrant groups
        # give no ordering guarantee across executor threads.
        self._action_cbg = ReentrantCallbackGroup()
        self._sub_cbg = ReentrantCallbackGroup()
        self._events_cbg = MutuallyExclusiveCallbackGroup()

        self.get_logger().info("Node created (unconfigured)")

    def setup(self) -> None:
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

        policy_action = self.get_parameter("policy_action_name").value
        self._policy_client = ActionClient(self, RunPolicy, policy_action, callback_group=self._action_cbg)

        recorder_action = self.get_parameter("recorder_action_name").value
        self._recorder_client = ActionClient(self, RecordEpisode, recorder_action, callback_group=self._action_cbg)

        if enable_reward:
            reward_action = self.get_parameter("reward_classifier_action_name").value
            self._reward_client = ActionClient(self, RunPolicy, reward_action, callback_group=self._action_cbg)

        # Policy mux: remapped policy output -> real command topic.
        for entry in self._contract.actions:
            for src in entry.sources:
                original_topic = src.channel.topic
                remapped_topic = policy_remap_prefix + original_topic
                msg_cls = get_message(src.channel.type)
                qos = qos_profile_from_dict(src.channel.qos)

                pub = self.create_publisher(msg_cls, original_topic, qos)
                self._command_publishers[original_topic] = (msg_cls, pub)

                sub = self.create_subscription(
                    msg_cls,
                    remapped_topic,
                    lambda msg, topic=original_topic: self._on_policy_output(msg, topic),
                    qos,
                    callback_group=self._sub_cbg,
                )
                self._mux_subs.append(sub)
                self.get_logger().info(f"Action mux: {remapped_topic} -> {original_topic}")

        # Teleop mux: teleop input -> its target action topic (see _on_teleop_input).
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

        # Teleop feedback: origin observation -> human device. Runs whatever
        # the mux state, since it informs whoever holds the device.
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

        # Reward publishers, created even without a classifier so human
        # reward overrides have somewhere to go.
        reward_channels = [src.channel for entry in self._contract.rewards for src in entry.sources]
        for ch in reward_channels:
            pub = self.create_publisher(get_message(ch.type), ch.topic, qos_profile_from_dict(ch.qos))
            self._reward_publishers[ch.topic] = (get_message(ch.type), pub)
            self.get_logger().info(f"Reward publisher: {ch.topic}")

        # Reward mux: remapped classifier output -> real reward topic.
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

        if self._contract.teleop and self._contract.teleop.events:
            events_spec = self._contract.teleop.events
            msg_cls = get_message(events_spec.channel.type)
            qos = qos_profile_from_dict(events_spec.channel.qos)
            sub = self.create_subscription(
                msg_cls,
                events_spec.channel.topic,
                lambda msg: self._on_teleop_events(msg, events_spec),
                qos,
                callback_group=self._events_cbg,
            )
            self._mux_subs.append(sub)
            self.get_logger().info(f"Teleop events: {events_spec.channel.topic}")

        # A leader arm that torque-tracks the command topic (so101_ros2_node's
        # hil_teacher mode) subscribes to this, so this node needs no
        # arm-specific service call.
        self._intervention_pub = self.create_publisher(Int8, "hil_intervention", 10)
        # Feedback and result for every episode, including service and button
        # starts, which have no action client to send them to.
        self._status_pub = self.create_publisher(ManageEpisodeStatus, "~/status", STATUS_QOS)
        self._result_pub = self.create_publisher(ManageEpisodeResult, "~/result", RESULT_QOS)

        # Services for callers that cannot use actions. Node-private (~/)
        # names, matching the episode recorder.
        self._episode_service = self.create_service(
            StartHILEpisode,
            "~/start_episode",
            self._on_start_episode,
            callback_group=self._action_cbg,
        )
        # ~/end_episode is the labelled end (the end_success/end_failure
        # buttons); ~/cancel_episode abandons the take (an action cancel).
        self._end_service = self.create_service(
            SetBool,
            "~/end_episode",
            self._on_end_episode,
            callback_group=self._action_cbg,
        )
        self._cancel_episode_service = self.create_service(
            Trigger,
            "~/cancel_episode",
            self._on_cancel_episode,
            callback_group=self._action_cbg,
        )
        self._intervention_service = self.create_service(
            SetBool,
            "~/set_intervention",
            self._on_set_intervention,
            callback_group=self._action_cbg,
        )
        self._reward_override_service = self.create_service(
            SetBool,
            "~/set_reward_override",
            self._on_set_reward_override,
            callback_group=self._action_cbg,
        )
        self._clear_reward_service = self.create_service(
            Trigger,
            "~/clear_reward_override",
            self._on_clear_reward_override,
            callback_group=self._action_cbg,
        )

        self._action_server = ActionServer(
            self,
            ManageEpisode,
            "manage_episode",
            execute_callback=self._execute,
            goal_callback=self.on_goal,
            cancel_callback=self.on_cancel,
            callback_group=self._action_cbg,
        )

        # Client to this server's own cancel service, so ~/cancel_episode can
        # perform a real cancel. Relative name, so it expands through the same
        # node-namespace rules as the action name above.
        self._cancel_client = self.create_client(
            CancelGoal,
            "manage_episode/_action/cancel_goal",
            callback_group=self._action_cbg,
        )

        self.get_logger().info(
            f"Configured: robot_type={self._contract.robot_type}, "
            f"reward_classifier={'enabled' if enable_reward else 'disabled'}, "
            f"actions={len(self._contract.actions)}, "
            f"teleop={'yes' if self._contract.teleop else 'no'}"
        )

    def teardown(self) -> None:
        """Destroy everything setup() created. Tolerates partial state."""
        self._bad_selectors.clear()  # a reconfigure may load a fixed contract
        for sub in self._mux_subs:
            self.destroy_subscription(sub)
        self._mux_subs = []

        for attr in ("_command_publishers", "_reward_publishers", "_teleop_feedback_publishers"):
            for _, pub in getattr(self, attr).values():
                self.destroy_publisher(pub)
            setattr(self, attr, {})

        for attr in ("_policy_client", "_reward_client", "_recorder_client", "_action_server"):
            entity = getattr(self, attr)
            if entity is not None:
                entity.destroy()
                setattr(self, attr, None)

        for attr in ("_intervention_pub", "_status_pub", "_result_pub"):
            publisher = getattr(self, attr)
            if publisher is not None:
                self.destroy_publisher(publisher)
                setattr(self, attr, None)

        if self._cancel_client is not None:
            self.destroy_client(self._cancel_client)
            self._cancel_client = None

        for attr in (
            "_episode_service",
            "_end_service",
            "_cancel_episode_service",
            "_intervention_service",
            "_reward_override_service",
            "_clear_reward_service",
        ):
            service = getattr(self, attr)
            if service is not None:
                self.destroy_service(service)
                setattr(self, attr, None)

        self._contract = None

    def _on_policy_output(self, msg, original_topic: str) -> None:
        """Forward policy output to command topic when in policy mode."""
        if self._episode.control != Control.POLICY:
            return

        # Publisher exists by construction: sub and publisher come from the
        # same contract entry.
        _, pub = self._command_publishers[original_topic]
        pub.publish(msg)

    def _on_teleop_input(self, msg, teleop_spec, action_spec, target_topic: str) -> None:
        """Republish a teleop input message as the target action while teleop has control.

        Decodes with the teleop spec and encodes with the action spec rather
        than republishing raw, because their select/apply can differ (topic,
        field layout, units).
        """
        if self._episode.control != Control.TELEOP:
            return

        try:
            decoded = decode_value(msg, teleop_spec)
            ros_msg = encode_value(decoded, action_spec)
        except Exception as e:
            self.get_logger().error(
                f"Teleop input decode/encode failed for target '{target_topic}': {e}", throttle_duration_sec=1.0
            )
            return

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
        if not self._episode.accept_classifier_reward(float(msg.data)):
            return

        _, pub = self._reward_publishers[original_topic]
        pub.publish(msg)

    def _on_teleop_events(self, msg, events_spec) -> None:
        """Dispatch press and release edges of the teleop event buttons.

        Events, as named in the contract's ``select``:
          is_intervention    teleop has control while held
          start_episode      start an episode with default_prompt; ignored if
                             one is running
          success / failure  label the episode without ending it; the label
                             holds until the other button,
                             ~/clear_reward_override, or the next episode
          end_success        end the episode, outcome success
          end_failure        end the episode, outcome failure

        The four label events are ignored when no episode is running.

        Edge-triggered because each message reports every button's level, and
        acting on levels would let unpressed buttons undo pressed ones.
        """
        levels = {}
        for event_name, selector in events_spec.select.items():
            try:
                levels[event_name] = bool(resolve_indexed(msg, selector))
            except (AttributeError, IndexError, ValueError) as e:
                # A bad selector is a contract error and fails on every message; say so once.
                if selector not in self._bad_selectors:
                    self._bad_selectors.add(selector)
                    self.get_logger().warning(f"Selector '{selector}' failed: {e} (not repeated)")
        for event_name, pressed in self._edges.update(levels):
            if pressed:
                self._on_event_press(event_name)
            else:
                self._on_event_release(event_name)

    def _on_event_press(self, event_name: str) -> None:
        """Act on a press edge of one teleop event button."""
        if event_name == "is_intervention":
            self._episode.set_control(Control.TELEOP)
            self.get_logger().info("Mux: policy -> teleop (human intervention)")

        elif event_name == "start_episode":
            self._start_episode_from_button()

        elif event_name in ("success", "failure", "end_success", "end_failure") and not self.busy:
            self.get_logger().info(f"{event_name} ignored: no active episode")

        elif event_name in ("success", "failure"):
            reward_val = self._label_episode(event_name == "success")
            self._publish_human_reward(reward_val)
            self.get_logger().info(f"Human reward override: {reward_val}")

        elif event_name in ("end_success", "end_failure"):
            reward_val = self._label_episode(event_name == "end_success")
            # A labelled end, not a cancel: the goal succeeds and the verdict
            # goes in `outcome`.
            self.signal_stop(TERMINATION_STOPPED)
            self._publish_human_reward(reward_val)
            self.get_logger().info(f"Episode ending, reward={reward_val} ({event_name})")

    def _on_event_release(self, event_name: str) -> None:
        """Act on a release edge of one teleop event button."""
        if event_name == "is_intervention":
            self._episode.set_control(Control.POLICY)
            self.get_logger().info("Mux: teleop -> policy (intervention released)")
        # Other events act on press only.

    def _label_episode(self, success: bool) -> float:
        """Latch the human's verdict on this episode and return its reward value.

        The reward override and the episode outcome are one act with two
        representations: the numeric value goes into the recorded reward stream
        for training, the label goes into the action result so a caller can ask
        "was that take good?" without knowing what
        ``human_reward_positive``/``negative`` are set to.

        Returns:
            The reward value that was latched, for publishing.

        """
        param = "human_reward_positive" if success else "human_reward_negative"
        reward_val = self.get_parameter(param).value
        self._episode.label(success, reward_val)
        return reward_val

    def _publish_human_reward(self, reward_val: float) -> None:
        """Publish a human-overridden reward value to all reward topics."""
        for _original_topic, (msg_cls, pub) in self._reward_publishers.items():
            msg = msg_cls()
            msg.data = reward_val
            pub.publish(msg)

    def _start_episode_from_button(self) -> None:
        """Start an episode from the start_episode button.

        Same claim and thread as the ~/start_episode service. With no caller
        to answer, a failed claim only logs.
        """
        reason = self.try_claim_work()
        if reason is not None:
            self.get_logger().warning(f"Episode button ignored: {reason}")
            return

        # Empty prompt: _run_episode falls back to default_prompt.
        threading.Thread(
            target=self._run_episode_detached,
            args=("", 0.0, 0.0),
            daemon=True,
        ).start()
        self.get_logger().info("Episode started via button")

    # on_goal and on_cancel come from the base.

    def _execute(self, goal_handle) -> ManageEpisode.Result:
        """Execute a full HIL episode via the action interface."""
        prompt = goal_handle.request.prompt or ""
        max_duration = goal_handle.request.max_duration_s
        reward_threshold = goal_handle.request.success_reward_threshold

        with self.goal_work(goal_handle) as stop_event:
            fields = self._run_episode(prompt, max_duration, reward_threshold, stop_event, goal_handle=goal_handle)

            result = ManageEpisode.Result()
            result.termination_reason = fields["termination_reason"]
            result.outcome = fields["outcome"]
            result.message = fields["message"]
            result.final_reward = fields["final_reward"]
            result.bag_path = fields["bag_path"]
            result.messages_written = fields["messages_written"]

            # finish_goal re-checks is_cancel_requested, so a cancel that
            # arrives after the feedback loop exits still ends CANCELED, not
            # SUCCEEDED (rcl allows succeed() from CANCELING).
            finish_goal(goal_handle, result)
            return result

    def _on_start_episode(
        self, request: StartHILEpisode.Request, response: StartHILEpisode.Response
    ) -> StartHILEpisode.Response:
        """Service: start an episode on a background thread and return immediately.

        End it with ~/end_episode or ~/cancel_episode. The result goes to the
        node log.
        """
        # The base claims the slot before returning, so a concurrent start
        # (service or action goal) is rejected; the episode thread releases it.
        return self.handle_start_service(
            response,
            self._run_episode_detached,
            (request.prompt or "", request.max_duration_s, request.success_reward_threshold),
            what="episode",
        )

    def _run_episode_detached(self, prompt: str, max_duration: float, reward_threshold: float) -> None:
        """Thread target for the service/button path: run under the work guard, log the result."""
        # No goal handle: stop comes from ~/end_episode, ~/cancel_episode, an
        # end_* button, or deactivate.
        with self.goal_work(None) as stop_event:
            fields = self._run_episode(prompt, max_duration, reward_threshold, stop_event)
            self.get_logger().info(
                f"Episode finished ({fields['termination_reason']}, outcome {fields['outcome']}): "
                f"{fields['message']} bag={fields['bag_path']!r} "
                f"reward={fields['final_reward']} msgs={fields['messages_written']}"
            )

    def _on_end_episode(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        """Service: end the active episode with a verdict. True = success, False = failure.

        The end_success/end_failure buttons as a service. The goal succeeds
        with ``termination_reason="stopped"`` and the verdict in ``outcome``.
        """
        if not self.busy:
            response.success = False
            response.message = "No active episode"
            return response

        reward_val = self._label_episode(request.data)
        self.signal_stop(TERMINATION_STOPPED)
        self._publish_human_reward(reward_val)

        response.success = True
        response.message = f"Episode ending: {'success' if request.data else 'failure'} (reward {reward_val})"
        self.get_logger().info(response.message)
        return response

    def _on_cancel_episode(self, _request, response: Trigger.Response) -> Trigger.Response:
        """Service: abandon the active episode.

        Ends an action goal CANCELED, as a `ros2 action` cancel would (see
        :meth:`RosettaLifecycleNode.cancel_current_work`). ``outcome`` keeps
        whatever label was latched.
        """
        return self.handle_cancel_service(response, self._cancel_client, what="episode")

    def _on_set_intervention(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        """Service: switch mux between policy (False) and teleop (True).

        Works between episodes too, so a person can drive the robot to a start
        pose; every episode starts with the policy in control.
        """
        self._episode.set_control(Control.TELEOP if request.data else Control.POLICY)
        response.success = True
        response.message = f"Control source: {'teleop' if request.data else 'policy'}"
        self.get_logger().info(response.message)
        return response

    def _on_set_reward_override(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        """Service: apply a human reward override. True = positive, False = negative.

        Labels the episode without ending it, like the success/failure buttons.
        Refused when no episode is running.
        """
        if not self.busy:
            response.success = False
            response.message = "No active episode"
            return response
        reward_val = self._label_episode(request.data)
        self._publish_human_reward(reward_val)
        response.success = True
        response.message = f"Reward override set to {reward_val}"
        self.get_logger().info(response.message)
        return response

    def _on_clear_reward_override(self, _request, response: Trigger.Response) -> Trigger.Response:
        """Service: release the human reward override, retracting the episode's label.

        Refused when no episode is running.
        """
        if not self.busy:
            response.success = False
            response.message = "No active episode"
            return response
        self._episode.clear_label()
        response.success = True
        response.message = "Reward override cleared"
        self.get_logger().info(response.message)
        return response

    def _run_episode(
        self,
        prompt: str,
        max_duration: float,
        reward_threshold: float,
        stop_event: threading.Event,
        goal_handle=None,
    ) -> dict:
        """
        Run one episode; shared by the action, service, and button paths.

        ``stop_event`` comes from the ``goal_work`` block that owns the claim.
        ``goal_handle`` is the ManageEpisode goal on the action path (it
        enables feedback) and None otherwise.

        Returns:
            The result field values, including ``termination_reason`` (how the
            episode ended) and ``outcome`` (whether the robot did the task).

        """
        # Resolved here so every entry point falls back the same way.
        prompt = prompt or self.get_parameter("default_prompt").value
        if max_duration <= 0.0:
            max_duration = self.get_parameter("default_max_duration_s").value

        self.get_logger().info(
            f"Starting episode: prompt={prompt!r}, max_duration={max_duration}s, reward_threshold={reward_threshold}"
        )

        self._episode.reset()

        enable_reward = self.get_parameter("enable_reward_classifier").value
        enable_recording = self.get_parameter("enable_recording").value
        manage_policy = self.get_parameter("manage_policy_lifecycle").value
        fields = {
            # Every early return below is a failure to start, so default to error.
            "termination_reason": TERMINATION_ERROR,
            "outcome": OUTCOME_UNLABELED,
            "message": "",
            "final_reward": 0.0,
            "bag_path": "",
            "messages_written": 0,
        }

        self._recorder_messages_written = 0
        try:
            if enable_recording:
                recorder_gh = self._send_child_goal(
                    self._recorder_client,
                    RecordEpisode.Goal(prompt=prompt),
                    "Episode recorder",
                    feedback_callback=self._on_recorder_feedback,
                )
                if recorder_gh is None:
                    fields["message"] = "Failed to start episode recorder"
                    return fields
                self._recorder_goal_handle = recorder_gh
            else:
                self.get_logger().info("Recording disabled (enable_recording=false) -- nothing will be saved")

            if manage_policy:
                policy_gh = self._send_child_goal(self._policy_client, RunPolicy.Goal(prompt=prompt), "Robot policy")
                if policy_gh is None:
                    fields["message"] = "Failed to start robot policy"
                    # wait_result: a retry usually follows, and the recorder's
                    # goal slot must be free (and its bag finalized) first.
                    self._cancel_child(self._recorder_goal_handle, "Episode recorder", wait_result=True)
                    return fields
                self._policy_goal_handle = policy_gh
            else:
                self.get_logger().info(
                    "manage_policy_lifecycle=false -- assuming a policy is already running "
                    "externally; not sending or cancelling a policy goal"
                )

            reward_gh = None
            # The client exists iff enable_reward (read-only param, set in setup).
            if enable_reward:
                reward_gh = self._send_child_goal(
                    self._reward_client, RunPolicy.Goal(prompt=prompt), "Reward classifier"
                )
                if reward_gh is None:
                    self.get_logger().warning("Failed to start reward classifier, continuing without it")
                self._reward_goal_handle = reward_gh

            reason = self._feedback_loop(goal_handle, max_duration, reward_threshold, stop_event)

            self.get_logger().info(f"Episode ending: {reason}")
            # wait_result: each child's goal slot must be free before the next
            # episode sends a new goal.
            if manage_policy:
                self._cancel_child(self._policy_goal_handle, "Robot policy", wait_result=True)
            if reward_gh is not None:
                self._cancel_child(self._reward_goal_handle, "Reward classifier", wait_result=True)

            recorder_result = self._stop_recorder()

            _, final_reward, outcome = self._episode.snapshot()
            outcome = resolve_outcome(outcome, reason)

            fields["termination_reason"] = reason
            fields["outcome"] = outcome
            fields["final_reward"] = final_reward
            # message stays empty: termination_reason and outcome already say
            # what happened.
            if recorder_result is not None:
                fields["bag_path"] = recorder_result.bag_path
                fields["messages_written"] = recorder_result.messages_written

        except Exception as e:
            self.get_logger().error(f"Episode error: {e}")
            fields["termination_reason"] = TERMINATION_ERROR
            fields["message"] = str(e)
            self._cancel_all_children()

        finally:
            self._policy_goal_handle = None
            self._reward_goal_handle = None
            self._recorder_goal_handle = None
            # Every exit, including the early failure returns above.
            self._result_pub.publish(ManageEpisodeResult(**fields))

        self.get_logger().info(f"Episode finished: {fields['message']}")
        return fields

    def _feedback_loop(
        self,
        goal_handle,
        max_duration: float,
        reward_threshold: float,
        stop_event: threading.Event,
    ) -> str:
        """
        Run the episode feedback loop until a termination condition is met.

        ``stop_event`` is set by an action cancel (including one forwarded by
        ~/cancel_episode), ~/end_episode, an end_* press, or deactivate.
        Waiting on it instead of sleeping makes a stop take effect at once
        rather than up to one interval late.

        Returns:
            Termination reason string.

        """
        interval = 1.0 / self.get_parameter("feedback_rate_hz").value
        # monotonic: an NTP step must not stretch or cut the episode timeout.
        start = time.monotonic()

        while not stop_event.wait(interval):
            elapsed = time.monotonic() - start
            source, reward, outcome = self._episode.snapshot()
            reason = termination_reason(elapsed, max_duration, reward, reward_threshold)
            if reason is not None:
                return reason

            # Publish HIL intervention state (0=policy, 1=human)
            if self._intervention_pub is not None:
                intervention_msg = Int8()
                intervention_msg.data = 0 if source == Control.POLICY else 1
                self._intervention_pub.publish(intervention_msg)

            # outcome is the label so far. A reward-threshold end promotes an
            # unlabeled episode to success only after the loop returns, so that
            # never shows here.
            status = ManageEpisodeStatus(
                elapsed_s=elapsed,
                current_reward=reward,
                control_source=source,
                outcome=outcome,
                messages_written=self._recorder_messages_written,
            )
            self._status_pub.publish(status)
            if goal_handle is not None:
                goal_handle.publish_feedback(copy_fields(status, ManageEpisode.Feedback()))

        # Whoever set the stop event recorded why in stop_reason.
        return self.stop_reason or TERMINATION_STOPPED

    def _send_child_goal(self, client, goal, name: str, *, feedback_callback=None):
        """Send a goal to a child action server; return the handle or None."""
        return send_goal(client, goal, name=name, logger=self.get_logger(), feedback_callback=feedback_callback)

    def _cancel_child(self, goal_handle, name: str, wait_result: bool = False):
        """Best-effort cancel of a child goal; with ``wait_result``, return its result."""
        return cancel_goal(goal_handle, name=name, logger=self.get_logger(), wait_result=wait_result)

    def _on_recorder_feedback(self, fb) -> None:
        """Track the recorder's live message count for ManageEpisode feedback."""
        # Single int write: GIL-atomic, no lock needed.
        self._recorder_messages_written = fb.feedback.messages_written

    def _stop_recorder(self):
        """Cancel the recorder and wait for its result, which carries bag_path.

        So the recorder's ``termination_reason`` is always ``"cancelled"``,
        whatever ended the episode. The real reason is on the ManageEpisode
        result.
        """
        result = self._cancel_child(self._recorder_goal_handle, "Episode recorder", wait_result=True)
        if result is not None:
            self.get_logger().info(f"Recorder stopped: {result.messages_written} messages -> {result.bag_path}")
        return result

    def _cancel_all_children(self) -> None:
        """Best-effort cancel of all child action goals.

        All three wait for the result: their goal slots must be free before
        the retry that usually follows, and the recorder's result means its
        bag is finalized.
        """
        self._cancel_child(self._policy_goal_handle, "Robot policy", wait_result=True)
        self._cancel_child(self._reward_goal_handle, "Reward classifier", wait_result=True)
        self._cancel_child(self._recorder_goal_handle, "Episode recorder", wait_result=True)


def main(args=None):
    """Run the human-in-the-loop manager node."""
    return spin_lifecycle_node(HilManagerNode, args=args)


if __name__ == "__main__":
    sys.exit(main())
