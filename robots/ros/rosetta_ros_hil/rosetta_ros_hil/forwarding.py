# Copyright 2026 Isaac Blankenau
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

"""Takeover: the robot's command topics carry the policy's output or the human's, whichever has control.

The policy runner publishes every contract action topic under a prefix
(``policy_runner_launch.py output_prefix:=``), and :class:`CommandForwarder`
republishes each one to the real topic while the policy has control. While the
human has control it drops those and instead re-encodes each teleop input onto
the action it targets: decoded with the teleop input's ``select``/``apply`` and
encoded with the action's, so the device and the robot may differ in message
type, field layout and units. Teleop feedback runs whoever has control.

There is no command watchdog here: while a person drives they are the safety,
and the policy runner's own watchdog already covers what it publishes.
"""

from __future__ import annotations

from typing import Callable

from rosidl_runtime_py.utilities import get_message

from rosetta.contract.schema import Contract
from rosetta.contract.specs import (
    StreamSpec,
    iter_action_specs,
    iter_observation_specs,
    iter_teleop_feedback_specs,
    iter_teleop_input_specs,
)
from rosetta.frames.codecs import decode_value, encode_value
from rosetta_ros.rclpy_utils import prefixed_topic, qos_profile_from_dict
from rosetta_ros_hil.hil_episode import Control


def _only_spec(specs: list[StreamSpec], topic: str, what: str) -> StreamSpec:
    """The single spec on ``topic``. Forwarding re-encodes whole messages, so two sources on one topic are ambiguous."""
    matches = [s for s in specs if s.source.channel.topic == topic]
    if len(matches) != 1:
        raise ValueError(
            f"{what} '{topic}' carries {len(matches)} contract sources; forwarding needs exactly one. "
            f"Give each source its own topic."
        )
    return matches[0]


class CommandForwarder:
    """Forwards the policy's commands or the human's onto the contract's action topics, per message.

    Creates its publishers and subscriptions on ``node`` (in ``callback_group``)
    and asks ``control`` who has control for every message. Raises ValueError
    if a teleop target or feedback origin topic is ambiguous (see :func:`_only_spec`).
    """

    def __init__(
        self,
        node,
        contract: Contract,
        policy_prefix: str,
        control: Callable[[], Control],
        callback_group,
    ) -> None:
        self._node = node
        self._control = control
        self._callback_group = callback_group
        self._publishers: dict[str, object] = {}  # real action topic -> publisher
        self._feedback_publishers: list = []
        self._subscriptions: list = []

        try:
            # One publisher and one subscription per action topic, however many
            # sources (and so specs) share it: the policy's message is forwarded whole.
            channels = {src.channel.topic: src.channel for entry in contract.actions for src in entry.sources}
            for topic, channel in channels.items():
                msg_cls = get_message(channel.type)
                qos = qos_profile_from_dict(channel.qos)
                self._publishers[topic] = node.create_publisher(msg_cls, topic, qos)
                policy_topic = prefixed_topic(policy_prefix, topic)
                self._subscribe(msg_cls, policy_topic, qos, lambda msg, t=topic: self._on_policy(msg, t))
                node.get_logger().info(f"Policy output: {policy_topic} -> {topic}")

            teleop = contract.teleop
            if teleop is not None and teleop.input:
                action_specs = list(iter_action_specs(contract))
                for tis, teleop_spec in zip(teleop.input, iter_teleop_input_specs(contract), strict=True):
                    action_spec = _only_spec(action_specs, tis.target, "Teleop input target")
                    channel = teleop_spec.source.channel
                    self._subscribe(
                        get_message(channel.type),
                        channel.topic,
                        qos_profile_from_dict(channel.qos),
                        lambda msg, t=teleop_spec, a=action_spec: self._on_teleop_input(msg, t, a),
                    )
                    node.get_logger().info(f"Teleop input: {channel.topic} -> {tis.target}")

            if teleop is not None and teleop.feedback:
                observation_specs = list(iter_observation_specs(contract))
                for tfs, feedback_spec in zip(teleop.feedback, iter_teleop_feedback_specs(contract), strict=True):
                    origin_spec = _only_spec(observation_specs, tfs.origin, "Teleop feedback origin")
                    out = feedback_spec.source.channel
                    pub = node.create_publisher(get_message(out.type), out.topic, qos_profile_from_dict(out.qos))
                    self._feedback_publishers.append(pub)
                    origin = origin_spec.source.channel
                    self._subscribe(
                        get_message(origin.type),
                        origin.topic,
                        qos_profile_from_dict(origin.qos),
                        lambda msg, o=origin_spec, f=feedback_spec, p=pub: self._on_feedback_origin(msg, o, f, p),
                    )
                    node.get_logger().info(f"Teleop feedback: {origin.topic} -> {out.topic}")
        except Exception:
            self.destroy()
            raise

    def _subscribe(self, msg_cls, topic: str, qos, callback) -> None:
        self._subscriptions.append(
            self._node.create_subscription(msg_cls, topic, callback, qos, callback_group=self._callback_group)
        )

    def destroy(self) -> None:
        """Destroy every publisher and subscription. Safe to call twice."""
        for sub in self._subscriptions:
            self._node.destroy_subscription(sub)
        for pub in [*self._publishers.values(), *self._feedback_publishers]:
            self._node.destroy_publisher(pub)
        self._subscriptions = []
        self._publishers = {}
        self._feedback_publishers = []

    def _now_ns(self) -> int:
        return self._node.get_clock().now().nanoseconds

    def _on_policy(self, msg, topic: str) -> None:
        if self._control() == Control.POLICY:
            self._publishers[topic].publish(msg)

    def _on_teleop_input(self, msg, teleop_spec, action_spec) -> None:
        if self._control() != Control.TELEOP:
            return
        try:
            command = encode_value(decode_value(msg, teleop_spec), action_spec, self._now_ns())
        except Exception as e:
            self._node.get_logger().error(
                f"Teleop input {teleop_spec.source.channel.topic} -> {action_spec.source.channel.topic} failed: {e}",
                throttle_duration_sec=1.0,
            )
            return
        self._publishers[action_spec.source.channel.topic].publish(command)

    def _on_feedback_origin(self, msg, origin_spec, feedback_spec, pub) -> None:
        try:
            pub.publish(encode_value(decode_value(msg, origin_spec), feedback_spec, self._now_ns()))
        except Exception as e:
            self._node.get_logger().error(
                f"Teleop feedback {origin_spec.source.channel.topic} -> {feedback_spec.source.channel.topic} "
                f"failed: {e}",
                throttle_duration_sec=1.0,
            )
