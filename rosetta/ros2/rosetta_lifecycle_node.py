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
Standalone lifecycle wrapper around :class:`TopicBridge`.

Used when a framework adapter (e.g. ``lerobot_robot_rosetta.Rosetta``) needs to
own its ROS2 node instead of attaching to an externally-managed bridge. It only
owns the bridge lifecycle.
"""

from __future__ import annotations

from typing import Any

from rclpy.lifecycle import Node, State, TransitionCallbackReturn

from rosetta.core.contract import ActionStreamSpec, ObservationStreamSpec
from rosetta.ros2.topic_bridge import TopicBridge


class RosettaLifecycleNode(Node):
    """Lifecycle wrapper around a :class:`TopicBridge` for standalone mode."""

    def __init__(
        self,
        node_name: str,
        observation_specs: list[ObservationStreamSpec],
        action_specs: list[ActionStreamSpec],
        fps: int,
        **kwargs,
    ):
        super().__init__(node_name, **kwargs)
        self._bridge = TopicBridge(observation_specs, action_specs, fps)

    @property
    def bridge(self) -> TopicBridge:
        return self._bridge

    def on_configure(self, _state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_configure() is called.")
        self._bridge.setup(self)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_activate() is called.")
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_deactivate() is called.")
        self._bridge.send_safety_action()
        self._bridge._last_action_ns = None
        return super().on_deactivate(state)

    def on_cleanup(self, _state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_cleanup() is called.")
        self._bridge.teardown()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, _state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_shutdown() is called.")
        self._bridge.teardown()
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().error(f"Error occurred in state: {state.label}")
        try:
            self._bridge.teardown()
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"Error during cleanup: {e}")
        return TransitionCallbackReturn.SUCCESS

    # -------------------- Bridge passthrough --------------------

    @property
    def is_active(self) -> bool:
        return self._bridge.is_active

    @property
    def is_configured(self) -> bool:
        return self._bridge.is_configured

    def sample_frame(self) -> dict[str, Any]:
        return self._bridge.sample_frame()

    def publish_frame(self, action_frame: dict[str, Any]) -> dict[str, Any]:
        return self._bridge.publish_frame(action_frame)

    def reset_state(self) -> None:
        self._bridge.reset_state()
