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
Frame I/O protocols: how the robot side and the policy side meet.

A live robot, to Rosetta, is a duck-typed feed of frame dicts
(``{contract_key: np.ndarray | str}``, see :mod:`rosetta.frames` for what a
frame is): observations sampled out on the clock set by the contract's
``fps``, actions published back in. These protocols name that surface so a
:class:`~rosetta.policies.PolicyRunner` can be written and tested against
frames alone, with no dependency on any pub/sub ecosystem.

``rosetta.robots.ros2.topic_bridge.TopicBridge`` is the ROS2 implementation.
A future robot interface (ROS1, zenoh, MQTT, ...) implements the same methods,
and every policy framework works with it unchanged.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class FrameIO(Protocol):
    """Bidirectional frame I/O with a live robot.

    What a :class:`~rosetta.policies.PolicyRunner` receives per goal: sample
    observations out, publish actions in, reset between episodes.

    This is the policy-facing surface only. A concrete implementation may
    expose richer robot-side methods beyond the protocol (e.g. TopicBridge's
    ``sample_values()``, the pre-assembly per-spec view); those never cross
    the robot/policy boundary.
    """

    def sample_frame(self) -> dict[str, Any]:
        """Current observation frame: ``{contract_key: np.ndarray | str}``."""
        ...

    @property
    def warmed_up(self) -> bool:
        """True once every observation stream has delivered at least one value."""
        ...

    def publish_frame(self, action_frame: dict[str, Any]) -> dict[str, Any]:
        """Publish one action frame (``{contract_key: np.ndarray}``). Returns it."""
        ...

    def send_safety_action(self) -> None:
        """Publish the contract's safety action (e.g. zero velocity)."""
        ...

    def reset_state(self) -> None:
        """Clear per-episode state (buffers, warmup) between runs."""
        ...
