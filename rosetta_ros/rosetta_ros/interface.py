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

"""The ``ros2`` robot interface, as contract loading sees it.

Registered under ``rosetta.robot_interfaces`` as ``ros2``. Imports nothing
from ROS at module level: rclpy and rosidl load when a method needs them.
"""

from __future__ import annotations

from typing import Any


class Ros2Interface:
    """:class:`~rosetta.robots.protocols.RobotInterface` for ROS 2 message types and QoS."""

    def register_codecs(self) -> None:
        """Import the built-in decoders and encoders, which register on import."""
        from . import decoders, encoders  # noqa: F401

    def provided_timelines(self, msg_type: str) -> frozenset[str]:
        """``receive`` for every type, plus ``header`` for types with a std_msgs Header."""
        from .timelines import provided_timelines

        return provided_timelines(msg_type)

    def validate_qos(self, qos: dict[str, Any]) -> None:
        """Parse ``qos`` into an rclpy QoS profile, raising ValueError if it doesn't parse."""
        from .rclpy_utils import qos_profile_from_dict

        qos_profile_from_dict(qos)
