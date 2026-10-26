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

"""A ROS 2 node as a :class:`~rosetta.policies.protocols.RunnerHost`: runner settings are node parameters."""

from __future__ import annotations

from typing import Any

from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node

__all__ = ["RosParameterHost"]


class RosParameterHost:
    """Hosts a policy runner on ``node``: each setting is a parameter, so launch and params files set it."""

    def __init__(self, node: Node) -> None:
        self._node = node

    def declare(self, name: str, default: Any, description: str = "", read_only: bool = False) -> None:
        """Declare parameter ``name`` unless the node (or an earlier setup) already did."""
        if not self._node.has_parameter(name):
            self._node.declare_parameter(
                name, default, ParameterDescriptor(description=description, read_only=read_only)
            )

    def get(self, name: str) -> Any:
        """The parameter's current value."""
        return self._node.get_parameter(name).value

    @property
    def logger(self) -> Any:
        """The node's logger."""
        return self._node.get_logger()
