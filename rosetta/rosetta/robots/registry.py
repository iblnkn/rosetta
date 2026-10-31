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

"""Resolving a :class:`~.protocols.RobotInterface` by the name a contract declares.

Robot interface packages register a class under :data:`ROBOT_INTERFACE_GROUP`
(see :mod:`rosetta.contract.interfaces`, which finds and caches it for contract
loading). :func:`load_robot_interface` returns that same instance, checked
against the full live protocol, ``connect`` included.
"""

from __future__ import annotations

from ..contract.interfaces import ROBOT_INTERFACE_GROUP, available_robot_interfaces, load_interface
from ..plugins import protocol_problems
from .protocols import RobotInterface

__all__ = ["ROBOT_INTERFACE_GROUP", "available_robot_interfaces", "load_robot_interface"]


def load_robot_interface(name: str) -> RobotInterface:
    """The :class:`~.protocols.RobotInterface` registered as ``name``, created once per process.

    Raises ``TypeError`` if the interface answers contract loading but cannot
    open a live robot.
    """
    interface = load_interface(name)
    if not isinstance(interface, RobotInterface):
        problems = protocol_problems(RobotInterface, interface)
        raise TypeError(
            f"Robot interface '{name}' ({type(interface).__name__}) does not implement "
            f"RobotInterface (missing or non-callable: {', '.join(problems)})."
        )
    return interface
