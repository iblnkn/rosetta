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

"""A live robot interface: what contract loading asks, plus ``connect``.

:class:`RobotInterface` extends :class:`rosetta.contract.interfaces.ContractInterface`,
the questions contract loading asks an interface (message types, timelines,
qos, built-in codecs). A framework that hosts a run itself (LeRobot's CLIs
driving a ``Robot``) also asks the interface for a live robot, through
:meth:`RobotInterface.connect`. The interface's own nodes and tools build
their live and recorded frame surfaces directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from rosetta.contract.interfaces import ContractInterface

if TYPE_CHECKING:
    from rosetta.contract.model import TeleopEventMap
    from rosetta.contract.specs import ActionStreamSpec, ObservationStreamSpec
    from rosetta.frames.protocols import Connection

__all__ = ["RobotInterface"]


@runtime_checkable
class RobotInterface(ContractInterface, Protocol):
    """A robot interface, registered under ``rosetta.robot_interfaces``.

    The entry point is a class with a zero-argument constructor. Every method
    but :meth:`connect` is called during contract loading (see
    :class:`~rosetta.contract.interfaces.ContractInterface`).
    """

    def connect(
        self,
        *,
        observations: Sequence[ObservationStreamSpec],
        actions: Sequence[ActionStreamSpec],
        fps: int,
        name: str,
        events: TeleopEventMap | None = None,
        use_sim_time: bool = False,
    ) -> Connection:
        """Open a live robot for a framework that hosts the run itself.

        The connection samples ``observations`` into frames and publishes
        ``actions`` from them, at ``fps``. With ``events``, it also follows
        that channel and reports each event's latest value. ``name``
        identifies the connection to the middleware (for example a node name).

        With ``use_sim_time`` the connection runs on the simulator's clock
        rather than wall time, and its ``clock`` reports ``simulated``.

        Returns the open connection, which the caller closes.
        """
        ...
