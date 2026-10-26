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

"""What a robot interface package gives Rosetta's core.

A contract names its robot interface (``robot_interface: ros2``). Loading the
contract looks that name up in :mod:`.registry` and asks the interface the
questions only it can answer: which message types exist, which timelines each
one carries, and whether a ``qos`` mapping parses. Codec discovery asks every
installed interface to register its built-in codecs.

The live and recorded frame surfaces an interface also provides
(:class:`~rosetta.frames.protocols.FrameIO`, an iterator of frame dicts) are
used directly by the interface's own nodes and tools, so they need no entry
here.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["RobotInterface"]


@runtime_checkable
class RobotInterface(Protocol):
    """A robot interface, registered under ``rosetta.robot_interfaces``.

    The entry point is a class with a zero-argument constructor. Every method
    is called during contract loading, possibly on a machine that only loads
    contracts, so construction and these methods should import their
    middleware lazily and cheaply.
    """

    def register_codecs(self) -> None:
        """Register this interface's built-in decoders and encoders.

        Called by :func:`rosetta.frames.codecs.discover_codecs` before codec
        plugins load, so a plugin can still replace a built-in with
        ``override=True``. Called once per process.
        """
        ...

    def provided_timelines(self, msg_type: str) -> frozenset[str]:
        """Timelines a channel of ``msg_type`` provides, by name.

        Contract loading checks every ``align.timeline`` against this set.

        Raises:
            ValueError: If ``msg_type`` is not a type this interface knows.

        """
        ...

    def validate_qos(self, qos: dict[str, Any]) -> None:
        """Check a channel's ``qos`` mapping.

        Raises:
            ValueError: If the mapping does not parse for this interface.

        """
        ...
