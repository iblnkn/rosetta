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

"""What contract loading asks of a robot interface, and finding one by name.

A contract names its robot interface (``robot_interface: ros2``). Loading the
contract looks that name up among the installed ``rosetta.robot_interfaces``
entry points and asks the interface the questions only it can answer: which
message types exist, which timelines each one carries, and whether a ``qos``
mapping parses. Codec discovery asks every installed interface to register its
built-in codecs.

Those questions are :class:`ContractInterface`. A live robot is more:
:class:`rosetta.robots.protocols.RobotInterface` adds ``connect``, and
:func:`rosetta.robots.registry.load_robot_interface` returns the same
instance this module caches, checked against the full protocol. Interface
packages register a class under :data:`ROBOT_INTERFACE_GROUP`::

    # in the interface package's setup.py / pyproject.toml
    entry_points={
        'rosetta.robot_interfaces': [
            'ros2 = rosetta_ros.interface:Ros2Interface',
        ],
    }

An unknown or ambiguous name raises ``ValueError``, a non-class or
non-conforming entry point raises ``TypeError``, and import or constructor
failures propagate raw. One instance per name is kept for the process.
"""

from __future__ import annotations

import importlib.metadata as _ilm
import threading
from typing import Any, Protocol, runtime_checkable

from ..codecs import load_codec_plugins
from ..errors import ContractValidationError
from ..plugins import instantiate_entry_point

__all__ = [
    "ROBOT_INTERFACE_GROUP",
    "ContractInterface",
    "available_robot_interfaces",
    "discover_codecs",
    "load_interface",
]

ROBOT_INTERFACE_GROUP = "rosetta.robot_interfaces"
"""Entry-point group a robot interface registers its class under."""


@runtime_checkable
class ContractInterface(Protocol):
    """The part of a robot interface that contract loading uses.

    The entry point is a class with a zero-argument constructor. These methods
    run during contract loading, possibly on a machine that only loads
    contracts, so construction and the methods should import their middleware
    lazily and cheaply.
    """

    def register_codecs(self) -> None:
        """Register this interface's built-in decoders and encoders.

        Called by :func:`discover_codecs` before codec plugins load, so a
        plugin can still replace a built-in with ``override=True``. Called once
        per process.
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


_instances: dict[str, ContractInterface] = {}
# Codec registration: "done" once every installed interface was tried,
# "failures" maps each interface that failed to its latched error.
_codec_state: dict = {"done": False, "failures": {}}
# RLock: registering codecs imports codec modules, which may look up the registry.
_lock = threading.RLock()


def available_robot_interfaces() -> list[str]:
    """Robot interface names registered under :data:`ROBOT_INTERFACE_GROUP`."""
    return sorted({ep.name for ep in _ilm.entry_points(group=ROBOT_INTERFACE_GROUP)})


def load_interface(name: str) -> ContractInterface:
    """The robot interface registered as ``name``, created once per process."""
    with _lock:
        if name not in _instances:
            eps = list(_ilm.entry_points(group=ROBOT_INTERFACE_GROUP))
            _instances[name] = instantiate_entry_point(
                eps, ROBOT_INTERFACE_GROUP, name, ContractInterface, "robot interface"
            )
        return _instances[name]


def _register_interface_codecs(required: str | None = None) -> None:
    """Register every installed robot interface's built-in codecs, once per process.

    Every interface is tried, so one broken package never blocks contracts for
    the others. Each failure is latched like a failed codec plugin (see
    :func:`rosetta.plugins.load_entry_point_plugins`): an interface that
    registered some codecs before failing would trip the duplicate-registration
    guard on a retry and hide the real error. It is raised, as
    ContractValidationError, only when ``required`` names that interface.
    """
    with _lock:
        if not _codec_state["done"]:
            for name in available_robot_interfaces():
                try:
                    load_interface(name).register_codecs()
                except Exception as e:
                    failure = ContractValidationError(f"Failed to load robot interface '{name}': {e}")
                    failure.__cause__ = e
                    _codec_state["failures"][name] = failure
            _codec_state["done"] = True
        failure = _codec_state["failures"].get(required)
    if failure is not None:
        raise failure


def discover_codecs(robot_interface: str | None = None) -> None:
    """
    Register every robot interface's built-in codecs, then import ``rosetta.codecs`` plugins.

    Raises if ``robot_interface`` (the one a contract names) failed to register.

    Each entry point's value is a module path. Loading it runs that module's
    ``@register_decoder`` / ``@register_encoder`` decorators, so installed
    plugins populate the registries keyed by msg_type. The contract therefore
    names a ``type:`` only, so a custom codec for a brand-new type needs no path
    in the contract. To *replace* a built-in codec, the plugin registers with
    ``override=True`` (global), or a spec uses the inline ``decoder:``/
    ``encoder:`` path (per-spec, deterministic).

    Idempotent, hard-error semantics live in the shared loader (see
    :func:`rosetta.plugins.load_entry_point_plugins`): a plugin that
    fails to import raises ContractValidationError, and that failure is
    latched, so every later call re-raises the same error until the process
    restarts with a fixed environment.
    """
    # Each installed robot interface registers its built-in codecs first, so a
    # plugin can replace a built-in with override=True.
    _register_interface_codecs(robot_interface)
    load_codec_plugins()
