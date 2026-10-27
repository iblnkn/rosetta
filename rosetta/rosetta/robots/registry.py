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

Robot interface packages register a class under :data:`ROBOT_INTERFACE_GROUP`,
named after the ``robot_interface`` value contracts use::

    # in the interface package's setup.py / pyproject.toml
    entry_points={
        'rosetta.robot_interfaces': [
            'ros2 = rosetta_ros.interface:Ros2Interface',
        ],
    }

Loading follows :mod:`rosetta.policies.registry`: an unknown or ambiguous name
raises ``ValueError``, a non-class or non-conforming entry point raises
``TypeError``, and import or constructor failures propagate raw. One instance
per name is kept for the process.
"""

from __future__ import annotations

import importlib.metadata as _ilm
import threading

from ..contract.errors import ContractValidationError
from ..contract.plugins import instantiate_entry_point
from .protocols import RobotInterface

__all__ = ["ROBOT_INTERFACE_GROUP", "available_robot_interfaces", "load_robot_interface"]

ROBOT_INTERFACE_GROUP = "rosetta.robot_interfaces"
"""Entry-point group a robot interface registers its :class:`~.protocols.RobotInterface` under."""

_instances: dict[str, RobotInterface] = {}
# Codec registration: "done" once every installed interface was tried,
# "failures" maps each interface that failed to its latched error.
_codec_state: dict = {"done": False, "failures": {}}
# RLock: registering codecs imports codec modules, which may look up the registry.
_lock = threading.RLock()


def available_robot_interfaces() -> list[str]:
    """Robot interface names registered under :data:`ROBOT_INTERFACE_GROUP`."""
    return sorted({ep.name for ep in _ilm.entry_points(group=ROBOT_INTERFACE_GROUP)})


def load_robot_interface(name: str) -> RobotInterface:
    """The :class:`~.protocols.RobotInterface` registered as ``name``, created once per process."""
    with _lock:
        if name not in _instances:
            eps = list(_ilm.entry_points(group=ROBOT_INTERFACE_GROUP))
            _instances[name] = instantiate_entry_point(
                eps, ROBOT_INTERFACE_GROUP, name, RobotInterface, "robot interface"
            )
        return _instances[name]


def register_interface_codecs(required: str | None = None) -> None:
    """Register every installed robot interface's built-in codecs, once per process.

    Every interface is tried, so one broken package never blocks contracts for
    the others. Each failure is latched like a failed codec plugin (see
    :func:`rosetta.contract.plugins.load_entry_point_plugins`): an interface that
    registered some codecs before failing would trip the duplicate-registration
    guard on a retry and hide the real error. It is raised, as
    ContractValidationError, only when ``required`` names that interface.
    """
    with _lock:
        if not _codec_state["done"]:
            for name in available_robot_interfaces():
                try:
                    load_robot_interface(name).register_codecs()
                except Exception as e:
                    failure = ContractValidationError(f"Failed to load robot interface '{name}': {e}")
                    failure.__cause__ = e
                    _codec_state["failures"][name] = failure
            _codec_state["done"] = True
        failure = _codec_state["failures"].get(required)
    if failure is not None:
        raise failure
