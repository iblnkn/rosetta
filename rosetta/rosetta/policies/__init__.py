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
The policy seam: running a trained policy against a live robot.

A policy runner, to Rosetta, implements :class:`PolicyRunner`: it drives a
policy against a live robot through a
:class:`~rosetta.frames.protocols.FrameIO` (inference). ``policy_runner_node``
loads the runner it names and hosts it through a :class:`RunnerHost`.

Implementations live in their own packages (``lerobot_rosetta``, ...) and
are resolved by name from setuptools entry points, so this package never
imports a framework, and no framework ever imports the robot side. The
registered object must be a class with a zero-argument constructor
implementing the protocol; the loader validates this structurally at load
time and fails fast otherwise. A package registers itself like so::

    # in the package's setup.py / pyproject.toml
    entry_points={
        'rosetta.policy_runners': [
            'lerobot = lerobot_rosetta.policy_runner:LeRobotPolicyRunner',
        ],
    }

This seam imports only lower layers of core (:mod:`rosetta.contract`,
:mod:`rosetta.frames`), never :mod:`rosetta.robots` or :mod:`rosetta.data`.
A framework that hosts a live run itself opens its robot through
:func:`rosetta.robots.registry.load_robot_interface` and ``connect``; it
never imports a robot interface package.

Modules: :mod:`.protocols` is what a runner *implements* (the protocol and
the value types it exchanges); :mod:`.registry` is how the hosting node
*resolves* one by name at runtime.
"""

from __future__ import annotations

from .protocols import PolicyRunner, RunnerFeedback, RunnerHost, RunnerResult
from .registry import POLICY_RUNNER_GROUP, available_policy_runners, load_policy_runner

__all__ = [
    "POLICY_RUNNER_GROUP",
    "PolicyRunner",
    "RunnerFeedback",
    "RunnerHost",
    "RunnerResult",
    "available_policy_runners",
    "load_policy_runner",
]
