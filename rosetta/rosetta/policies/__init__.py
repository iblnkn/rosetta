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
The policy side: learning frameworks adapted onto Rosetta's frames.

A policy framework, to Rosetta, is anything that implements two seams:

- :class:`DatasetWriter` — offline: consume frame dicts, write the
  framework's dataset format (training data).
- :class:`PolicyRunner` — online: drive a policy against a live robot
  through a :class:`~rosetta.frames.protocols.FrameIO` (inference).

Implementations live in their own adapter packages (LeRobot, ...) and are
resolved by name from setuptools entry points, so this
package never imports a framework — and no framework ever imports the robot
side. The registered object must be a class with a zero-argument constructor
implementing the protocol; the loaders validate this structurally at load
time and fail fast otherwise. An adapter registers itself like so::

    # in the adapter package's setup.py / pyproject.toml
    entry_points={
        'rosetta.dataset_writers': [
            'lerobot = lerobot_rosetta.dataset_writer:LeRobotDatasetWriter',
        ],
        'rosetta.policy_runners': [
            'lerobot = lerobot_rosetta.policy_runner:LeRobotPolicyRunner',
        ],
    }

This side imports only Rosetta's waist (:mod:`rosetta.contract`,
:mod:`rosetta.frames`) — never :mod:`rosetta.robots`.

Modules: :mod:`.protocols` is what an adapter *implements* (the two
Protocols and the value types they exchange); :mod:`.registry` is how the
hosting node *resolves* one by name at runtime.
"""

from __future__ import annotations

from .protocols import DatasetWriter, NodeLike, PolicyRunner, RunnerFeedback, RunnerResult
from .registry import (
    available_dataset_writers,
    available_policy_runners,
    load_dataset_writer,
    load_policy_runner,
)

__all__ = [
    "DatasetWriter",
    "NodeLike",
    "PolicyRunner",
    "RunnerFeedback",
    "RunnerResult",
    "available_dataset_writers",
    "available_policy_runners",
    "load_dataset_writer",
    "load_policy_runner",
]
