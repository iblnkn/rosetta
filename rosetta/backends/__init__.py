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
Backend abstraction layer.

Rosetta's core (contract, ops, converters, StreamBuffer) and ROS2 I/O are
backend-neutral: they converge every robot onto a flat *frame dict*
(``{contract_key: np.ndarray | str, ...}``). A *backend* is a leaf package that
consumes that frame dict to talk to a specific policy-learning framework
(LeRobot, vla_foundry, ...).

This package defines the two seams every backend implements:

- :class:`DatasetWriter` -- offline: frame dicts -> a framework's dataset format.
- :class:`PolicyRunner`   -- online: drive a policy against a live
  :class:`~rosetta.ros2.topic_bridge.TopicBridge`.

Backends register implementations under setuptools entry-point groups
``rosetta.dataset_writers`` and ``rosetta.policy_runners`` so that ``rosetta``
core imports neither LeRobot nor vla_foundry.
"""

from .protocols import (
    DatasetWriter,
    PolicyRunner,
    RunnerFeedback,
    RunnerResult,
    load_dataset_writer,
    load_policy_runner,
)

__all__ = [
    'DatasetWriter',
    'PolicyRunner',
    'RunnerFeedback',
    'RunnerResult',
    'load_dataset_writer',
    'load_policy_runner',
]
