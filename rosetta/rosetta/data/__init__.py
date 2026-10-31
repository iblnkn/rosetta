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
The data seam: writing contract frames into a framework's dataset format.

A dataset writer, to Rosetta, implements :class:`DatasetWriter`: it consumes
frame dicts offline and writes training data. ``rosetta_port`` replays
recordings into frames and hands them to the writer it names.

Implementations live in their own packages (``lerobot_dataset_rosetta``,
...) and are resolved by name from setuptools entry points, so this package
never imports a framework. The registered object must be a class with a
zero-argument constructor implementing the protocol; the loader validates
this structurally at load time and fails fast otherwise. A package registers
itself like so::

    # in the package's setup.py / pyproject.toml
    entry_points={
        'rosetta.dataset_writers': [
            'lerobot = lerobot_dataset_rosetta.dataset_writer:LeRobotDatasetWriter',
        ],
    }

Reading data back is not a seam: a reader such as ``rerun_rosetta`` hands
frames straight to a training loop, so nothing in Rosetta resolves one by
name.

This seam imports only lower layers of core (:mod:`rosetta.contract`,
:mod:`rosetta.frames`), never :mod:`rosetta.robots` or
:mod:`rosetta.policies`.

Modules: :mod:`.protocols` is what a writer *implements*; :mod:`.registry`
is how the porter *resolves* one by name.
"""

from __future__ import annotations

from .protocols import DatasetWriter
from .registry import DATASET_WRITER_GROUP, available_dataset_writers, load_dataset_writer

__all__ = ["DATASET_WRITER_GROUP", "DatasetWriter", "available_dataset_writers", "load_dataset_writer"]
