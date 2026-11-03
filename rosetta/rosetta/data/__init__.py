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

A :class:`FrameReader` is the other schedule: it applies the contract at
training time, serving frames from recordings straight to a framework's
trainer, so a contract change needs no new port. Readers register under
``rosetta.frame_readers`` and take keyword options (see :mod:`.readers`).

This seam imports only lower layers of core (:mod:`rosetta.contract`,
:mod:`rosetta.frames`), never :mod:`rosetta.robots` or
:mod:`rosetta.policies`.

Modules: :mod:`.protocols` and :mod:`.readers` are what writers and readers
*implement*; :mod:`.registry` resolves either by name; :mod:`.stats` is
shared by readers.
"""

from __future__ import annotations

from .protocols import DatasetWriter
from .readers import FrameReader, ReaderInfo, StreamingFrameReader, Windows
from .registry import (
    DATASET_WRITER_GROUP,
    FRAME_READER_GROUP,
    available_dataset_writers,
    available_frame_readers,
    load_dataset_writer,
    load_frame_reader,
)

__all__ = [
    "DATASET_WRITER_GROUP",
    "FRAME_READER_GROUP",
    "DatasetWriter",
    "FrameReader",
    "ReaderInfo",
    "StreamingFrameReader",
    "Windows",
    "available_dataset_writers",
    "available_frame_readers",
    "load_dataset_writer",
    "load_frame_reader",
]
