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

"""Resolving a :class:`~.protocols.DatasetWriter` or :class:`~.readers.FrameReader` by name.

A package registers its writer under :data:`DATASET_WRITER_GROUP` (see
:mod:`rosetta.data` for the registration snippet); these helpers look it up
at runtime without this package ever importing it. Loading fails fast: an
unknown or ambiguous name raises ``ValueError``; a non-class or
non-conforming entry point raises ``TypeError``; import and constructor
failures propagate raw with their full tracebacks.
"""

from __future__ import annotations

import importlib.metadata as _ilm
from typing import Any

from ..plugins import instantiate_entry_point, protocol_problems, resolve_entry_point
from .protocols import DatasetWriter
from .readers import FrameReader

__all__ = [
    "DATASET_WRITER_GROUP",
    "FRAME_READER_GROUP",
    "available_dataset_writers",
    "available_frame_readers",
    "frame_reader_class",
    "load_dataset_writer",
    "load_frame_reader",
]

DATASET_WRITER_GROUP = "rosetta.dataset_writers"
"""Entry-point group a package registers its :class:`~.protocols.DatasetWriter` under."""


def load_dataset_writer(name: str) -> DatasetWriter:
    """Instantiate the ``DatasetWriter`` registered as ``name``.

    The entry point must be a class whose zero-argument instances implement
    :class:`~.protocols.DatasetWriter`: a non-class or non-conforming entry
    point raises ``TypeError`` here; import/constructor failures propagate
    raw with their full tracebacks.
    """
    eps = list(_ilm.entry_points(group=DATASET_WRITER_GROUP))
    return instantiate_entry_point(eps, DATASET_WRITER_GROUP, name, DatasetWriter, "dataset writer")


def available_dataset_writers() -> list[str]:
    """Writer names registered under :data:`DATASET_WRITER_GROUP`."""
    return sorted({ep.name for ep in _ilm.entry_points(group=DATASET_WRITER_GROUP)})


FRAME_READER_GROUP = "rosetta.frame_readers"
"""Entry-point group a package registers its :class:`~.readers.FrameReader` class under."""


def load_frame_reader(name: str, **options: Any) -> FrameReader:
    """Construct the ``FrameReader`` registered as ``name`` with ``options``.

    Unlike writers, a reader is built from options (where the recordings are,
    which contract), so its class takes keyword arguments. An unknown name
    raises ``ValueError``; a non-class or non-conforming entry point raises
    ``TypeError``; import and constructor failures propagate raw.
    """
    reader = frame_reader_class(name)(**options)
    if not isinstance(reader, FrameReader):
        problems = ", ".join(protocol_problems(FrameReader, reader))
        raise TypeError(
            f"Entry point '{name}' in group '{FRAME_READER_GROUP}' does not implement FrameReader "
            f"(missing or non-callable: {problems})."
        )
    return reader


def frame_reader_class(name: str) -> type:
    """The class registered as frame reader ``name``, unconstructed, to inspect what options it takes.

    An unknown name raises ``ValueError``; a non-class entry point raises
    ``TypeError``; import failures propagate raw.
    """
    eps = list(_ilm.entry_points(group=FRAME_READER_GROUP))
    cls = resolve_entry_point(eps, FRAME_READER_GROUP, name, "frame reader").load()
    if not isinstance(cls, type):
        raise TypeError(f"Entry point '{name}' in group '{FRAME_READER_GROUP}' must be a class; got {cls!r}.")
    return cls


def available_frame_readers() -> list[str]:
    """Reader names registered under :data:`FRAME_READER_GROUP`."""
    return sorted({ep.name for ep in _ilm.entry_points(group=FRAME_READER_GROUP)})
