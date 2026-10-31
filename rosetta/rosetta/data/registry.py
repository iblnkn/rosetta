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

"""Resolving a :class:`~.protocols.DatasetWriter` by name.

A package registers its writer under :data:`DATASET_WRITER_GROUP` (see
:mod:`rosetta.data` for the registration snippet); these helpers look it up
at runtime without this package ever importing it. Loading fails fast: an
unknown or ambiguous name raises ``ValueError``; a non-class or
non-conforming entry point raises ``TypeError``; import and constructor
failures propagate raw with their full tracebacks.
"""

from __future__ import annotations

import importlib.metadata as _ilm

from ..plugins import instantiate_entry_point
from .protocols import DatasetWriter

__all__ = ["DATASET_WRITER_GROUP", "available_dataset_writers", "load_dataset_writer"]

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
