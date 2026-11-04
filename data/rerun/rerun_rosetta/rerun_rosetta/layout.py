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

"""Recording layouts: how a robot interface's recordings look in Rerun.

This package knows Rerun's dataloader and nothing about any robot middleware.
What it needs from the middleware, where a channel's messages sit in a
recording and how one becomes the input that interface's decoders take, comes
from a layout the interface package registers under ``rerun_rosetta.layouts``,
named after the contract's ``robot_interface``::

    entry_points={
        "rerun_rosetta.layouts": ["ros2 = rosetta_ros.offline.rrd_layout:Rosbag2RrdLayout"],
    }
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from rosetta.contract.specs import iter_observation_specs
from rosetta.plugins import instantiate_entry_point

if TYPE_CHECKING:
    import pyarrow as pa
    from rerun.catalog import DatasetEntry

    from rosetta.contract.schema import Channel, Contract

LAYOUT_GROUP = "rerun_rosetta.layouts"


@runtime_checkable
class RecordingLayout(Protocol):
    """How one robot interface's recordings are laid out in a Rerun recording."""

    receive_timeline: str
    """The contract timeline whose values are the recording's receive times, such as ``receive``."""

    index: str
    """The Rerun timeline holding those receive times."""

    def column(self, channel: Channel, columns: Collection[str]) -> str | None:
        """The column holding ``channel``'s messages, of the recording's ``columns``.

        Returns None when the channel wasn't recorded. Raises ``ValueError``,
        naming the fix, when it was recorded only in a form this layout can't
        read.
        """
        ...

    def message(self, cell: pa.Scalar) -> Any:
        """One recorded message, as the interface's decoders take it."""
        ...


def load_layout(robot_interface: str) -> RecordingLayout:
    """The layout registered for ``robot_interface``.

    Resolved through Rosetta's shared entry-point loader, so it fails the way
    every Rosetta registry does.

    Raises:
        ValueError: If no installed package, or more than one, registers a layout under that name.
        TypeError: If the entry point isn't a class, or the class doesn't implement
            :class:`RecordingLayout` (the error names each missing member).

    """
    eps = list(entry_points(group=LAYOUT_GROUP))
    return instantiate_entry_point(eps, LAYOUT_GROUP, robot_interface, RecordingLayout, "Rerun recording layout")


def observation_starts(entry: DatasetEntry, contract: Contract, segments: list[str] | None = None) -> dict[str, int]:
    """Each segment holding every observation stream, with the index value at which the last of them starts.

    That is where the porter's warmup ends: its first frame is the first tick
    at or after it. The porter skips an episode missing an observation topic,
    and so does this reader, warning about each one left out. Rerun's manifest
    scan would instead fail on it outright, so the segments are checked first:
    one query per stream's entity, for each segment's first index value with
    any of its rows. A missing action stream is left to zero-fill, as the
    porter does.
    """
    import pyarrow as pa
    from datafusion import col
    from datafusion import functions as f

    layout = load_layout(contract.robot_interface)
    columns = entry.schema().column_names()
    candidates = sorted(segments if segments is not None else entry.segment_ids())
    entities: dict[str, str] = {}
    for spec in iter_observation_specs(contract):
        try:
            column = layout.column(spec.source.channel, columns)
        except ValueError:
            continue  # compile_contract reports a stream recorded in a form the layout can't read
        if column is not None:
            entities[f"{spec.key} ({spec.source.channel.topic})"] = column.split(":", 1)[0]
    first: dict[str, dict[str, int]] = {}
    for entity in set(entities.values()):
        table = (
            entry.filter_contents([f"{entity}/**"])
            .reader(index=layout.index)
            .aggregate([col("rerun_segment_id")], [f.min(col(layout.index)).alias("first")])
            .to_arrow_table()
        )
        segment_ids = table.column("rerun_segment_id").to_pylist()
        first[entity] = dict(zip(segment_ids, table.column("first").cast(pa.int64()).to_pylist(), strict=True))
    starts = {}
    for segment in candidates:
        missing = sorted(stream for stream, entity in entities.items() if segment not in first[entity])
        if missing:
            logging.warning("Skipping segment %s: no data for observation streams %s", segment, missing)
        else:
            starts[segment] = max((first[entity][segment] for entity in set(entities.values())), default=None)
    if not starts:
        raise ValueError("No segment of this dataset has data for every observation stream in the contract")
    return starts
