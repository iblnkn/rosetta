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

from collections.abc import Collection
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from rosetta.plugins import instantiate_entry_point

if TYPE_CHECKING:
    import pyarrow as pa

    from rosetta.contract.schema import Channel

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
