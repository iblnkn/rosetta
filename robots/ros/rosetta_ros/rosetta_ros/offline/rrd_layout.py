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

"""How a ros2 recording made with rosbag2_storage_rrd is laid out in Rerun.

Registered under ``rerun_rosetta.layouts`` as ``ros2``, so rerun_rosetta can
train on ``storage_id: rrd`` bags without knowing ROS. The plugin writes one
entity per topic. A message's ``reflected`` form, an Arrow struct whose field
names mirror the message definition, sits under ``<pkg.msg.Type>:message``,
and its receive time on the ``recv_timestamp`` timeline.

The decoders this interface registers are duck-typed on the message: they read
attributes, index sequences and take buffers. So a :class:`MessageView` over
the reflected struct is all they need, and nothing here imports ROS.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

    from rosetta.contract.schema import Channel

__all__ = [
    "MessageView",
    "Rosbag2RrdLayout",
]

# The plugin's raw form: the CDR bytes under one archetype for every type.
RAW_COMPONENT = "rosbag2.SerializedBagMessage:serialized_data"


class Rosbag2RrdLayout:
    """``RecordingLayout`` for ros2 recordings made with rosbag2_storage_rrd."""

    receive_timeline = "receive"
    index = "recv_timestamp"

    def column(self, channel: Channel, columns: Collection[str]) -> str | None:
        """The channel's reflected column, or None when the topic wasn't recorded.

        Raises:
            ValueError: If the topic was recorded only in the plugin's raw form.

        """
        reflected = f"{channel.topic}:{channel.type.replace('/', '.')}:message"
        if reflected in columns:
            return reflected
        if f"{channel.topic}:{RAW_COMPONENT}" in columns:
            raise ValueError(
                f"Topic {channel.topic} was recorded only in raw form, and training reads the reflected form. "
                f"Add it with `ros2 bag convert` and a storage config of `representations: [reflected]` "
                f"(see rosbag2_storage_rrd's README, 'Adding representations after recording')."
            )
        return None

    def message(self, cell: pa.Scalar) -> MessageView:
        return MessageView(cell)


class MessageView:
    """Read-only attribute access to one reflected message.

    Fields convert on first access, so a decoder pays only for the fields it
    reads. Nested structs are views, primitive sequences are numpy arrays
    (read-only, zero-copy where Arrow allows), sequences of structs are lists
    of views, and everything else is the plain Python value. A field the struct
    doesn't have raises ``AttributeError``, as it would on the real message.
    """

    __slots__ = ("_fields", "_row")

    def __init__(self, row: pa.StructScalar) -> None:
        import pyarrow as pa  # imported here so rosetta_ros loads without Rerun's dependencies

        if not pa.types.is_struct(row.type):
            raise TypeError(f"MessageView needs a struct row, got Arrow type {row.type}")
        self._row = row
        self._fields: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._fields[name]
        except KeyError:
            pass
        if self._row.type.get_field_index(name) < 0:
            raise AttributeError(f"message has no field '{name}'")
        value = self._fields[name] = _convert(self._row[name])
        return value

    def __repr__(self) -> str:
        return f"MessageView({', '.join(field.name for field in self._row.type)})"


def _convert(value: pa.Scalar) -> Any:
    import pyarrow as pa

    if not value.is_valid:
        return None
    if pa.types.is_struct(value.type):
        return MessageView(value)
    if pa.types.is_list(value.type) or pa.types.is_large_list(value.type) or pa.types.is_fixed_size_list(value.type):
        items = value.values
        if pa.types.is_struct(items.type) or pa.types.is_list(items.type) or pa.types.is_large_list(items.type):
            return [_convert(items[i]) for i in range(len(items))]
        if pa.types.is_string(items.type) or pa.types.is_large_string(items.type):
            return items.to_pylist()
        return items.to_numpy(zero_copy_only=False)
    return value.as_py()
