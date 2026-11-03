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

"""The column decoder bridging Rerun's dataloader to rosetta's decode path.

One decoder per contract stream. Each recorded message, as the robot
interface's layout presents it, goes through ``decode_value(msg, spec)``, so
the codecs the contract's robot interface registered run exactly as they do in
the porter and the live bridge. The server's latest-at fill is the contract's
``hold`` strategy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from rerun.experimental.dataloader import ColumnDecoder

from rosetta.frames.codecs import decode_value

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pyarrow as pa
    from rerun.experimental.dataloader import DecodeRequest, FieldBatch

    from rosetta.contract.specs import StreamSpec


class ContractDecoder(ColumnDecoder[Any]):
    """Decode one contract stream's column into frame values.

    An unwindowed field decodes to one value per sample; a windowed field to a
    list with one value per window offset. Values are ``np.ndarray`` (or
    ``str`` for string codecs), or ``None`` where no message existed yet, which
    ``FrameLayout.assemble`` zero-fills exactly as the porter does.

    ``message`` turns one recorded cell into the input the decoders take (the
    layout's :meth:`~.layout.RecordingLayout.message`).
    """

    def __init__(self, spec: StreamSpec, message: Callable[[pa.Scalar], Any]) -> None:
        self._spec = spec
        self._message = message

    def decode(self, batch: FieldBatch, requests: Sequence[DecodeRequest]) -> list[Any]:
        # Decode each recorded message once per block. Overlapping windows share
        # rows, and the server's latest-at fill gives each sample its own copy of
        # a held message, so a row equal to the one before it reuses its value:
        # decode_value is a pure function of the message and the spec.
        # Fields set no selector, so a request's output rows are the batch's rows.
        assert batch.select is None
        decoded: dict[int, Any] = {}
        previous: tuple[pa.Scalar, Any] | None = None

        def value(row: int) -> Any:
            nonlocal previous
            if row in decoded:
                return _unshared(decoded[row])
            cell = batch.rows(row, row + 1)[0]
            if previous is not None and cell.equals(previous[0]):
                decoded[row] = _unshared(previous[1])
            else:
                decoded[row] = self._decode_cell(cell)
            previous = (cell, decoded[row])
            return decoded[row]

        out: list[Any] = []
        for request in requests:
            values = [value(row) for row in request.output_row_indices]
            if batch.is_windowed:
                out.append(values)
            else:
                (single,) = values
                out.append(single)
        return out

    def _decode_cell(self, row: pa.Scalar) -> Any:
        # Component columns are List<...> with one cell per logged row.
        if not row.is_valid or not len(row.values):
            return None
        return decode_value(self._message(row.values[0]), self._spec)

    def __repr__(self) -> str:
        return f"ContractDecoder({self._spec.key!r}, {self._spec.source.channel.topic!r})"


def _unshared(value: Any) -> Any:
    """A copy of a reused array, so no two samples share one buffer."""
    return value.copy() if isinstance(value, np.ndarray) else value
