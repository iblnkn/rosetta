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

from collections import OrderedDict
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


DEFAULT_CACHE_BYTES = 128 * 1024 * 1024
"""Decoded values a windowed :class:`ContractDecoder` keeps, per stream and per DataLoader worker."""

_MISSING = object()


class ContractDecoder(ColumnDecoder[Any]):
    """Decode one contract stream's column into frame values.

    An unwindowed field decodes to one value per sample; a windowed field to a
    list with one value per window offset. Values are ``np.ndarray`` (or
    ``str`` for string codecs), or ``None`` where no message existed yet, which
    ``FrameLayout.assemble`` zero-fills exactly as the porter does.

    ``message`` turns one recorded cell into the input the decoders take (the
    layout's :meth:`~.layout.RecordingLayout.message`).

    Decoded values are cached by the grid tick they were sampled at,
    ``(segment, anchor + whole steps)``, up to ``cache_bytes`` per decoder. The server's
    latest-at fill makes a stream's value at a tick the same for every sample
    that reaches it, so a 50-step action window, whose ticks the next 49 samples
    share, is decoded once per tick instead of once per sample. ``window_ns``
    are the field's window offsets in index units, as the dataloader computes
    them, and ``step_ns`` is the grid period. Anchors in a segment are whole
    steps apart, and :func:`~.query.compile_contract` gives offsets in whole
    steps, so the key is the exact point each sample queries.

    Only a windowed field caches. An unwindowed field queries each tick once
    per epoch, so a cached value would never be read again.
    """

    def __init__(
        self,
        spec: StreamSpec,
        message: Callable[[pa.Scalar], Any],
        *,
        window_ns: tuple[int, ...] = (0,),
        step_ns: int = 1,
        cache_bytes: int = DEFAULT_CACHE_BYTES,
    ) -> None:
        self._spec = spec
        self._message = message
        self._window_ns = window_ns
        self._step_ns = step_ns
        self._cache_bytes = cache_bytes if len(window_ns) > 1 else 0
        self._cache: OrderedDict[tuple[str, int], Any] = OrderedDict()
        self._cached_bytes = 0

    def decode(self, batch: FieldBatch, requests: Sequence[DecodeRequest]) -> list[Any]:
        # Decode each recorded message once per block. Overlapping windows share
        # rows, and the server's latest-at fill gives each sample its own copy of
        # a held message, so a row equal to the one before it reuses its value:
        # decode_value is a pure function of the message and the spec.
        # Fields set no selector, so a request's output rows are the batch's rows.
        # Decoded and cached values are never handed out: every sample gets its
        # own copy, so no two samples, and no sample and the cache, share a buffer.
        assert batch.select is None
        decoded: dict[int, Any] = {}
        previous: tuple[pa.Scalar, Any] | None = None

        def value(row: int) -> Any:
            nonlocal previous
            if row not in decoded:
                cell = batch.rows(row, row + 1)[0]
                if previous is not None and cell.equals(previous[0]):
                    decoded[row] = previous[1]
                else:
                    decoded[row] = self._decode_cell(cell)
                previous = (cell, decoded[row])
            return decoded[row]

        out: list[Any] = []
        for request in requests:
            anchor = index_ns(request.index_value)
            values = []
            for row, offset in zip(request.output_row_indices, self._window_ns, strict=True):
                key = (request.segment_id, anchor + round(offset / self._step_ns) * self._step_ns)
                cached = self._cache.get(key, _MISSING)
                if cached is _MISSING:
                    cached = value(row)
                    self._remember(key, cached)
                else:
                    self._cache.move_to_end(key)
                values.append(_unshared(cached))
            if batch.is_windowed:
                out.append(values)
            else:
                (single,) = values
                out.append(single)
        return out

    def _remember(self, key: tuple[str, int], value: Any) -> None:
        size = _nbytes(value)
        if size > self._cache_bytes:
            return
        self._cache[key] = value
        self._cached_bytes += size
        while self._cached_bytes > self._cache_bytes:
            _, evicted = self._cache.popitem(last=False)
            self._cached_bytes -= _nbytes(evicted)

    def __getstate__(self) -> dict[str, Any]:
        # DataLoader workers each start with an empty cache.
        state = self.__dict__.copy()
        state["_cache"], state["_cached_bytes"] = OrderedDict(), 0
        return state

    def _decode_cell(self, row: pa.Scalar) -> Any:
        # Component columns are List<...> with one cell per logged row.
        if not row.is_valid or not len(row.values):
            return None
        return decode_value(self._message(row.values[0]), self._spec)

    def __repr__(self) -> str:
        return f"ContractDecoder({self._spec.key!r}, {self._spec.source.channel.topic!r})"


def _nbytes(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, str):
        return len(value)
    return 64


def _unshared(value: Any) -> Any:
    """A copy of an array, so no two samples share one buffer."""
    return value.copy() if isinstance(value, np.ndarray) else value


class PositionDecoder(ColumnDecoder[Any]):
    """Where each sample sits: ``(segment_id, index_ns)``, read off the request, not the data.

    Rerun's samples carry no segment or index value. A field decoded by this
    decoder, on a column every sample has, puts them back, so a reader can
    number episodes and frames.
    """

    def decode(self, batch: FieldBatch, requests: Sequence[DecodeRequest]) -> list[Any]:
        return [(request.segment_id, index_ns(request.index_value)) for request in requests]

    def __repr__(self) -> str:
        return "PositionDecoder()"


def index_ns(value: Any) -> int:
    """An index value (int, ``datetime64`` or ``timedelta64``) in integer nanoseconds."""
    if isinstance(value, np.datetime64):
        return int(value.astype("datetime64[ns]").astype(np.int64))
    if isinstance(value, np.timedelta64):
        return int(value.astype("timedelta64[ns]").astype(np.int64))
    return int(value)
