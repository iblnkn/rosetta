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


"""ContractDecoder decodes each recorded message once per fetch block."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

pytest.importorskip("rerun")

from rerun.experimental.dataloader import DecodeRequest, FieldBatch
from rerun_rosetta import decoders


@pytest.fixture
def decode_calls(monkeypatch):
    calls = []

    def fake_decode_value(msg, spec):
        calls.append(msg)
        return np.full(3, msg, dtype=np.uint8)

    monkeypatch.setattr(decoders, "decode_value", fake_decode_value)
    return calls


def _batch(messages: list[int], *, windowed: bool = False) -> FieldBatch:
    # A Rerun component column: one list cell per row, holding one struct.
    cells = [[{"data": bytes([m]) * 8}] for m in messages]
    return FieldBatch(column=pa.array(cells), is_windowed=windowed)


def _request(position: int, rows: tuple[int, ...], *, segment: str = "s", anchor: int | None = None) -> DecodeRequest:
    return DecodeRequest(
        sample_position=position,
        segment_id=segment,
        index_value=position if anchor is None else anchor,
        decode_row_indices=rows,
        output_row_indices=rows,
        starts_at_keyframe=True,
    )


def _decoder(window_ns=(0,), **kwargs):
    spec = SimpleNamespace(key="k", source=SimpleNamespace(channel=SimpleNamespace(topic="/t")))
    return decoders.ContractDecoder(spec, lambda cell: cell["data"].as_py()[0], window_ns=window_ns, **kwargs)


def test_held_message_decodes_once(decode_calls):
    # The latest-at fill repeats a held message on consecutive rows: 1 1 2 2 2 1.
    messages = [1, 1, 2, 2, 2, 1]
    out = _decoder().decode(_batch(messages), [_request(i, (i,)) for i in range(len(messages))])
    assert decode_calls == [1, 2, 1]
    assert [int(v[0]) for v in out] == messages


def test_reused_values_are_not_shared(decode_calls):
    out = _decoder().decode(_batch([7, 7, 7]), [_request(i, (i,)) for i in range(3)])
    assert len({id(v) for v in out}) == 3
    out[0][:] = 0
    assert int(out[1][0]) == 7


def test_overlapping_windows_share_rows(decode_calls):
    out = _decoder((0, 1, 2)).decode(
        _batch([1, 2, 3, 4], windowed=True), [_request(0, (0, 1, 2)), _request(1, (1, 2, 3))]
    )
    assert decode_calls == [1, 2, 3, 4]
    assert [[int(v[0]) for v in w] for w in out] == [[1, 2, 3], [2, 3, 4]]
    assert out[0][1] is not out[1][0]


def test_window_points_are_decoded_once_across_fetch_blocks(decode_calls):
    """Shuffled samples land in different blocks; a point another sample decoded is reused."""
    decoder = _decoder((0, 1, 2))
    first = decoder.decode(_batch([1, 2, 3], windowed=True), [_request(0, (0, 1, 2), anchor=10)])
    # A later block, rows in a different order: anchor 11 shares points 11 and 12.
    second = decoder.decode(_batch([9, 2, 3, 4], windowed=True), [_request(0, (1, 2, 3), anchor=11)])
    assert decode_calls == [1, 2, 3, 4]
    assert [int(v[0]) for v in second[0]] == [2, 3, 4]
    assert second[0][0] is not first[0][1]


def test_segments_never_share_points(decode_calls):
    decoder = _decoder((0, 1))
    decoder.decode(_batch([1, 2], windowed=True), [_request(0, (0, 1), segment="a", anchor=5)])
    out = decoder.decode(_batch([3, 4], windowed=True), [_request(0, (0, 1), segment="b", anchor=5)])
    assert decode_calls == [1, 2, 3, 4]
    assert [int(v[0]) for v in out[0]] == [3, 4]


def test_cache_stays_within_its_byte_budget(decode_calls):
    decoder = _decoder((0, 1), cache_bytes=6)  # two 3-byte values
    for anchor in range(0, 8, 2):  # windows (0, 1), (2, 3), ... share no points
        decoder.decode(_batch([anchor + 1, anchor + 2], windowed=True), [_request(0, (0, 1), anchor=anchor)])
    assert decoder._cached_bytes <= 6
    assert len(decoder._cache) == 2
    decoder.decode(_batch([1, 2], windowed=True), [_request(0, (0, 1), anchor=0)])  # evicted: decoded again
    assert decode_calls == [1, 2, 3, 4, 5, 6, 7, 8, 1, 2]


def test_a_miss_shares_no_buffer_with_the_cache(decode_calls):
    decoder = _decoder((0, 1))
    first = decoder.decode(_batch([1, 2], windowed=True), [_request(0, (0, 1), anchor=0)])
    first[0][1][:] = 0  # a sample transformed in place
    second = decoder.decode(_batch([2, 3], windowed=True), [_request(0, (0, 1), anchor=1)])
    assert decode_calls == [1, 2, 3]  # point 1 came from the cache
    assert int(second[0][0][0]) == 2


def test_an_unwindowed_field_caches_nothing(decode_calls):
    decoder = _decoder()
    decoder.decode(_batch([1]), [_request(0, (0,), anchor=0)])
    assert not decoder._cache


def test_one_tick_queried_a_nanosecond_apart_is_one_key(decode_calls):
    """At 30 fps the dataloader queries tick 2 as anchor + 66_666_667 and, from the next anchor, + 66_666_666."""
    step = 33_333_333
    decoder = _decoder((0, round(1 / 30 * 1e9), round(2 / 30 * 1e9)), step_ns=step)
    decoder.decode(_batch([1, 2, 3], windowed=True), [_request(0, (0, 1, 2), anchor=0)])
    decoder.decode(_batch([2, 3, 4], windowed=True), [_request(0, (0, 1, 2), anchor=step)])
    assert decode_calls == [1, 2, 3, 4]


def test_a_grid_offset_by_half_a_step_keeps_ticks_apart(decode_calls):
    """Anchors needn't sit on multiples of the step from zero (here they sit on half steps)."""
    step = 100
    decoder = _decoder((0, step), step_ns=step)
    for k in range(0, 8, 2):
        decoder.decode(_batch([k + 1, k + 2], windowed=True), [_request(0, (0, 1), anchor=50 + k * step)])
    assert decode_calls == [1, 2, 3, 4, 5, 6, 7, 8]
