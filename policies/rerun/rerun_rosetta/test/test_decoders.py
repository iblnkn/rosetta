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


def _request(position: int, rows: tuple[int, ...]) -> DecodeRequest:
    return DecodeRequest(
        sample_position=position,
        segment_id="s",
        index_value=position,
        decode_row_indices=rows,
        output_row_indices=rows,
        starts_at_keyframe=True,
    )


def _decoder():
    spec = SimpleNamespace(key="k", source=SimpleNamespace(channel=SimpleNamespace(topic="/t")))
    return decoders.ContractDecoder(spec, lambda cell: cell["data"].as_py()[0])


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
    out = _decoder().decode(_batch([1, 2, 3, 4], windowed=True), [_request(0, (0, 1, 2)), _request(1, (1, 2, 3))])
    assert decode_calls == [1, 2, 3, 4]
    assert [[int(v[0]) for v in w] for w in out] == [[1, 2, 3], [2, 3, 4]]
    assert out[0][1] is not out[1][0]
