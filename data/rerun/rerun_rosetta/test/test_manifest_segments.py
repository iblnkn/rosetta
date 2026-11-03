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

"""A saved manifest is refused once the dataset's segments change."""

import pytest

pytest.importorskip("rerun")

import pyarrow as pa
import pyarrow.parquet as pq
from rerun_rosetta.dataset import _record_segments, _saved_segments, _segment_change


@pytest.mark.parametrize(
    ("saved", "current", "named"),
    [
        (["ep0", "ep1"], ["ep0", "ep1", "ep2"], "1 added: ['ep2']"),  # new recordings registered
        (["ep0", "ep1"], ["ep0"], "1 removed: ['ep1']"),  # a narrower `segments`
        (None, ["ep0"], "saved before"),  # a manifest from before segments were recorded
    ],
)
def test_a_manifest_for_other_segments_is_refused(saved, current, named):
    assert named in _segment_change(saved, current)


def test_a_manifest_for_the_same_segments_is_accepted():
    assert _segment_change(["ep1", "ep0"], ["ep0", "ep1"]) is None


def test_recording_segments_keeps_the_manifests_own_header(tmp_path):
    # Rerun keeps its manifest header in the same schema metadata. Losing it
    # would break Manifest.from_parquet on every saved manifest.
    path = tmp_path / "manifest.parquet"
    table = pa.table({"segment_id": ["ep0", "ep0", "ep1"]})
    pq.write_table(table.replace_schema_metadata({b"rerun.header": b"kept"}), path)

    _record_segments(path, ["ep0", "ep1"])

    assert pq.read_schema(path).metadata[b"rerun.header"] == b"kept"
    assert _saved_segments(path) == ["ep0", "ep1"]
    assert pq.read_table(path).column("segment_id").to_pylist() == ["ep0", "ep0", "ep1"]
