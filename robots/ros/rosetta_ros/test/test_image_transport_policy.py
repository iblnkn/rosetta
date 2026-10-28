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

"""One-stream-per-camera auto-record policy (image_transport_auto_skips)."""

import pytest

from rosetta_ros.bag_topics import image_transport_auto_skips

IMG = "sensor_msgs/msg/Image"
CIMG = "sensor_msgs/msg/CompressedImage"
PKT = "theora_image_transport/msg/Packet"

RAW = "/cam/image_raw"


_ALL_VARIANTS = ("compressed", "compressedDepth", "theora", "zstd")


def _cam(base: str = RAW, variants: tuple[str, ...] = _ALL_VARIANTS) -> dict[str, str]:
    """A camera family: raw plus the given re-encodings (default: all)."""
    types = {"compressed": CIMG, "compressedDepth": CIMG, "theora": PKT, "zstd": CIMG}
    return {base: IMG, **{f"{base}/{v}": types[v] for v in variants}}


def _kept(graph, recorded=frozenset(), **kw):
    return set(graph) - image_transport_auto_skips(graph, set(recorded), **kw)


def test_prefers_compressed_when_nothing_pinned():
    assert _kept(_cam()) == {f"{RAW}/compressed"}


@pytest.mark.parametrize(
    ("recorded", "include", "expect"),
    [
        ({RAW}, None, {RAW}),  # the contract records raw: the family is covered
        # The contract-declared form is the ONLY member recorded, raw included.
        ({f"{RAW}/compressed"}, None, {f"{RAW}/compressed"}),
        # An adjunct channel lands in the recorded set too, pinning a second form.
        ({f"{RAW}/compressed", RAW}, None, {f"{RAW}/compressed", RAW}),
        (set(), {RAW}, {RAW}),  # include_topics pins raw over the preference
        (set(), {RAW, f"{RAW}/compressed"}, {RAW, f"{RAW}/compressed"}),
    ],
    ids=["contract-raw", "contract-compressed", "adjunct", "include-raw", "include-two"],
)
def test_pinned_members_record_and_the_rest_skip(recorded, include, expect):
    kw = {"include": lambda t: t in include} if include else {}
    assert _kept(_cam(), recorded=recorded, **kw) == expect


@pytest.mark.parametrize(
    ("variants", "expect"),
    [
        ((), {RAW}),  # raw-only camera records raw
        (("zstd", "theora"), {f"{RAW}/zstd"}),  # no compressed: next best wins
        (("theora",), {f"{RAW}/theora"}),
    ],
)
def test_preference_ladder_falls_through(variants, expect):
    assert _kept(_cam(RAW, variants)) == expect


def test_excluded_best_falls_to_next_candidate():
    kept = _kept(_cam(), exclude=lambda t: t.endswith("/compressed"))
    assert kept == {f"{RAW}/zstd"}


def test_fully_excluded_family_records_nothing():
    assert _kept(_cam(), exclude=lambda t: True) == set()


@pytest.mark.parametrize(
    "graph",
    [
        {f"{RAW}/compressed": CIMG},  # no raw Image base advertised
        {"/foo": "std_msgs/msg/String", "/foo/compressed": CIMG},  # base is not an Image
    ],
    ids=["compressed-only", "non-image-base"],
)
def test_without_a_raw_image_base_there_is_no_family(graph):
    # Not this policy's business: nothing is skipped.
    assert image_transport_auto_skips(graph, set()) == set()


def test_families_are_independent():
    graph = {**_cam("/a/image_raw"), **_cam("/b/image_raw")}
    kept = _kept(graph, recorded={"/a/image_raw"})
    assert kept == {"/a/image_raw", "/b/image_raw/compressed"}
