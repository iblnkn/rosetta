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

from rosetta.robots.ros2.nodes.episode_recorder_node import image_transport_auto_skips

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


def test_contract_raw_covers_family():
    assert _kept(_cam(), recorded={RAW}) == {RAW}


def test_contract_compressed_covers_family():
    # The contract-declared form is the ONLY family member recorded — raw
    # included. Adjunct or include_topics are the ways to pin a second form.
    assert _kept(_cam(), recorded={f"{RAW}/compressed"}) == {f"{RAW}/compressed"}


def test_adjunct_pins_a_second_stream():
    # Contract observation records /compressed; adjunct additionally declares
    # raw (adjunct channels land in the recorded set): both record.
    kept = _kept(_cam(), recorded={f"{RAW}/compressed", RAW})
    assert kept == {f"{RAW}/compressed", RAW}


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


def test_raw_recorded_unless_blacklisted():
    # Only raw advertised and it is excluded: nothing to record.
    assert _kept(_cam(RAW, ()), exclude=lambda t: t == RAW) == set()


def test_include_pins_raw_over_preference():
    kept = _kept(_cam(), include=lambda t: t == RAW)
    assert kept == {RAW}


def test_include_pins_multiple_members():
    kept = _kept(_cam(), include=lambda t: t in (RAW, f"{RAW}/compressed"))
    assert kept == {RAW, f"{RAW}/compressed"}


def test_compressed_only_camera_is_not_a_family():
    # No raw Image base advertised: not this policy's business.
    graph = {f"{RAW}/compressed": CIMG}
    assert image_transport_auto_skips(graph, set()) == set()


def test_non_image_base_is_not_a_family():
    graph = {"/foo": "std_msgs/msg/String", "/foo/compressed": CIMG}
    assert image_transport_auto_skips(graph, set()) == set()


def test_families_are_independent():
    graph = {**_cam("/a/image_raw"), **_cam("/b/image_raw")}
    kept = _kept(graph, recorded={"/a/image_raw"})
    assert kept == {"/a/image_raw", "/b/image_raw/compressed"}
