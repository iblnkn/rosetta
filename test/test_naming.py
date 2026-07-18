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

"""Frame-key naming helpers: the semantic camera name and the flatten primitive.

A key's ROLE is deliberately not derivable here — it comes from the spec's
type (KeyLayout.is_action), pinned in test_frame_layout.py. ``camera_name``
yields the semantic (dotted) camera name every adapter shares; only backends
whose sink needs a flat identifier apply ``sanitize_field_name`` on top.
"""

import pytest
from rosetta.frames.naming import camera_name, sanitize_field_name


@pytest.mark.parametrize(
    ("raw", "sanitized"),
    [
        ("observation.state", "observation_state"),  # dots
        ("a-b", "a_b"),  # hyphen
        ("a b", "a_b"),  # space
        ("café", "caf_"),  # non-ASCII
        ("", ""),  # empty passes through
        ("1abc", "1abc"),  # leading digit preserved (pins actual behavior)
        ("already_ok", "already_ok"),  # no-op on clean names
    ],
)
def test_sanitize_field_name(raw, sanitized):
    assert sanitize_field_name(raw) == sanitized


def test_camera_name_preserves_dots():
    """The semantic camera name keeps its dotted hierarchy (LeRobot/GR00T use it as-is)."""
    assert camera_name("observation.images.wrist.right") == "wrist.right"


def test_camera_name_no_prefix_key():
    """A key without the image prefix is returned verbatim (removeprefix no-ops)."""
    assert camera_name("wrist.cam") == "wrist.cam"
