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

"""Indexed-selector resolution (teleop event selectors). Pure, no ROS."""

import types

import pytest
from rosetta.robots.ros2.field_access import resolve_indexed


def _joy(buttons=(0, 1, 0), axes=(0.5, -0.5)):
    return types.SimpleNamespace(buttons=list(buttons), axes=list(axes))


def test_numeric_segment_indexes_sequence():
    assert resolve_indexed(_joy(), "buttons.1") == 1
    assert resolve_indexed(_joy(), "axes.0") == 0.5


def test_plain_segments_walk_attributes():
    msg = types.SimpleNamespace(pose=types.SimpleNamespace(x=3.0))
    assert resolve_indexed(msg, "pose.x") == 3.0


def test_negative_index_rejected():
    # Wrap-from-the-end would silently read the wrong element.
    with pytest.raises(ValueError, match="non-negative"):
        resolve_indexed(_joy(), "buttons.-1")


def test_out_of_range_index_raises():
    with pytest.raises(IndexError):
        resolve_indexed(_joy(), "buttons.9")
