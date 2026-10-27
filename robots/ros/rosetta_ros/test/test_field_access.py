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

"""Field selectors: indexed resolution (teleop events) and the Joy / parallel-array grammars. Pure, no ROS."""

import types

import pytest

from rosetta_ros.field_access import (
    JOINT_TRAJECTORY_FIELDS,
    parse_field_selector,
    parse_joy_selector,
    resolve_indexed,
)


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


@pytest.mark.parametrize(
    ("selector", "expected"),
    [("axes.0", ("axes", 0)), ("buttons.3", ("buttons", 3)), ("2", ("axes", 2))],
)
def test_joy_selector_parses(selector, expected):
    assert parse_joy_selector(selector) == expected


@pytest.mark.parametrize(
    ("selector", "match"),
    [
        ("axes.-1", "non-negative"),  # Python wraparound would silently read the wrong element
        ("axes.x", "must be an integer"),
        ("triggers.0", "Unknown Joy field"),
    ],
)
def test_joy_selector_rejects_bad_input(selector, match):
    with pytest.raises(ValueError, match=match):
        parse_joy_selector(selector)


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ("velocity.elbow", ("velocities", "elbow")),  # singular alias
        ("velocities.elbow", ("velocities", "elbow")),  # plural spelling
        ("elbow", ("positions", "elbow")),  # bare name -> default field
    ],
)
def test_field_selector_parses_trajectory_aliases(selector, expected):
    parsed = parse_field_selector(
        selector, JOINT_TRAJECTORY_FIELDS, default="position", msg_type="JointTrajectoryPoint"
    )
    assert parsed == expected
