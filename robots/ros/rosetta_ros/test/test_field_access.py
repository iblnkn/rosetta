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

"""Field selectors: indexed resolution (teleop events) and the Joy / parallel-array
grammars, the latter through the decoders that parse them. No rclpy: messages are duck-typed."""

import types

import numpy as np
import pytest

from rosetta.frames.codecs import DECODERS, discover_codecs
from rosetta_ros.field_access import resolve_indexed


def _joy(buttons=(0, 1, 0), axes=(0.5, -0.5)):
    return types.SimpleNamespace(buttons=list(buttons), axes=list(axes))


def _decode(msg_type, msg, names):
    discover_codecs()
    # The decoders read only the selector list off the spec.
    return DECODERS[msg_type](msg, types.SimpleNamespace(names=names))


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
    [("axes.0", 0.5), ("buttons.1", 1.0), ("1", -0.5)],  # a bare index means axes
)
def test_joy_selector_reads_its_field(selector, expected):
    np.testing.assert_array_equal(_decode("sensor_msgs/msg/Joy", _joy(), [selector]), [expected])


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
        _decode("sensor_msgs/msg/Joy", _joy(), [selector])


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ("velocity.elbow", 2.0),  # singular alias
        ("velocities.elbow", 2.0),  # plural spelling
        ("elbow", 1.0),  # bare name -> default field (position)
    ],
)
def test_trajectory_selector_aliases(selector, expected):
    point = types.SimpleNamespace(positions=[1.0], velocities=[2.0], accelerations=[], effort=[])
    msg = types.SimpleNamespace(joint_names=["elbow"], points=[point])
    np.testing.assert_array_equal(_decode("trajectory_msgs/msg/JointTrajectory", msg, [selector]), [expected])
