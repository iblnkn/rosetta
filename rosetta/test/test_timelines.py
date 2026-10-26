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

"""The ros2 interface's timeline registry: attestation and extraction."""

import types

import pytest

from rosetta.robots.ros2.timelines import (
    TIMELINES,
    get_message_timestamp_ns,
    provided_timelines,
    stamp_from_header_ns,
)


def test_provided_timelines_by_type():
    pytest.importorskip("rosidl_runtime_py")
    assert provided_timelines("sensor_msgs/msg/JointState") == {"receive", "header"}
    assert provided_timelines("std_msgs/msg/Float64") == {"receive"}
    with pytest.raises(ValueError, match="NoSuchThing"):
        provided_timelines("sensor_msgs/msg/NoSuchThing")


def test_header_timeline_attests_field_type_not_name():
    # A field merely *named* header must not validate a timeline the
    # extractor can never produce.
    class WrongHeader:
        @staticmethod
        def get_fields_and_field_types():
            return {"header": "uint32"}

    class RealHeader:
        @staticmethod
        def get_fields_and_field_types():
            return {"header": "std_msgs/Header"}

    assert not TIMELINES["header"].provides(WrongHeader)
    assert TIMELINES["header"].provides(RealHeader)


def test_unknown_timeline_raises_instead_of_dropping():
    # Contract loading validates align.timeline, so an unknown name here is
    # a programming error — a loud KeyError, not a silent per-message drop.
    spec = types.SimpleNamespace(source=types.SimpleNamespace(align=types.SimpleNamespace(timeline="publish")))
    with pytest.raises(KeyError, match="publish"):
        get_message_timestamp_ns(object(), spec, 123)


@pytest.mark.parametrize(
    ("sec", "nanosec", "expected"),
    [
        (0, 0, None),  # zero-initialized stamp is indistinguishable from unset
        (0, 1, 1),  # ...but sim time just past t=0 is a real stamp
        (5, 0, 5_000_000_000),
    ],
)
def test_stamp_from_header_ns(sec, nanosec, expected):
    msg = types.SimpleNamespace(header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=sec, nanosec=nanosec)))
    assert stamp_from_header_ns(msg) == expected
