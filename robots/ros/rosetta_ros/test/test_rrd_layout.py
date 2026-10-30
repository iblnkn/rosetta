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

"""The ros2 rrd layout: where a channel's messages are, and the duck-typed view decoders read."""

from __future__ import annotations

import numpy as np
import pytest

from rosetta.contract.model import Channel

pa = pytest.importorskip("pyarrow")

from rosetta_ros.offline.rrd_layout import RAW_COMPONENT, MessageView, Rosbag2RrdLayout  # noqa: E402

JS = Channel(topic="/js", type="sensor_msgs/msg/JointState")


def _view(row: dict, row_type: pa.DataType | None = None) -> MessageView:
    return MessageView(pa.array([row], type=row_type)[0])


def test_joint_state_fields():
    msg = _view(
        {
            "header": {"stamp": {"sec": 5, "nanosec": 100}, "frame_id": "base"},
            "name": ["a", "b"],
            "position": [0.1, 0.2],
            "velocity": [],
            "effort": [],
        }
    )
    assert msg.name == ["a", "b"]
    np.testing.assert_allclose(msg.position, [0.1, 0.2])
    assert len(msg.velocity) == 0
    assert msg.header.stamp.sec == 5
    assert msg.header.frame_id == "base"


def test_nested_structs_and_fixed_arrays():
    msg = _view(
        {
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "orientation_covariance": [0.0] * 9,
            "angular_velocity": {"x": 1.5, "y": 0.0, "z": -2.0},
        }
    )
    assert msg.angular_velocity.z == -2.0
    assert msg.orientation.w == 1.0
    assert len(msg.orientation_covariance) == 9


def test_uint8_blob_is_a_buffer():
    row_type = pa.struct([("format", pa.string()), ("data", pa.list_(pa.uint8()))])
    msg = _view({"format": "jpeg", "data": list(range(64))}, row_type)
    assert msg.format == "jpeg"
    assert np.frombuffer(msg.data, dtype=np.uint8).tobytes() == bytes(range(64))


def test_sequence_of_structs():
    msg = _view({"points": [{"positions": [1.0, 2.0]}, {"positions": [3.0]}]})
    assert len(msg.points) == 2
    np.testing.assert_allclose(msg.points[0].positions, [1.0, 2.0])


def test_unknown_field_raises():
    msg = _view({"data": "x"})
    with pytest.raises(AttributeError, match="no field 'nope'"):
        msg.nope


def test_non_struct_rejected():
    with pytest.raises(TypeError, match="struct row"):
        MessageView(pa.scalar(1.0))


def test_reflected_column_is_found():
    column = Rosbag2RrdLayout().column(JS, ["/js:sensor_msgs.msg.JointState:message"])
    assert column == "/js:sensor_msgs.msg.JointState:message"


def test_unrecorded_topic_has_no_column():
    assert Rosbag2RrdLayout().column(JS, ["/other:sensor_msgs.msg.JointState:message"]) is None


def test_raw_only_topic_raises_with_the_fix():
    with pytest.raises(ValueError, match="ros2 bag convert"):
        Rosbag2RrdLayout().column(JS, [f"/js:{RAW_COMPONENT}"])


def test_receive_time_is_on_recv_timestamp():
    layout = Rosbag2RrdLayout()
    assert (layout.receive_timeline, layout.index) == ("receive", "recv_timestamp")
