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

"""Recorder topic-list dedup.

Regression: a topic referenced in two contract sections (e.g. observation +
reward on one topic) produced two live subscriptions — the second overwrote
the _subs dict entry while the first stayed alive on the graph, so every
message was written to the bag twice — and create_topic registered the topic
twice with the bag writer.
"""

import pytest
from rclpy.qos import QoSProfile, ReliabilityPolicy

from rosetta.contract.schema import parse_contract
from rosetta.robots.ros2.bag_topics import contract_topics
from rosetta.robots.ros2.bag_topics import dedup_topics as _dedup


def test_identical_duplicate_collapses_to_one():
    entry = ("/joint_states", "sensor_msgs/msg/JointState", QoSProfile(depth=10))
    assert _dedup([entry, entry]) == [entry]


def test_distinct_topics_pass_through_in_order():
    a = ("/a", "sensor_msgs/msg/JointState", QoSProfile(depth=10))
    b = ("/b", "std_msgs/msg/Float32", QoSProfile(depth=10))
    assert _dedup([a, b]) == [a, b]


def test_conflicting_types_raise():
    a = ("/t", "sensor_msgs/msg/JointState", QoSProfile(depth=10))
    b = ("/t", "std_msgs/msg/Float32", QoSProfile(depth=10))
    with pytest.raises(ValueError, match="different types"):
        _dedup([a, b])


def test_conflicting_qos_raises():
    a = ("/t", "sensor_msgs/msg/JointState", QoSProfile(depth=10))
    b = (
        "/t",
        "sensor_msgs/msg/JointState",
        QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT),
    )
    with pytest.raises(ValueError, match="conflicting qos"):
        _dedup([a, b])


def test_same_qos_dict_derived_profiles_dedup():
    # Two sections declaring the same qos dict resolve to equal numeric
    # policies and collapse to one entry.
    from rosetta.robots.ros2.rclpy_utils import qos_profile_from_dict

    qos_a = qos_profile_from_dict({"reliability": "best_effort", "depth": 5})
    qos_b = qos_profile_from_dict({"reliability": "best_effort", "depth": 5})
    a = ("/t", "sensor_msgs/msg/JointState", qos_a)
    b = ("/t", "sensor_msgs/msg/JointState", qos_b)
    assert len(_dedup([a, b])) == 1


_CONTRACT = """
robot_type: test
robot_interface: ros2
fps: 30
observations:
  observation.state:
    channel: {topic: /joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.j1]
actions:
  action:
    channel: {topic: /joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.j1]
"""


@pytest.mark.parametrize(
    ("use_sim_time", "expected"),
    [(False, ["/joint_states"]), (True, ["/joint_states", "/clock"])],
)
def test_contract_topics_dedups_and_adds_clock_under_sim_time(use_sim_time, expected):
    topics = contract_topics(parse_contract(_CONTRACT, source="test"), use_sim_time=use_sim_time)
    assert [t[0] for t in topics] == expected
