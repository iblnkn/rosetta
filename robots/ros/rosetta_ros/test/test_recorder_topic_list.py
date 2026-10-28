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

"""Recorder topic list: dedup and contract_topics.

A topic referenced in several contract sections (e.g. observation and reward on
one topic) must yield one entry; otherwise the recorder subscribes twice and
writes every message to the bag twice. Conflicting types or QoS for one topic
raise. contract_topics adds /clock when use_sim_time is set.
"""

import pytest
from rclpy.qos import QoSProfile, ReliabilityPolicy

from rosetta.contract.schema import parse_contract
from rosetta_ros.bag_topics import contract_topics
from rosetta_ros.bag_topics import dedup_topics as _dedup


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


def _contract(obs_topic, act_topic, qos=None):
    qos_field = f", qos: {qos}" if qos else ""
    return f"""
robot_type: test
robot_interface: ros2
fps: 30
observations:
  observation.state:
    channel: {{topic: {obs_topic}, type: sensor_msgs/msg/JointState{qos_field}}}
    align: {{strategy: hold, timeline: receive}}
    select: [position.j1]
actions:
  action:
    channel: {{topic: {act_topic}, type: sensor_msgs/msg/JointState{qos_field}}}
    align: {{strategy: hold, timeline: receive}}
    select: [position.j1]
"""


@pytest.mark.parametrize(
    ("contract", "use_sim_time", "expected"),
    [
        (_contract("/joint_states", "/joint_states"), False, ["/joint_states"]),
        (_contract("/joint_states", "/joint_states"), True, ["/joint_states", "/clock"]),
        (_contract("/a", "/b"), False, ["/a", "/b"]),
        # Two sections declaring the same qos dict resolve to equal profiles.
        (_contract("/t", "/t", qos="{reliability: best_effort, depth: 5}"), False, ["/t"]),
    ],
    ids=["shared-topic", "sim-time-adds-clock", "distinct-in-order", "same-qos-dict"],
)
def test_contract_topics_dedups_and_adds_clock_under_sim_time(contract, use_sim_time, expected):
    topics = contract_topics(parse_contract(contract, source="test"), use_sim_time=use_sim_time)
    assert [t[0] for t in topics] == expected
