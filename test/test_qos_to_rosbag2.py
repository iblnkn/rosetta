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

"""qos_to_rosbag2: rclpy QoSProfile -> rosbag2_py QoS for bag topic metadata.

Regression: "infinite"/unset durations discovered from real publishers come
back as RMW_DURATION_INFINITE (int64 nanoseconds max), which overflows
rosbag2_py Duration's int32-seconds/uint32-nanoseconds constructor. The
converter must clamp. rosbag2_py QoS exposes no getters (setter-only pybind
API), so these tests pin construction behavior: clamped values build,
unclamped infinite values provably do not.
"""

import pytest
from rclpy.duration import Duration
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rosbag2_py._storage import Duration as Rosbag2Duration
from rosbag2_py._storage import QoS as Rosbag2QoS

from rosetta.robots.ros2.nodes.node_utils import _rosbag2_duration, qos_to_rosbag2

RMW_DURATION_INFINITE_NS = 2**63 - 1


def test_unclamped_infinite_duration_overflows_rosbag2():
    # The failure mode the clamp exists for: without it, an infinite RMW
    # duration cannot even construct a rosbag2 Duration.
    with pytest.raises(TypeError):
        Rosbag2Duration(RMW_DURATION_INFINITE_NS // 1_000_000_000, 0)


def test_infinite_duration_clamps_and_constructs():
    assert isinstance(_rosbag2_duration(Duration(nanoseconds=RMW_DURATION_INFINITE_NS)), Rosbag2Duration)


def test_finite_duration_constructs():
    assert isinstance(_rosbag2_duration(Duration(seconds=1, nanoseconds=500)), Rosbag2Duration)


def test_profile_with_infinite_durations_converts():
    # What get_publishers_info_by_topic reports for a publisher that never
    # set deadline/lifespan/lease.
    infinite = Duration(nanoseconds=RMW_DURATION_INFINITE_NS)
    profile = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        deadline=infinite,
        lifespan=infinite,
        liveliness_lease_duration=infinite,
    )
    assert isinstance(qos_to_rosbag2(profile), Rosbag2QoS)


def test_default_profile_converts():
    assert isinstance(qos_to_rosbag2(QoSProfile(depth=10)), Rosbag2QoS)
