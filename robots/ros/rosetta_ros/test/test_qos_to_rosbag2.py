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

Infinite or unset durations discovered from real publishers come back as
RMW_DURATION_INFINITE (int64 nanoseconds max), which overflows rosbag2_py
Duration's int32-seconds/uint32-nanoseconds constructor (it raises TypeError),
so the converter must clamp. rosbag2_py QoS exposes no getters (setter-only
pybind API), so these tests pin that conversion constructs without raising.
"""

from rclpy.duration import Duration
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rosbag2_py._storage import QoS as Rosbag2QoS

from rosetta_ros.bag_topics import qos_to_rosbag2

RMW_DURATION_INFINITE_NS = 2**63 - 1


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
