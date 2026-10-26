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

"""Per-episode TRANSIENT_LOCAL subscription QoS depth.

The depth must be at least the live publisher count. /tf_static is
TRANSIENT_LOCAL and commonly has several publishers (robot_state_publisher plus
a calibration static_transform_publisher), each redelivering its latched sample
when the per-episode subscription is created. With the contract's configured
depth (conventionally 1, a DDS matching hint), one redelivered sample evicts
another in the RMW queue before the executor drains it (verified against
rmw_zenoh and cyclonedds).
"""

from rosetta.robots.ros2.bag_topics import MAX_LATCHED_SUB_DEPTH
from rosetta.robots.ros2.bag_topics import latched_sub_depth as _depth


def test_multiple_publishers_widen_beyond_configured_depth():
    # depth=1 (as configured for /tf_static in stone.yaml) with 2 live
    # publishers must retain both, not evict down to 1.
    assert _depth(configured_depth=1, publisher_count=2) == 2


def test_single_publisher_keeps_configured_depth():
    assert _depth(configured_depth=1, publisher_count=1) == 1


def test_configured_depth_wins_when_larger_than_publisher_count():
    assert _depth(configured_depth=5, publisher_count=1) == 5


def test_clamped_to_safety_ceiling():
    assert _depth(configured_depth=1, publisher_count=10_000) == MAX_LATCHED_SUB_DEPTH
