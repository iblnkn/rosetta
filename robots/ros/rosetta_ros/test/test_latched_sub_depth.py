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

import pytest

from rosetta_ros.bag_topics import MAX_LATCHED_SUB_DEPTH, latched_sub_depth


@pytest.mark.parametrize(
    ("configured_depth", "publisher_count", "expected"),
    [
        (1, 2, 2),  # /tf_static in stone.yaml: two publishers must both be retained
        (1, 1, 1),  # single publisher keeps the configured depth
        (5, 1, 5),  # configured depth wins when larger
        (1, 10_000, MAX_LATCHED_SUB_DEPTH),  # clamped to the safety ceiling
    ],
)
def test_latched_sub_depth(configured_depth, publisher_count, expected):
    assert latched_sub_depth(configured_depth=configured_depth, publisher_count=publisher_count) == expected
