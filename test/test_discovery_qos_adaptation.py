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

"""adapt_publisher_qos: subscription QoS for record_all-discovered topics.

Regression: the subscription QoS used to be copied from the FIRST publisher
only. A RELIABLE subscription never matches a BEST_EFFORT publisher, so on a
topic with mixed publishers the incompatible publisher's messages were
silently absent from every bag. Mirror ros2 bag record: request RELIABLE /
TRANSIENT_LOCAL only when every publisher offers it.
"""

from types import SimpleNamespace

from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosetta.robots.ros2.nodes.episode_recorder_node import adapt_publisher_qos


def _pub(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, depth=10):
    return SimpleNamespace(qos_profile=QoSProfile(depth=depth, reliability=reliability, durability=durability))


def test_all_reliable_stays_reliable():
    qos = adapt_publisher_qos([_pub(), _pub()])
    assert qos.reliability == ReliabilityPolicy.RELIABLE


def test_mixed_reliability_downgrades_to_best_effort():
    qos = adapt_publisher_qos([_pub(), _pub(reliability=ReliabilityPolicy.BEST_EFFORT)])
    assert qos.reliability == ReliabilityPolicy.BEST_EFFORT


def test_all_latched_stays_transient_local():
    qos = adapt_publisher_qos([_pub(durability=DurabilityPolicy.TRANSIENT_LOCAL)])
    assert qos.durability == DurabilityPolicy.TRANSIENT_LOCAL


def test_mixed_durability_downgrades_to_volatile():
    qos = adapt_publisher_qos(
        [_pub(durability=DurabilityPolicy.TRANSIENT_LOCAL), _pub(durability=DurabilityPolicy.VOLATILE)]
    )
    assert qos.durability == DurabilityPolicy.VOLATILE


def test_no_publishers_gets_default_profile():
    qos = adapt_publisher_qos([])
    assert qos.depth == 10


def test_depth_takes_widest_publisher():
    qos = adapt_publisher_qos([_pub(depth=5), _pub(depth=25)])
    assert qos.depth == 25


def test_depth_never_below_default():
    qos = adapt_publisher_qos([_pub(depth=1)])
    assert qos.depth == 10
