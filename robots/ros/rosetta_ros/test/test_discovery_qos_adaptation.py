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

Like ros2 bag record, request RELIABLE / TRANSIENT_LOCAL only when every
publisher offers it. A RELIABLE subscription never matches a BEST_EFFORT
publisher, so copying one publisher's QoS on a mixed topic silently drops the
other publisher's messages from the bag.
"""

from types import SimpleNamespace

import pytest
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from rosetta_ros.bag_topics import adapt_publisher_qos


def _pub(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, depth=10):
    return SimpleNamespace(qos_profile=QoSProfile(depth=depth, reliability=reliability, durability=durability))


_RELIABLE = _pub()
_BEST_EFFORT = _pub(reliability=ReliabilityPolicy.BEST_EFFORT)
_LATCHED = _pub(durability=DurabilityPolicy.TRANSIENT_LOCAL)
_VOLATILE = _pub()


@pytest.mark.parametrize(
    ("pubs", "field", "expected"),
    [
        ([_RELIABLE, _RELIABLE], "reliability", ReliabilityPolicy.RELIABLE),
        ([_RELIABLE, _BEST_EFFORT], "reliability", ReliabilityPolicy.BEST_EFFORT),
        ([_LATCHED], "durability", DurabilityPolicy.TRANSIENT_LOCAL),
        ([_LATCHED, _VOLATILE], "durability", DurabilityPolicy.VOLATILE),
    ],
    ids=["all-reliable", "mixed-reliability", "all-latched", "mixed-durability"],
)
def test_policy_is_requested_only_when_every_publisher_offers_it(pubs, field, expected):
    assert getattr(adapt_publisher_qos(pubs), field) == expected


@pytest.mark.parametrize(
    ("depths", "expected"),
    [([], 10), ([5, 25], 25), ([1], 10)],
    ids=["no-publishers", "widest-publisher", "never-below-default"],
)
def test_depth_is_the_widest_publisher_but_never_below_default(depths, expected):
    assert adapt_publisher_qos([_pub(depth=d) for d in depths]).depth == expected
