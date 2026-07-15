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

"""qos_profile_from_dict: strict contract-qos translation.

A qos typo must die with a ValueError (surfaced as ContractValidationError
at load), never silently become the default policy. No rclpy.init needed —
QoSProfile is plain construction.
"""

import pytest
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rosetta.robots.ros2.ros2_utils import qos_profile_from_dict


@pytest.mark.parametrize(
    ("d", "expected"),
    [
        (None, QoSProfile(depth=10)),
        ({}, QoSProfile(depth=10)),
        ({"depth": 5}, QoSProfile(depth=5)),
        # Integral floats are accepted: YAML writers produce 5.0.
        ({"depth": 5.0}, QoSProfile(depth=5)),
        # Values are case-folded, matching the loader's enum normalization.
        ({"reliability": " BEST_EFFORT "}, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)),
        (
            {"reliability": "best_effort", "history": "keep_all", "durability": "transient_local", "depth": 1},
            QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_ALL,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                depth=1,
            ),
        ),
    ],
)
def test_valid_mappings(d, expected):
    assert qos_profile_from_dict(d) == expected


@pytest.mark.parametrize(
    ("d", "match"),
    [
        ({"reliablity": "reliable"}, "Unknown qos key"),
        ({"reliability": "best-effort"}, "Invalid qos reliability"),
        ({"history": "keep-all"}, "Invalid qos history"),
        ({"durability": "transient-local"}, "Invalid qos durability"),
        ({"depth": -1}, "depth"),
        ({"depth": True}, "depth"),
        ({"depth": 2.5}, "depth"),
        ({"depth": "ten"}, "depth"),
    ],
)
def test_invalid_mappings_raise(d, match):
    with pytest.raises(ValueError, match=match):
        qos_profile_from_dict(d)
