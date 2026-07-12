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

"""TRANSIENT_LOCAL buffer retention depth.

Regression: /tf_static is TRANSIENT_LOCAL, and multiple independent nodes
commonly publish it (robot_state_publisher plus a sensor-calibration
static_transform_publisher). The buffer's retention depth used to be taken
straight from the contract's configured QoS depth (conventionally 1, a
wire/DDS matching hint) -- so a second publisher's latched message silently
evicted the first's from the buffer. It must be sized to at least the live
publisher count instead.
"""

from rosetta.robots.ros2.nodes.episode_recorder_node import (
    MAX_BUFFERED_MESSAGES_PER_TOPIC,
    EpisodeRecorderNode,
)

_depth = EpisodeRecorderNode._latched_buffer_depth


def test_multiple_publishers_widen_beyond_configured_depth():
    # depth=1 (as configured for /tf_static in stone.yaml) with 2 live
    # publishers must retain both, not evict down to 1.
    assert _depth(configured_depth=1, publisher_count=2) == 2


def test_single_publisher_keeps_configured_depth():
    assert _depth(configured_depth=1, publisher_count=1) == 1


def test_configured_depth_wins_when_larger_than_publisher_count():
    assert _depth(configured_depth=5, publisher_count=1) == 5


def test_clamped_to_safety_ceiling():
    assert _depth(configured_depth=1, publisher_count=10_000) == MAX_BUFFERED_MESSAGES_PER_TOPIC
