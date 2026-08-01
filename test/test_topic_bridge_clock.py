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

"""TopicBridge.clock exposes the host node's (sim-time aware) clock.

Consumers that pace on robot time (the LeRobot adapter's tick gates) reach
the node clock only through this accessor, so it must be the node's own clock
object and must fail loudly before setup() binds a node.
"""

from types import SimpleNamespace

import pytest

from rosetta.robots.ros2.topic_bridge import TopicBridge


class _FakeNode:
    def __init__(self):
        self._clock = SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=0))

    def get_clock(self):
        return self._clock


def test_clock_raises_before_setup():
    bridge = TopicBridge([], [], fps=10)
    with pytest.raises(RuntimeError, match="before setup"):
        _ = bridge.clock


def test_clock_is_host_node_clock():
    bridge = TopicBridge([], [], fps=10)
    node = _FakeNode()
    bridge._node = node
    assert bridge.clock is node.get_clock()
