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

"""TopicBridge's public surface: the FrameIO protocol and the host clock.

PolicyRunner.run() is typed against FrameIO, so a renamed or dropped bridge
method must fail here rather than hide behind adapter test doubles. Consumers
that pace on robot time (the LeRobot adapter's tick gates) reach the node
clock only through ``clock``, so it must be the node's own (sim-time aware)
clock object and must fail loudly before setup() binds a node.
"""

import pytest
from bridge_helpers import FakeNode

from rosetta.frames.protocols import FrameIO
from rosetta_ros.topic_bridge import TopicBridge


def test_topic_bridge_satisfies_frame_io():
    assert isinstance(TopicBridge([], [], fps=10), FrameIO)


def test_clock_raises_before_setup():
    bridge = TopicBridge([], [], fps=10)
    with pytest.raises(RuntimeError, match="before setup"):
        _ = bridge.clock


def test_clock_is_host_node_clock():
    bridge = TopicBridge([], [], fps=10)
    node = FakeNode()
    bridge._node = node
    assert bridge.clock is node.get_clock()
