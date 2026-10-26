# Copyright 2026 Isaac Blankenau
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

"""TopicBridge structurally satisfies the FrameIO protocol PolicyRunner.run() is typed against."""

from rosetta.frames.protocols import FrameIO
from rosetta_ros.topic_bridge import TopicBridge


def test_topic_bridge_satisfies_frame_stream():
    assert isinstance(TopicBridge([], [], fps=10), FrameIO)
