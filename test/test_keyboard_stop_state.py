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

"""The keyboard UI must not mark recording stopped when the stop call failed.

Clearing _recording on a failed stop desyncs the UI from the recorder: the
recorder keeps recording, the next start gets "Already recording" while the
keyboard claims idle.
"""

from types import SimpleNamespace

from rosetta.robots.ros2.nodes.episode_keyboard_node import EpisodeKeyboardNode


def _bare_node(recording: bool) -> EpisodeKeyboardNode:
    """Node instance without ROS init: only the fields _on_stop_done touches."""
    node = object.__new__(EpisodeKeyboardNode)
    node._service_pending = True
    node._recording = recording
    node._messages = []
    node._msg = node._messages.append
    return node


class _Future:
    def __init__(self, result=None, error=None):
        self._result, self._error = result, error

    def result(self):
        if self._error:
            raise self._error
        return self._result


def test_stop_success_clears_recording():
    node = _bare_node(recording=True)
    node._on_stop_done(_Future(SimpleNamespace(success=True, message="")))
    assert node._recording is False
    assert node._service_pending is False


def test_stop_failure_keeps_recording():
    node = _bare_node(recording=True)
    node._on_stop_done(_Future(SimpleNamespace(success=False, message="writer busy")))
    assert node._recording is True  # recorder is still recording; stay in sync
    assert node._service_pending is False  # but allow a retry


def test_stop_exception_keeps_recording():
    node = _bare_node(recording=True)
    node._on_stop_done(_Future(error=RuntimeError("service died")))
    assert node._recording is True
    assert node._service_pending is False
