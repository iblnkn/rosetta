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

"""start_episode is start-and-return: the service must never hold an
executor thread for the episode.

The old synchronous handler ran the whole episode inside the callback
(unbounded with max_duration_s=0) on a 4-thread executor — one concurrent
call from starvation. Now the handler claims the busy guard, spawns the
episode thread, and returns; the thread releases the guard on every exit.
"""

import threading

import pytest
from rosetta_interfaces.srv import StartHILEpisode

from rosetta.robots.ros2.nodes.node_utils import wait_until
from rosetta.robots.ros2.nodes.rosetta_hil_manager_node import RosettaHilManagerNode


@pytest.fixture
def node(rclpy_ctx):
    n = RosettaHilManagerNode()
    yield n
    n.destroy_node()


def _call(node, prompt="t"):
    request = StartHILEpisode.Request(prompt=prompt)
    return node._handle_start_episode(request, StartHILEpisode.Response())


def test_start_returns_immediately_and_claims_guard(node, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(
        node,
        "_run_episode",
        lambda *_a: (release.wait(5.0), _fields())[1],
    )
    node._accepting_goals = True

    resp = _call(node)
    assert resp.accepted is True  # returned while the episode still runs
    assert node._episode_busy.busy

    # Concurrent start while the episode runs is rejected.
    resp2 = _call(node)
    assert resp2.accepted is False
    assert "in progress" in resp2.message

    release.set()
    assert wait_until(lambda: not node._episode_busy.busy, timeout=5.0)


def test_guard_released_when_episode_raises(node, monkeypatch):
    def _boom(*_a):
        raise RuntimeError("episode exploded")

    monkeypatch.setattr(node, "_run_episode", _boom)
    node._accepting_goals = True

    resp = _call(node)
    assert resp.accepted is True
    # The leak regression: the thread's finally must release the guard.
    assert wait_until(lambda: not node._episode_busy.busy, timeout=5.0)


def test_inactive_node_rejects_without_claiming(node):
    node._accepting_goals = False
    resp = _call(node)
    assert resp.accepted is False
    assert resp.message == "Node not active"
    assert not node._episode_busy.busy


def _fields():
    return (
        {
            "success": True,
            "message": "done",
            "termination_reason": "timeout",
            "final_reward": 0.0,
            "bag_path": "/tmp/bag",
            "messages_written": 3,
        },
        False,
    )
