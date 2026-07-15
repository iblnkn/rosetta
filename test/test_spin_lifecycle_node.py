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

"""spin_lifecycle_node: the shutdown transition is driven on the way out.

destroy_node() never runs on_shutdown -- only the lifecycle state machine
does -- so exiting through spin_lifecycle_node must trigger_shutdown() before
destroying the node, or external resources (an open bag writer, a policy
server subprocess, the safety send) leak on Ctrl-C.
"""

from rosetta.robots.ros2.nodes import node_utils


class _FakeExecutor:
    def __init__(self, num_threads):
        self.num_threads = num_threads

    def add_node(self, node):
        self.node = node

    def spin_once(self, timeout_sec):
        raise KeyboardInterrupt


class _FakeLogger:
    def __init__(self, calls):
        self._calls = calls

    def warning(self, msg):
        self._calls.append(f"warning: {msg}")


class _FakeNode:
    def __init__(self, calls, shutdown_raises=False):
        self.calls = calls
        self._shutdown_raises = shutdown_raises

    def trigger_shutdown(self):
        self.calls.append("trigger_shutdown")
        if self._shutdown_raises:
            raise RuntimeError("context already invalid")

    def destroy_node(self):
        self.calls.append("destroy_node")

    def get_logger(self):
        return _FakeLogger(self.calls)


def _patch_rclpy(monkeypatch, calls):
    monkeypatch.setattr(node_utils.rclpy, "init", lambda **kwargs: calls.append("init"))
    monkeypatch.setattr(node_utils.rclpy, "try_shutdown", lambda: calls.append("try_shutdown"))
    monkeypatch.setattr(node_utils, "MultiThreadedExecutor", _FakeExecutor)


def test_shutdown_transition_driven_before_destroy(monkeypatch):
    calls: list[str] = []
    _patch_rclpy(monkeypatch, calls)

    assert node_utils.spin_lifecycle_node(lambda: _FakeNode(calls)) == 0

    assert calls == ["init", "trigger_shutdown", "destroy_node", "try_shutdown"]


def test_node_destroyed_even_when_shutdown_transition_fails(monkeypatch):
    calls: list[str] = []
    _patch_rclpy(monkeypatch, calls)

    assert node_utils.spin_lifecycle_node(lambda: _FakeNode(calls, shutdown_raises=True)) == 0

    assert "destroy_node" in calls and "try_shutdown" in calls
    assert any(c.startswith("warning:") for c in calls)  # failure surfaced, not swallowed
