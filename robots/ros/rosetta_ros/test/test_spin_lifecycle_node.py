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

import pytest
from rclpy.exceptions import InvalidHandle
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn

from rosetta_ros import node_utils


class _FakeExecutor:
    """Stands in for the spin loop: Ctrl-C on the first spin."""

    def __init__(self, num_threads):
        pass

    def add_node(self, node):
        pass

    def spin_once(self, timeout_sec):
        raise KeyboardInterrupt


class _Node(LifecycleNode):
    def __init__(self, name, *, shutdown_raises=False):
        super().__init__(name)
        self.shutdown_saw_live_node = None
        self._shutdown_raises = shutdown_raises

    def on_shutdown(self, state):
        # rclpy still runs on_shutdown after destroy_node(), but the node's
        # publishers are gone by then, so a final safety send would fail.
        try:
            self.get_name()
            self.shutdown_saw_live_node = True
        except InvalidHandle:
            self.shutdown_saw_live_node = False
        return TransitionCallbackReturn.SUCCESS

    def trigger_shutdown(self):
        if self._shutdown_raises:
            raise RuntimeError("context already invalid")
        return super().trigger_shutdown()


@pytest.fixture
def spin(monkeypatch):
    # The session context belongs to conftest: spin_lifecycle_node must not
    # init or shut it down here.
    monkeypatch.setattr(node_utils.rclpy, "init", lambda **kwargs: None)
    monkeypatch.setattr(node_utils.rclpy, "try_shutdown", lambda: None)
    monkeypatch.setattr(node_utils, "MultiThreadedExecutor", _FakeExecutor)
    return node_utils.spin_lifecycle_node


def test_ctrl_c_runs_on_shutdown_before_destroy(spin):
    node = _Node("spin_shutdown")

    assert spin(lambda: node) == 0

    assert node.shutdown_saw_live_node is True


def test_node_destroyed_even_when_shutdown_transition_fails(spin):
    node = _Node("spin_shutdown_fails", shutdown_raises=True)

    assert spin(lambda: node) == 0

    with pytest.raises(InvalidHandle):
        node.get_name()
