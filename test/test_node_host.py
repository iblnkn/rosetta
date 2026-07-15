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

"""NodeHost contract: start/stop lifecycle, rollback, and spin-death poisoning.

The rollback test is a regression guard: a failing make_node() left a live
rclpy Context behind with rclpy.init() already called; a retry on the same
NodeHost instance then overwrote that reference, leaking it and its DDS
participant for the life of the process. start() must roll back via stop()
before re-raising.
"""

import pytest
import rclpy
from rosetta.robots.ros2.node_host import NodeHost


def test_start_rolls_back_context_on_failure():
    host = NodeHost()

    with pytest.raises(RuntimeError):
        host.start(lambda ctx: (_ for _ in ()).throw(RuntimeError("boom")))

    assert host._context is None
    assert host.node is None
    assert host._executor is None
    assert host._spin_thread is None

    # A retry on the same instance must get a fresh context, not reuse or
    # leak the failed one.
    node = host.start(lambda ctx: rclpy.create_node("test_node_host_retry", context=ctx))
    assert node is not None
    assert host.node is node

    host.stop()
    assert host.node is None
    assert host._context is None


def test_stop_without_start_is_noop():
    NodeHost().stop()


def test_start_is_idempotent():
    host = NodeHost()
    try:
        node = host.start(lambda ctx: rclpy.create_node("test_node_host_idem", context=ctx))
        context = host._context

        def fail_factory(_ctx):
            raise AssertionError("second start() must not invoke the factory")

        assert host.start(fail_factory) is node
        assert host._context is context
    finally:
        host.stop()


def test_spin_thread_death_poisons_node_access():
    """A callback exception must not leave a host that looks healthy.

    The executor re-raises callback exceptions out of spin(), killing the
    spin thread. Accessing host.node afterwards must raise (chained to the
    original error) instead of serving a node whose subscriptions, timers,
    and watchdog have all silently stopped. stop() still tears down and
    resets the host.
    """

    def make(ctx):
        node = rclpy.create_node("test_node_host_spin_death", context=ctx)
        node.create_timer(0.01, lambda: (_ for _ in ()).throw(RuntimeError("callback boom")))
        return node

    host = NodeHost()
    host.start(make)
    host._spin_thread.join(timeout=5.0)
    assert not host._spin_thread.is_alive(), "timer exception should have killed the spin thread"

    with pytest.raises(RuntimeError, match="spin thread died") as excinfo:
        _ = host.node
    assert "callback boom" in str(excinfo.value.__cause__)

    # start()'s idempotency early-return must consult the same poison: a
    # retry must fail loudly, not silently receive the dead node.
    with pytest.raises(RuntimeError, match="spin thread died"):
        host.start(lambda ctx: rclpy.create_node("never_built", context=ctx))

    host.stop()
    assert host.node is None
