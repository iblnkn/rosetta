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

"""NodeHost: own a node on a private rclpy context (init, executor, spin thread).

The standalone-mode plumbing shared by the LeRobot robot and teleoperator
adapters, which embed a node in a process rosetta does not own (LeRobot's).
Each host initializes its own rclpy Context, never the global one, so
multiple hosts in one process (robot + teleoperator) and any other rclpy
user start and stop independently. Initializing a non-default context also
installs no signal handlers, leaving SIGINT/SIGTERM to the host process.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import rclpy
from rclpy.context import Context
from rclpy.executors import Executor, ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node

THREAD_JOIN_TIMEOUT_SEC = 1.0


class NodeHost:
    """Creates a node via a factory, spins it on a daemon thread, owns teardown."""

    def __init__(self) -> None:
        self._node: Node | None = None
        self._context: Context | None = None
        self._executor: SingleThreadedExecutor | None = None
        self._spin_thread: threading.Thread | None = None
        self._spin_exc: BaseException | None = None

    @property
    def node(self) -> Node | None:
        """The hosted node, or None before start()/after stop().

        Raises instead of returning a node whose spin thread has died: a
        dead executor means subscriptions, timers, and any safety watchdog
        have all silently stopped, so serving the node as if it were healthy
        would let callers keep reading ever-staler data.
        """
        if self._spin_exc is not None:
            raise RuntimeError("NodeHost spin thread died; node is not being spun") from self._spin_exc
        return self._node

    def start(self, make_node: Callable[[Context], Node]) -> Node:
        """Create the node and start spinning it (idempotent).

        A second call returns the existing node and ignores the factory.
        The factory receives the host's private context and must pass it to
        the node constructor. The executor exists for subscriptions, timers,
        and external lifecycle service requests. Direct trigger_*() calls are
        synchronous and local, so they do not need it.

        If anything below raises, stop() rolls back whatever was already
        built instead of leaking the context: a retry on this same instance
        must get a fresh context, not silently orphan the failed one.
        """
        if self._node is not None:
            # Route through the property: it raises if the spin thread died,
            # so a retry can't silently receive a dead node.
            return self.node
        self._spin_exc = None
        self._context = Context()
        try:
            rclpy.init(context=self._context)
            self._node = make_node(self._context)
            self._executor = SingleThreadedExecutor(context=self._context)
            self._executor.add_node(self._node)
            thread = threading.Thread(target=self._spin, args=(self._executor,), daemon=True)
            thread.start()
            self._spin_thread = thread
        except Exception:
            self.stop()
            raise
        return self._node

    def _spin(self, executor: Executor) -> None:
        try:
            executor.spin()  # blocks until stop() shuts the executor down
        except ExternalShutdownException:
            pass  # context torn down out from under the executor
        except BaseException as e:
            # A callback raised. Poison the host so the next node access
            # fails loudly, then re-raise for the thread excepthook.
            self._spin_exc = e
            raise

    def stop(self) -> None:
        """Shut down executor, spin thread, node, and context. Safe when never started.

        The host is reusable afterwards: a later start() builds a fresh
        context. Raises if the spin thread outlives the join timeout. The
        node must never be destroyed under a still-running executor thread,
        so node and context stay alive for a retry to re-join.
        """
        if self._executor is not None:
            self._executor.shutdown()

        if self._spin_thread is not None:
            self._spin_thread.join(timeout=THREAD_JOIN_TIMEOUT_SEC)
            if self._spin_thread.is_alive():
                raise RuntimeError(f"NodeHost spin thread did not exit within {THREAD_JOIN_TIMEOUT_SEC}s")
            self._spin_thread = None
        self._executor = None

        if self._node is not None:
            self._node.destroy_node()
            self._node = None

        if self._context is not None:
            self._context.try_shutdown()
            self._context = None

        # The poison's job is done once the dead node is torn down.
        self._spin_exc = None
