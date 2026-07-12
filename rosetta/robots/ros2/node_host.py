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
Each host initializes its own rclpy Context — never the global one — so
multiple hosts in one process (robot + teleoperator) and any other rclpy
user start and stop independently. Initializing a non-default context also
installs no signal handlers, leaving SIGINT/SIGTERM to the host process.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

import rclpy
from rclpy.context import Context
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node

THREAD_JOIN_TIMEOUT_SEC = 1.0


class NodeHost:
    """Creates a node via a factory, spins it on a daemon thread, owns teardown."""

    def __init__(self) -> None:
        self.node: Optional[Node] = None
        self._context: Optional[Context] = None
        self._executor: Optional[SingleThreadedExecutor] = None
        self._spin_thread: Optional[threading.Thread] = None

    def start(self, make_node: Callable[[Context], Node]) -> Node:
        """Create the node (idempotent) and start spinning it.

        The factory receives the host's private context and must pass it to
        the node constructor. The spin thread starts before any lifecycle
        transition is triggered — transitions are service calls that need a
        spinning executor.
        """
        if self.node is not None:
            return self.node
        self._context = Context()
        rclpy.init(context=self._context)
        self.node = make_node(self._context)
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()
        return self.node

    def _spin(self) -> None:
        try:
            self._executor.spin()  # blocks until stop() shuts the executor down
        except ExternalShutdownException:
            pass  # context torn down out from under the executor

    def stop(self) -> None:
        """Shut down executor, spin thread, node, and context. Safe when never started.

        The host is reusable afterwards: a later start() builds a fresh context.
        """
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

        if self._spin_thread is not None:
            self._spin_thread.join(timeout=THREAD_JOIN_TIMEOUT_SEC)
            self._spin_thread = None

        if self.node is not None:
            self.node.destroy_node()
            self.node = None

        if self._context is not None:
            self._context.try_shutdown()
            self._context = None

    def __enter__(self) -> NodeHost:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
