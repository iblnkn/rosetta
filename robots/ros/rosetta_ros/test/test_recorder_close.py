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

"""Finalizing a bag doesn't hold the lock every subscription callback takes.

The node spins on one executor thread, so a callback blocked on the writer
lock while the MCAP index is written would stall cancels, services and
lifecycle transitions for as long as the finalize takes.
"""

import threading
from types import SimpleNamespace

from rosetta_ros.nodes.episode_recorder_node import EpisodeRecorderNode


class _SlowWriter:
    def __init__(self):
        self.closing, self.release = threading.Event(), threading.Event()

    def close(self):
        self.closing.set()
        self.release.wait(5.0)


def _node(writer):
    node = SimpleNamespace(
        _writer=writer, _writer_lock=threading.Lock(), _finalize_lock=threading.Lock(), _episode_subs={}
    )
    node.destroy_subscription = lambda sub: None
    return node


def _close_on_thread(node):
    closer = threading.Thread(target=EpisodeRecorderNode._close_writer, args=(node,))
    closer.start()
    return closer


def test_the_writer_lock_is_free_while_the_bag_finalizes():
    writer = _SlowWriter()
    node = _node(writer)
    closer = _close_on_thread(node)
    try:
        assert writer.closing.wait(2.0)
        assert node._writer is None  # callbacks see no writer and return at once
        assert node._writer_lock.acquire(timeout=0.5), "a callback would block until the finalize ends"
        node._writer_lock.release()
    finally:
        writer.release.set()
        closer.join(5.0)


def test_a_close_at_shutdown_waits_for_the_finalize_in_progress():
    # teardown() closes the writer too. Returning while the record thread is
    # still finalizing would let the process exit and leave the bag unindexed.
    writer = _SlowWriter()
    node = _node(writer)
    finalizing = _close_on_thread(node)
    try:
        assert writer.closing.wait(2.0)
        teardown = _close_on_thread(node)
        teardown.join(0.5)
        assert teardown.is_alive(), "teardown returned while the bag was still finalizing"
        writer.release.set()
        teardown.join(5.0)
        assert not teardown.is_alive()
    finally:
        writer.release.set()
        finalizing.join(5.0)
