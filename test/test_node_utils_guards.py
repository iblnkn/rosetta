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

"""BusyGuard: accept-time mutual exclusion for one-goal-at-a-time nodes.

Regression guard for the goal-accept race: under a MultiThreadedExecutor
with a ReentrantCallbackGroup, two goal requests could both pass a bare
`self._active_goal is not None` check before either _execute assigned it,
running two policy loops against one runner/bridge.
"""

import threading

from rosetta.robots.ros2.nodes.node_utils import BusyGuard


def test_second_acquire_rejected_until_release():
    g = BusyGuard()
    assert g.try_acquire()
    assert not g.try_acquire()
    assert g.busy
    g.release()
    assert not g.busy
    assert g.try_acquire()


def test_release_is_idempotent():
    g = BusyGuard()
    g.release()
    g.release()
    assert g.try_acquire()


def test_concurrent_acquire_exactly_one_winner():
    # 32 threads race try_acquire simultaneously; exactly one must win.
    g = BusyGuard()
    start = threading.Barrier(32)
    wins = []

    def worker():
        start.wait()
        if g.try_acquire():
            wins.append(1)

    threads = [threading.Thread(target=worker) for _ in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(wins) == 1
