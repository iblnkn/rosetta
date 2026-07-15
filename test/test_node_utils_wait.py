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

"""wait_until: the bounded polling wait under every deactivate/stop path.

Production deactivation (RosettaLifecycleNode._stop_and_secure) and several
tests wait on this helper — a deadline or re-check bug here would silently
turn those waits into no-ops or hangs, so its contract is pinned directly.
"""

import time

from rosetta.robots.ros2.nodes.node_utils import wait_until


def test_true_predicate_returns_immediately():
    start = time.monotonic()
    assert wait_until(lambda: True, timeout=5.0)
    # No poll sleeps: an already-true condition must not burn the timeout.
    assert time.monotonic() - start < 1.0


def test_false_predicate_times_out_within_bound():
    start = time.monotonic()
    assert not wait_until(lambda: False, timeout=0.2, poll=0.05)
    elapsed = time.monotonic() - start
    assert elapsed >= 0.2
    # May overshoot by at most one poll interval (plus scheduling slack).
    assert elapsed < 1.0


def test_predicate_becoming_true_is_observed():
    deadline = time.monotonic() + 0.1
    assert wait_until(lambda: time.monotonic() >= deadline, timeout=5.0, poll=0.02)


def test_condition_true_during_final_poll_is_not_reported_as_timeout():
    # The post-loop re-check exists for exactly this: the condition turns
    # true during the last sleep, after the loop's deadline check.
    flips = iter([False, True])
    assert wait_until(lambda: next(flips, True), timeout=0.01, poll=0.05)


def test_zero_timeout_degenerates_to_single_check():
    assert wait_until(lambda: True, timeout=0.0)
    assert not wait_until(lambda: False, timeout=0.0)
