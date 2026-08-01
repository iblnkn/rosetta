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

"""Unit tests for _TickGate, the sim-time pacer behind get_observation/send_action.

Pure logic tests: a fake clock stands in for the node clock and the module's
time.sleep is patched to advance it, so nothing here actually sleeps and every
release point is deterministic.
"""

import threading
from types import SimpleNamespace

import pytest

import lerobot_robot_rosetta.rosetta as adapter
from lerobot_robot_rosetta.rosetta import GATE_SLICE_SEC, _TickGate

PERIOD_NS = 100


class _FakeClock:
    def __init__(self, active=True, t=0):
        self.ros_time_is_active = active
        self.t = t

    def now(self):
        return SimpleNamespace(nanoseconds=self.t)


def _patch_sleep(monkeypatch, clock, step_ns):
    """Each slice-sleep advances the fake clock by step_ns; returns call log."""
    calls = []

    def _sleep(sec):
        assert sec == GATE_SLICE_SEC
        calls.append(sec)
        clock.t += step_ns

    monkeypatch.setattr(adapter.time, "sleep", _sleep)
    return calls


def test_no_clock_is_passthrough():
    gate = _TickGate(None, PERIOD_NS)
    gate.wait()
    assert gate._last_ns is None


def test_wall_clock_is_passthrough():
    gate = _TickGate(_FakeClock(active=False, t=500), PERIOD_NS)
    gate.wait()
    gate.wait()
    assert gate._last_ns is None


def test_clock_without_ros_time_attr_is_passthrough():
    # e.g. a plain clock double, or a non-ROSClock: getattr defaults to False
    clock = SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=0))
    gate = _TickGate(clock, PERIOD_NS)
    gate.wait()
    assert gate._last_ns is None


def test_first_gated_call_releases_immediately(monkeypatch):
    clock = _FakeClock(t=500)
    calls = _patch_sleep(monkeypatch, clock, step_ns=10)
    gate = _TickGate(clock, PERIOD_NS)
    gate.wait()
    assert calls == []
    assert gate._last_ns == 500


@pytest.mark.parametrize("step_ns", [10, 30, 250])  # slow, moderate, fast sim
def test_blocks_until_one_period_of_sim_time(monkeypatch, step_ns):
    clock = _FakeClock(t=0)
    calls = _patch_sleep(monkeypatch, clock, step_ns)
    gate = _TickGate(clock, PERIOD_NS)
    gate.wait()  # anchor at 0
    gate.wait()  # must block until t >= 100
    assert len(calls) > 0
    assert clock.t >= PERIOD_NS


def test_stop_event_releases_without_advancing_anchor():
    clock = _FakeClock(t=0)  # frozen: a paused sim
    stop = threading.Event()
    stop.set()
    gate = _TickGate(clock, PERIOD_NS, stop)
    gate.wait()  # anchor at 0
    gate.wait()  # would block forever without the stop escape
    assert gate._last_ns == 0


def test_stall_reanchors_without_burst(monkeypatch):
    clock = _FakeClock(t=0)
    calls = _patch_sleep(monkeypatch, clock, step_ns=30)
    gate = _TickGate(clock, PERIOD_NS)
    gate.wait()  # anchor at 0
    clock.t = 1000  # sim stalled (or paused) for 10 periods
    gate.wait()
    assert calls == []  # released immediately...
    assert gate._last_ns == 1000  # ...and re-anchored: no catch-up burst
    gate.wait()  # next tick costs a full period again
    assert len(calls) > 0
    assert clock.t >= 1000 + PERIOD_NS


def test_on_time_cadence_is_exact(monkeypatch):
    # Slice overshoot must not accumulate: anchors land on exact multiples of
    # the period even though each release overshoots by up to one slice-step.
    clock = _FakeClock(t=0)
    _patch_sleep(monkeypatch, clock, step_ns=30)
    gate = _TickGate(clock, PERIOD_NS)
    gate.wait()  # anchor at 0
    for expected_anchor in (100, 200, 300):
        gate.wait()
        assert gate._last_ns == expected_anchor


def test_backwards_jump_reanchors(monkeypatch):
    # Gazebo reset / bag loop: sim clock rewinds below the anchor.
    clock = _FakeClock(t=1000)
    calls = _patch_sleep(monkeypatch, clock, step_ns=30)
    gate = _TickGate(clock, PERIOD_NS)
    gate.wait()  # anchor at 1000
    clock.t = 50
    gate.wait()
    assert calls == []
    assert gate._last_ns == 50


def test_reset_clears_anchor(monkeypatch):
    clock = _FakeClock(t=500)
    calls = _patch_sleep(monkeypatch, clock, step_ns=30)
    gate = _TickGate(clock, PERIOD_NS)
    gate.wait()
    gate.reset()
    gate.wait()  # first call again: immediate
    assert calls == []
    assert gate._last_ns == 500
