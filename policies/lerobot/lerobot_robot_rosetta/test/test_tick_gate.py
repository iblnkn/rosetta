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
release point is deterministic. Pacing through the adapter (first call, one
period, reset, stop) is covered in test_adapter.py.
"""

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
        assert len(calls) < 1000, "gate never released"
        clock.t += step_ns

    monkeypatch.setattr(adapter.time, "sleep", _sleep)
    return calls


@pytest.mark.parametrize(
    "clock",
    [
        None,
        _FakeClock(active=False, t=500),
        # e.g. a plain clock double, or a non-ROSClock: getattr defaults to False
        SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=0)),
    ],
    ids=["no_clock", "wall_clock", "clock_without_ros_time_attr"],
)
def test_gate_is_passthrough_without_ros_time(monkeypatch, clock):
    def _sleep(sec):
        raise AssertionError("a passthrough gate slept")

    monkeypatch.setattr(adapter.time, "sleep", _sleep)
    gate = _TickGate(clock, PERIOD_NS)
    gate.wait()
    gate.wait()  # a gated second call would sleep: no time has passed


def test_stall_reanchors_without_burst(monkeypatch):
    clock = _FakeClock(t=0)
    calls = _patch_sleep(monkeypatch, clock, step_ns=30)
    gate = _TickGate(clock, PERIOD_NS)
    gate.wait()  # anchor at 0
    clock.t = 1000  # sim stalled (or paused) for 10 periods
    gate.wait()
    assert calls == []  # released immediately...
    gate.wait()  # ...and the next tick costs a full period again: no catch-up burst
    assert 1000 + PERIOD_NS <= clock.t < 1000 + PERIOD_NS + 30


def test_on_time_cadence_is_exact(monkeypatch):
    # Slice overshoot must not accumulate: releases stay on the period grid
    # even though each release overshoots by up to one slice-step.
    clock = _FakeClock(t=0)
    _patch_sleep(monkeypatch, clock, step_ns=30)
    gate = _TickGate(clock, PERIOD_NS)
    gate.wait()  # anchor at 0
    releases = []
    for _ in range(3):
        gate.wait()
        releases.append(clock.t)
    # Anchors 100, 200, 300; each release is the first 30ns slice at or past one.
    assert releases == [120, 210, 300]


def test_backwards_jump_reanchors(monkeypatch):
    # Gazebo reset / bag loop: sim clock rewinds below the anchor.
    clock = _FakeClock(t=1000)
    calls = _patch_sleep(monkeypatch, clock, step_ns=30)
    gate = _TickGate(clock, PERIOD_NS)
    gate.wait()  # anchor at 1000
    clock.t = 50
    gate.wait()
    assert calls == []  # released immediately, re-anchored at 50
    gate.wait()
    assert 50 + PERIOD_NS <= clock.t < 50 + PERIOD_NS + 30
