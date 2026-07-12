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

"""Tests for StreamBuffer resampling policies and clock-reset handling."""

from rosetta.contract.schema import ResamplePolicy
from rosetta.frames.resample import StreamBuffer

STEP = 100  # ns between ticks
TOL = 50  # ns asof tolerance


def test_sample_before_any_push_returns_none():
    buf = StreamBuffer(ResamplePolicy.HOLD.value, STEP)
    assert buf.sample(1000) is None


def test_hold_carries_last_value_forward():
    buf = StreamBuffer(ResamplePolicy.HOLD.value, STEP)
    buf.push(1000, 1.0)
    # Hold returns the last value at any later tick.
    assert buf.sample(1000) == 1.0
    assert buf.sample(5000) == 1.0


def test_drop_only_returns_recent_value():
    buf = StreamBuffer(ResamplePolicy.DROP.value, STEP)
    buf.push(1000, 2.0)
    # Within one step window of the sample: returned.
    assert buf.sample(1000 + STEP - 1) == 2.0
    # Older than the step window: dropped.
    assert buf.sample(1000 + STEP + 1) is None


def test_asof_respects_tolerance():
    buf = StreamBuffer(ResamplePolicy.ASOF.value, STEP, tol_ns=TOL)
    buf.push(1000, 3.0)
    assert buf.sample(1000 + TOL) == 3.0  # within tolerance
    assert buf.sample(1000 + TOL + 1) is None  # outside tolerance


def test_push_keeps_newest_by_timestamp():
    buf = StreamBuffer(ResamplePolicy.HOLD.value, STEP)
    buf.push(2000, "new")
    buf.push(1000, "old")  # older timestamp ignored
    assert buf.sample(3000) == "new"


SEC = 1_000_000_000  # ns


def test_clock_reset_clears_buffer():
    # Realistic reset magnitudes: a sim restart jumps the sampling clock
    # backwards by whole seconds, far beyond the skew tolerance.
    buf = StreamBuffer(ResamplePolicy.HOLD.value, STEP)
    buf.push(5 * SEC, 9.0)
    assert buf.sample(1 * SEC) is None
    assert buf.sample(6 * SEC) is None  # buffer was cleared, nothing to hold


def test_normal_jitter_does_not_clear():
    buf = StreamBuffer(ResamplePolicy.HOLD.value, STEP)
    buf.push(1000, 7.0)
    # Sampling exactly at the buffered stamp must NOT be treated as a reset
    # (strict > comparison), so the value is still available.
    assert buf.sample(1000) == 7.0


# ---------- clock skew vs reset ----------


def test_slight_future_stamp_served_under_all_policies():
    # A sensor host clock slightly ahead (multi-machine header stamps) must
    # not wipe the freshest value — it is served as age 0 under every policy.
    ahead = 50_000_000  # 50 ms
    for policy, kwargs in [
        (ResamplePolicy.HOLD.value, {}),
        (ResamplePolicy.ASOF.value, {"tol_ns": TOL}),
        (ResamplePolicy.DROP.value, {}),
    ]:
        buf = StreamBuffer(policy, STEP, **kwargs)
        tick = 10 * SEC
        buf.push(tick + ahead, 4.2)
        assert buf.sample(tick) == 4.2, policy
        # last_ts untouched: a persistently-ahead sensor keeps serving.
        assert buf.sample(tick + 1) == 4.2, policy


def test_forward_jump_beyond_tolerance_clears():
    buf = StreamBuffer(ResamplePolicy.HOLD.value, STEP)
    tick = 10 * SEC
    buf.push(tick + 2 * SEC, 1.0)  # 2 s ahead: beyond the 1 s default
    assert buf.sample(tick) is None
    assert buf.sample(tick + 1) is None  # cleared, stays cleared


def test_reset_tolerance_floor_scales_with_step():
    # At very low rates one frame period can exceed the 1 s default; the
    # tolerance floors at two periods.
    buf = StreamBuffer(ResamplePolicy.HOLD.value, step_ns=1 * SEC)
    assert buf.reset_tol_ns == 2 * SEC


def test_explicit_zero_tolerance_restores_strict_behavior():
    buf = StreamBuffer(ResamplePolicy.HOLD.value, STEP, reset_tol_ns=0)
    buf.push(1001, 5.0)
    assert buf.sample(1000) is None  # any future stamp clears
