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

from rosetta.common.contract import ResamplePolicy
from rosetta.common.contract_utils import StreamBuffer

STEP = 100  # ns between ticks
TOL = 50    # ns asof tolerance


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
    assert buf.sample(1000 + TOL) == 3.0       # within tolerance
    assert buf.sample(1000 + TOL + 1) is None  # outside tolerance


def test_push_keeps_newest_by_timestamp():
    buf = StreamBuffer(ResamplePolicy.HOLD.value, STEP)
    buf.push(2000, 'new')
    buf.push(1000, 'old')  # older timestamp ignored
    assert buf.sample(3000) == 'new'


def test_clock_reset_clears_buffer():
    buf = StreamBuffer(ResamplePolicy.HOLD.value, STEP)
    buf.push(5000, 9.0)
    # A tick in the past relative to the buffered stamp signals a clock reset
    # (e.g. sim restart): the stale value is cleared and None returned.
    assert buf.sample(1000) is None
    assert buf.sample(6000) is None  # buffer was cleared, nothing to hold


def test_normal_jitter_does_not_clear():
    buf = StreamBuffer(ResamplePolicy.HOLD.value, STEP)
    buf.push(1000, 7.0)
    # Sampling exactly at the buffered stamp must NOT be treated as a reset
    # (strict > comparison), so the value is still available.
    assert buf.sample(1000) == 7.0
