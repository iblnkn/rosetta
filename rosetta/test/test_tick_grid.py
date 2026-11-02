# Copyright 2026 Isaac Blankenau
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

"""The frame grid: exact tick times, and how many ticks a recording holds."""

from fractions import Fraction

import pytest

from rosetta.frames.resample import num_ticks, step_ns, tick_ns

SEC = 1_000_000_000


@pytest.mark.parametrize("fps", [7, 10, 15, 30, 50, 60])
def test_ticks_stay_within_a_nanosecond_of_exact_after_an_hour(fps):
    start = 1_700_000_000 * SEC
    index = fps * 3600
    exact = start + Fraction(index * SEC, fps)
    assert 0 <= exact - tick_ns(start, index, fps) < 1


def test_adding_the_rounded_period_would_drift():
    # The reason ticks aren't start + index * step_ns: at 15 fps that is 36 us early after an hour.
    index = 15 * 3600
    assert tick_ns(0, index, 15) - index * step_ns(15) == 36_000


@pytest.mark.parametrize("fps", [7, 10, 15, 30])
@pytest.mark.parametrize("duration", [0, 1, SEC // 30, SEC // 10, SEC // 10 - 1, SEC // 10 + 1, 2 * SEC + 7])
def test_num_ticks_counts_ticks_at_or_before_the_end(fps, duration):
    expected = sum(1 for k in range(fps * 4) if tick_ns(0, k, fps) <= duration)
    assert num_ticks(duration, fps) == expected
