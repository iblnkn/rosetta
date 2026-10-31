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

"""Stats from sufficient statistics equal stats computed on every value."""

import numpy as np
import pytest

pytest.importorskip("rerun")

from rerun_rosetta.stats import QUANTILES, histogram_stats, quantile_name, value_stats


def _numpy_stats(values):
    return {
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        **{quantile_name(q): v for q, v in zip(QUANTILES, np.quantile(values, QUANTILES, axis=0), strict=True)},
    }


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("pixels_per_frame", [1, 2, 7, 480 * 480])
def test_histogram_stats_equal_stats_over_every_pixel(seed, pixels_per_frame):
    rng = np.random.default_rng(seed)
    frames = [rng.integers(0, 256, size=(pixels_per_frame, 3), dtype=np.uint8) for _ in range(3)]
    if seed % 2:  # concentrated values, so quantiles land between equal and unequal neighbours
        frames = [np.minimum(f, 40) for f in frames]
    histograms = np.zeros((3, 256), dtype=np.int64)
    for f in frames:
        for c in range(3):
            histograms[c] += np.bincount(f[:, c], minlength=256)

    got = histogram_stats(histograms, frames=len(frames))
    want = _numpy_stats(np.concatenate(frames).astype(np.float64))
    assert got["count"].tolist() == [3]
    for name, value in want.items():
        np.testing.assert_allclose(got[name], value, rtol=0, atol=1e-9, err_msg=name)


def test_value_stats_of_concatenated_episodes_equal_stats_of_all_values():
    rng = np.random.default_rng(0)
    episodes = [rng.normal(size=(n, 4)) for n in (5, 1, 17)]
    got = value_stats(np.concatenate(episodes))
    want = _numpy_stats(np.concatenate(episodes))
    assert got["count"].tolist() == [23]
    for name, value in want.items():
        np.testing.assert_array_equal(got[name], value, err_msg=name)
