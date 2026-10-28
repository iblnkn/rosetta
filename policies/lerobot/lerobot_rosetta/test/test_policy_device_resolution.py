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

"""Unit tests for resolve_policy_device (CPU fallback for unavailable backends)."""

import pytest

from lerobot_rosetta.device import resolve_policy_device

KNOWN_BACKENDS = {"cuda", "mps", "xpu", "cpu"}


def fake_availability(available):
    """A stand-in for lerobot's is_torch_device_available, mirroring its contract."""

    def is_available(backend):
        if backend not in KNOWN_BACKENDS:
            raise ValueError(f"Unknown device {backend}.")
        return backend == "cpu" or backend in available

    return is_available


@pytest.mark.parametrize(
    ("requested", "available", "expected"),
    [
        # Available backends are honored, including an explicit device index.
        ("cuda", {"cuda"}, "cuda"),
        ("cuda:1", {"cuda"}, "cuda:1"),
        ("xpu", {"xpu"}, "xpu"),
        ("mps", {"mps"}, "mps"),
        # Unavailable backends fall back to CPU.
        ("cuda", set(), "cpu"),
        ("cuda:1", set(), "cpu"),
        ("xpu", set(), "cpu"),
        ("mps", set(), "cpu"),
        # CPU is always available.
        ("cpu", set(), "cpu"),
        # Devices LeRobot doesn't know pass through; the server is the authority.
        ("npu", set(), "npu"),
        ("openvino", {"cuda"}, "openvino"),
        # Normalization: case and surrounding whitespace are stripped.
        ("  CUDA  ", {"cuda"}, "cuda"),
        ("XPU", {"xpu"}, "xpu"),
        # Empty / missing request defaults to CPU.
        ("", {"cuda"}, "cpu"),
        (None, {"cuda"}, "cpu"),
    ],
)
def test_resolve_policy_device(requested, available, expected):
    assert resolve_policy_device(requested, is_available=fake_availability(available)) == expected
