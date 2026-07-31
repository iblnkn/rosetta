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

"""Unit tests for _resolve_policy_device (ported from rosetta PR #33)."""

import pytest

from lerobot_rosetta.policy_runner import _resolve_policy_device


@pytest.mark.parametrize(
    ("requested", "kwargs", "expected"),
    [
        # CUDA: honored when available (incl. explicit index), else CPU fallback.
        ("cuda", {"torch_available": True, "cuda_available": True}, "cuda"),
        ("cuda:0", {"torch_available": True, "cuda_available": True}, "cuda:0"),
        ("cuda", {"torch_available": True, "cuda_available": False}, "cpu"),
        ("cuda", {"torch_available": False, "cuda_available": False}, "cpu"),
        # XPU: honored when available, else CPU fallback.
        ("xpu", {"torch_available": True, "xpu_available": True}, "xpu"),
        ("xpu", {"torch_available": True, "xpu_available": False}, "cpu"),
        ("xpu", {"torch_available": False, "xpu_available": False}, "cpu"),
        # NPU / openvino alias: both resolve to "npu" when available.
        ("npu", {"npu_available": True}, "npu"),
        ("openvino", {"npu_available": True}, "npu"),
        ("npu", {"npu_available": False}, "cpu"),
        ("openvino", {"npu_available": False}, "cpu"),
        # Devices without an availability probe pass through unchanged.
        ("cpu", {}, "cpu"),
        ("mps", {}, "mps"),
        # Normalization: case and surrounding whitespace are stripped.
        ("  CUDA  ", {"torch_available": True, "cuda_available": True}, "cuda"),
        ("XPU", {"torch_available": True, "xpu_available": True}, "xpu"),
        # Empty / missing request normalizes to the empty string.
        ("", {}, ""),
        (None, {}, ""),
    ],
)
def test_resolve_policy_device(requested, kwargs, expected):
    assert _resolve_policy_device(requested, **kwargs) == expected
