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

"""Tests for image/string feature building and image zero-fill shape agreement.

Regression guard for the image zero-fill bug: a missing image frame must
zero-fill with the same shape as a real decoded frame and the declared feature.
Numeric feature/zero-fill agreement lives in test_frame_layout.py — numeric
shapes are a whole-key concern (concatenation), owned by FrameLayout.
"""

import numpy as np
import pytest
from rosetta.frames.layout import (
    DECODED_IMAGE_CHANNELS,
    build_feature,
    zero_image_for_spec,
)


class _ImageSpec:
    """Duck-typed stand-in for an ObservationStreamSpec image stream."""

    is_image = True
    dtype = "video"
    namespace = None
    operators = ()

    def __init__(self, key, resize):
        self.key = key
        self.image_resize = resize
        self.names = None


class _StringSpec:
    is_image = False
    dtype = "string"
    namespace = None
    operators = ()

    def __init__(self, key):
        self.key = key
        self.names = ()


class _VectorSpec:
    is_image = False
    namespace = None
    operators = ()

    def __init__(self, key, dtype, names):
        self.key = key
        self.dtype = dtype
        self.names = names


def test_decoded_image_channels_is_three():
    # Decoders normalize every supported encoding to 3-channel RGB.
    assert DECODED_IMAGE_CHANNELS == 3


def test_feature_and_zeros_agree_for_images():
    spec = _ImageSpec("observation.images.cam", (64, 48))
    feat = build_feature(spec)
    zeros = zero_image_for_spec(spec)
    assert feat["shape"] == (64, 48, 3)
    assert zeros.shape == (64, 48, 3)
    assert zeros.dtype == np.uint8


def test_string_feature():
    feat = build_feature(_StringSpec("task2"))
    assert feat == {"dtype": "string", "shape": (1,), "names": None}


def test_build_feature_rejects_numeric_specs():
    # Numeric features come from FrameLayout.lerobot_features() only.
    spec = _VectorSpec("observation.state", "float32", ["a", "b"])
    with pytest.raises(ValueError, match="lerobot_features"):
        build_feature(spec)


def test_image_spec_without_resize_rejected():
    with pytest.raises(ValueError, match="image_resize"):
        build_feature(_ImageSpec("observation.images.cam", None))
    with pytest.raises(ValueError, match="image_resize"):
        zero_image_for_spec(_ImageSpec("observation.images.cam", None))
