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

"""Tests for feature building and zero-fill shape agreement.

Regression guard for the image zero-fill bug: a missing image frame must
zero-fill with the same shape as a real decoded frame and the declared feature.
"""

import numpy as np

from rosetta.frames.layout import (
    DECODED_IMAGE_CHANNELS,
    build_feature,
    zeros_for_spec,
)


class _ImageSpec:
    """Duck-typed stand-in for an ObservationStreamSpec image stream."""

    is_image = True
    dtype = "video"

    def __init__(self, key, resize):
        self.key = key
        self.image_resize = resize
        self.names = None


class _VectorSpec:
    is_image = False

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
    zeros = zeros_for_spec(spec)
    assert feat["shape"] == (64, 48, 3)
    assert zeros.shape == (64, 48, 3)
    assert zeros.dtype == np.uint8


def test_vector_feature_and_zeros_agree():
    spec = _VectorSpec("observation.state", "float32", ["a", "b", "c"])
    feat = build_feature(spec)
    zeros = zeros_for_spec(spec)
    assert feat["shape"] == (3,)
    assert feat["dtype"] == "float32"
    assert zeros.shape == (3,)
    assert zeros.dtype == np.float32
