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

"""Tests for the image decoders (sensor_msgs Image / CompressedImage).

Raw-Image tests are fully ROS-free (duck-typed messages, plain numpy).
CompressedImage tests need cv2, which comes from the default pixi env's
lerobot feature (absent in the ``ci`` env) — they importorskip it, matching
the deliberately soft cv2 import in decoders.py.
"""

import types

import numpy as np
import pytest

from rosetta.robots.ros2.decoders import (
    DepthEncodingNotSupported,
    decode_ros_image,
)


def _image(height, width, encoding, data, step=0):
    return types.SimpleNamespace(height=height, width=width, encoding=encoding, step=step, data=data)


def _rgb_pixels(h, w):
    """An RGB test image with asymmetric channel values so swaps are caught."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = 10  # R
    img[..., 1] = 20  # G
    img[..., 2] = 30  # B
    img[0, 0] = (1, 2, 3)  # one distinct pixel pins spatial layout too
    return img


# --- raw Image: supported encodings ------------------------------------------


@pytest.mark.parametrize("encoding", ["rgb8", "bgr8", "rgba8", "bgra8", "mono8", "8uc1"])
def test_raw_image_decodes_to_rgb(encoding):
    h, w = 4, 5
    rgb = _rgb_pixels(h, w)
    if encoding == "rgb8":
        raw = rgb
    elif encoding == "bgr8":
        raw = rgb[..., ::-1]
    elif encoding == "rgba8":
        raw = np.concatenate([rgb, np.full((h, w, 1), 255, np.uint8)], axis=-1)
    elif encoding == "bgra8":
        raw = np.concatenate([rgb[..., ::-1], np.full((h, w, 1), 255, np.uint8)], axis=-1)
    else:  # mono8 / 8uc1: grayscale replicated to 3 channels
        raw = rgb[..., :1]

    out = decode_ros_image(_image(h, w, encoding, raw.tobytes()))
    expected = np.repeat(rgb[..., :1], 3, axis=-1) if encoding in ("mono8", "8uc1") else rgb
    assert out.shape == (h, w, 3)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("encoding", ["rgb8", "bgr8", "rgba8", "bgra8", "mono8"])
def test_raw_image_output_is_writable(encoding):
    # frombuffer views of msg.data are read-only; the decoder must hand
    # downstream (resize no-op path, torch.from_numpy) a writable array.
    h, w = 2, 2
    ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}[encoding]
    out = decode_ros_image(_image(h, w, encoding, bytes(h * w * ch)))
    assert out.flags.writeable


def test_raw_image_row_padding_is_stripped():
    # step > w*ch: rows carry trailing pad bytes that must not leak into pixels.
    h, w, ch, pad = 3, 2, 3, 4
    rgb = _rgb_pixels(h, w)
    rows = np.concatenate([rgb.reshape(h, w * ch), np.full((h, pad), 255, np.uint8)], axis=1)
    out = decode_ros_image(_image(h, w, "rgb8", rows.tobytes(), step=w * ch + pad))
    np.testing.assert_array_equal(out, rgb)


def test_raw_image_step_zero_means_tightly_packed():
    h, w = 2, 3
    rgb = _rgb_pixels(h, w)
    out = decode_ros_image(_image(h, w, "rgb8", rgb.tobytes(), step=0))
    np.testing.assert_array_equal(out, rgb)


# --- raw Image: rejections ----------------------------------------------------


@pytest.mark.parametrize("encoding", ["mono16", "16uc1", "32fc1", "32fc"])
def test_raw_image_depth_encoding_raises(encoding):
    with pytest.raises(DepthEncodingNotSupported):
        decode_ros_image(_image(2, 2, encoding, bytes(16)))


def test_raw_image_unsupported_encoding_raises():
    with pytest.raises(ValueError, match="Unsupported image encoding"):
        decode_ros_image(_image(2, 2, "yuv422", bytes(8)))


def test_raw_image_missing_encoding_raises():
    with pytest.raises(ValueError, match="no encoding"):
        decode_ros_image(_image(2, 2, "", bytes(12)))


# --- CompressedImage ----------------------------------------------------------


def _compressed(data, fmt):
    return types.SimpleNamespace(format=fmt, data=data)


def _decode_compressed(msg):
    from rosetta.frames.codecs import DECODERS, discover_codecs

    discover_codecs()
    return DECODERS["sensor_msgs/msg/CompressedImage"](msg, spec=None)


def test_compressed_png_default_bgr_order():
    cv2 = pytest.importorskip("cv2")
    # A publisher that imencodes a BGR matrix (the cv_bridge convention):
    # decode must return it as RGB.
    rgb = _rgb_pixels(6, 8)
    ok, buf = cv2.imencode(".png", rgb[..., ::-1])  # PNG: lossless round-trip
    assert ok
    out = _decode_compressed(_compressed(buf.tobytes(), "bgr8; png compressed bgr8"))
    np.testing.assert_array_equal(out, rgb)
    assert out.flags.writeable


def test_compressed_bare_format_defaults_to_bgr():
    cv2 = pytest.importorskip("cv2")
    rgb = _rgb_pixels(6, 8)
    ok, buf = cv2.imencode(".png", rgb[..., ::-1])
    assert ok
    for fmt in ("png", ""):
        out = _decode_compressed(_compressed(buf.tobytes(), fmt))
        np.testing.assert_array_equal(out, rgb)


def test_compressed_rgb_order_skips_swap():
    cv2 = pytest.importorskip("cv2")
    # compressed_image_transport records the source matrix order after
    # "compressed"; imdecode round-trips that order, so a declared rgb8
    # stream must NOT be swapped. This used to channel-swap silently.
    rgb = _rgb_pixels(6, 8)
    ok, buf = cv2.imencode(".png", rgb)  # publisher compressed an RGB matrix as-is
    assert ok
    out = _decode_compressed(_compressed(buf.tobytes(), "rgb8; png compressed rgb8"))
    np.testing.assert_array_equal(out, rgb)


@pytest.mark.parametrize("fmt", ["16UC1; compressedDepth png", "32FC1; compressedDepth", "16uc1; png"])
def test_compressed_depth_format_raises(fmt):
    # Depth PNGs would silently downcast to 8-bit garbage through
    # IMREAD_COLOR — rejected loudly, like the raw-Image path.
    with pytest.raises(DepthEncodingNotSupported):
        _decode_compressed(_compressed(bytes(16), fmt))


def test_compressed_garbage_bytes_raise():
    pytest.importorskip("cv2")
    with pytest.raises(ValueError, match="imdecode failed"):
        _decode_compressed(_compressed(b"not an image", "jpeg"))
