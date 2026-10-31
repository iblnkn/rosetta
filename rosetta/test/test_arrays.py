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

"""Device images: an image delivered in device memory stays there through decode, resize and assembly.

FakeDeviceArray (fake_arrays.py) stands in for an accelerator's array, so an
image that stayed in device memory is told apart from a host one by its type.
"""

import numpy as np
import pytest
from fake_arrays import FakeDeviceArray, FakeDeviceMemory

from rosetta.contract.operators import Invertibility, Operator, OperatorContext, build_operator, register_operator
from rosetta.contract.schema import Align, Channel, Source
from rosetta.contract.specs import ObservationStreamSpec
from rosetta.frames.arrays import is_device_array, take, to_host
from rosetta.frames.codecs import decode_value, host_only_reason, register_decoder
from rosetta.frames.layout import FrameLayout

pytestmark = pytest.mark.usefixtures("scratch_registries", "fake_interface")

_IMG = "_test_msgs/msg/Image"
_HOST_IMG = "_test_msgs/msg/HostImage"
_VEC = "_test_msgs/msg/Vector"
_RESIZE = (("resize", (2, 3)),)


def _image(h=4, w=6):
    return np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)


@pytest.fixture
def decoders():
    """A device-capable image decoder, a host-only one, and a numeric one.

    The device-capable decoder imports a FakeDeviceMemory message as a device
    array, the way a robot interface imports a device-backed message, and
    passes numpy through.
    """

    @register_decoder(_IMG, dtype="video", device_capable=True)
    def _device_decoder(msg, spec):
        return FakeDeviceArray(msg.data) if isinstance(msg, FakeDeviceMemory) else msg

    @register_decoder(_HOST_IMG, dtype="video")
    def _host_decoder(msg, spec):
        return msg

    @register_decoder(_VEC, dtype="float64")
    def _vector_decoder(msg, spec):
        return np.asarray(msg, dtype=np.float64)


def _spec(msg_type=_IMG, *, ops=_RESIZE, is_image=True, decoder=None):
    ctx = OperatorContext(is_image=is_image)
    operators = tuple(build_operator(name, list(args) if args else args, ctx) for name, args in ops)
    return ObservationStreamSpec(
        key="observation.images.cam" if is_image else "observation.state",
        names=() if is_image else ("a",),
        fps=30,
        source=Source(channel=Channel(topic="/cam", type=msg_type, decoder=decoder), align=Align("hold", "receive")),
        dtype="video" if is_image else "float64",
        operators=operators,
        is_image=is_image,
        image_resize=operators[-1].output_hw if is_image and operators else None,
    )


@register_operator("_test_host_only", kind=Invertibility.FORWARD_ONLY)
class _HostOnly(Operator):
    def forward(self, arr):
        return arr


def _inline(msg, spec):
    return msg


# ---------- arrays ----------


def test_only_another_librarys_dlpack_array_is_a_device_array():
    assert is_device_array(FakeDeviceArray(np.zeros(1)))
    assert not is_device_array(np.zeros(1))
    assert not is_device_array([1, 2])


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_take_matches_np_take_on_numpy_and_device_arrays(axis):
    img = _image()
    idx = np.array([2, 0, 1])
    host = take(img, idx, axis)
    device = take(FakeDeviceArray(img), idx, axis)
    np.testing.assert_array_equal(host, np.take(img, idx, axis=axis))
    assert isinstance(device, FakeDeviceArray)
    np.testing.assert_array_equal(device.data, host)


def test_to_host_passes_numpy_through():
    arr = np.arange(3)
    assert to_host(arr) is arr


def test_to_host_brings_a_device_array_to_numpy():
    host = to_host(FakeDeviceArray(np.arange(3)))
    assert isinstance(host, np.ndarray)
    np.testing.assert_array_equal(host, np.arange(3))


def test_to_host_refuses_values_that_are_not_arrays():
    with pytest.raises(TypeError, match="to_host got list"):
        to_host([1, 2])


# ---------- resize ----------


@pytest.mark.parametrize("shape", [(5, 7), (5, 7, 3), (480, 640, 3)])
@pytest.mark.parametrize("size", [(2, 3), (11, 13), (240, 240)])
def test_resize_picks_the_same_pixels_on_the_device(shape, size):
    img = np.random.default_rng(0).integers(0, 256, size=shape, dtype=np.uint8)
    resize = build_operator("resize", list(size), OperatorContext(is_image=True))
    host = resize.forward(img)
    device = resize.forward(FakeDeviceArray(img))
    assert isinstance(host, np.ndarray)
    assert isinstance(device, FakeDeviceArray)
    np.testing.assert_array_equal(device.data, host)


# ---------- host_only_reason ----------


def test_device_capable_image_stream_can_take_device_memory(decoders):
    assert host_only_reason(_spec()) is None


@pytest.mark.parametrize(
    ("spec", "reason"),
    [
        (lambda: _spec(_VEC, ops=(), is_image=False), "only image streams"),
        (lambda: _spec(decoder="test_arrays:_inline"), "inline decoder"),
        (lambda: _spec(_HOST_IMG), "decoder runs on the host"),
        (lambda: _spec(ops=(("_test_host_only", None), *_RESIZE)), "_test_host_only run on the host"),
    ],
    ids=["numeric", "inline-decoder", "host-decoder", "host-operator"],
)
def test_host_only_reason_names_the_blocker(decoders, spec, reason):
    assert reason in host_only_reason(spec())


# ---------- decode_value ----------


def test_host_message_decodes_to_numpy(decoders):
    assert isinstance(decode_value(_image(), _spec()), np.ndarray)


def test_device_message_decodes_and_resizes_in_device_memory(decoders):
    value = decode_value(FakeDeviceMemory(_image()), _spec())
    assert isinstance(value, FakeDeviceArray)
    np.testing.assert_array_equal(value.data, decode_value(_image(), _spec()))


def test_device_message_on_a_host_only_operator_is_refused(decoders):
    spec = _spec(ops=(("_test_host_only", None), *_RESIZE))
    with pytest.raises(ValueError, match="_test_host_only run on numpy only"):
        decode_value(FakeDeviceMemory(_image()), spec)


# ---------- FrameLayout ----------


def test_layout_passes_device_images_through():
    img = FakeDeviceArray(_image(2, 3))
    assert FrameLayout([_spec()]).assemble([img])["observation.images.cam"] is img


def test_layout_refuses_a_non_uint8_device_image():
    img = FakeDeviceArray(np.zeros((2, 3, 3), np.float32))
    with pytest.raises(ValueError, match="dtype float32, not uint8"):
        FrameLayout([_spec()]).assemble([img])


def test_layout_zero_fills_missing_images_with_numpy():
    zeros = FrameLayout([_spec()]).assemble([None])["observation.images.cam"]
    assert isinstance(zeros, np.ndarray)
    assert zeros.shape == (2, 3, 3)
