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

"""tensor_msgs/ExperimentalTensor: read as torch_conversions reads it, on the host and in device memory.

Messages are SimpleNamespaces carrying a DLTensor's fields, so these run
without tensor_msgs installed. The device cases stand torch tensors in for
device memory, on the CPU and, where available, on CUDA.
"""

import types

import numpy as np
import pytest

from rosetta.frames.arrays import to_host
from rosetta_ros import decoders
from rosetta_ros.decoders import TensorDataMissing, decode_tensor

_CODES = {
    "uint8": (1, 8),
    "int16": (0, 16),
    "int32": (0, 32),
    "float16": (2, 16),
    "float32": (2, 32),
    "float64": (2, 64),
    "bool": (6, 8),
}


def _message(arr, *, strides=None, byte_offset=0, buffer=None):
    """An ExperimentalTensor over ``arr``: contiguous unless ``strides`` (elements) and ``buffer`` say otherwise."""
    code, bits = _CODES[arr.dtype.name]
    data = buffer if buffer is not None else arr.tobytes()
    return types.SimpleNamespace(
        dtype_code=code,
        dtype_bits=bits,
        dtype_lanes=1,
        shape=list(arr.shape),
        strides=list(strides or []),
        byte_offset=byte_offset,
        data=data,
    )


@pytest.mark.parametrize("dtype", sorted(_CODES))
def test_contiguous_tensor_decodes_to_its_values(dtype):
    arr = (np.arange(24).reshape(2, 3, 4) % 2 if dtype == "bool" else np.arange(24).reshape(2, 3, 4)).astype(dtype)
    out = decode_tensor(_message(arr), (2, 3, 4), dtype)
    assert out.dtype == arr.dtype
    np.testing.assert_array_equal(out, arr)


def test_tensor_is_cast_to_the_stream_dtype():
    arr = np.arange(6, dtype=np.float16).reshape(2, 3)
    out = decode_tensor(_message(arr), (2, 3), "float32")
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, arr.astype(np.float32))


def test_strided_view_at_a_byte_offset_reads_the_right_elements():
    base = np.arange(40, dtype=np.float32).reshape(5, 8)
    view = base[1:4, ::2]  # rows 1-3, every other column
    msg = _message(view, strides=[8, 2], byte_offset=8 * 4, buffer=base.tobytes())
    np.testing.assert_array_equal(decode_tensor(msg, (3, 4), "float32"), view)


def test_host_tensor_is_a_copy_of_the_message_data():
    arr = np.arange(4, dtype=np.float32)
    out = decode_tensor(_message(arr), (4,), "float32")
    assert out.flags.writeable
    assert out.base is None or not isinstance(out.base, (bytes, memoryview))


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"shape": [2, 2]}, "the stream declares"),
        ({"dtype_lanes": 2}, "not supported"),
        ({"dtype_code": 4, "dtype_bits": 16}, "only be read in device memory"),
        ({"dtype_code": 9}, "not supported"),
        ({"strides": [1]}, "1 strides for 2 dimensions"),
        ({"strides": [-3, 1]}, "negative strides"),
        ({"byte_offset": 4}, "reads past"),
    ],
)
def test_malformed_tensors_are_refused(change, match):
    msg = _message(np.zeros((2, 3), np.float32))
    for field, value in change.items():
        setattr(msg, field, value)
    with pytest.raises(ValueError, match=match):
        decode_tensor(msg, (2, 3), "float32")


def test_empty_tensor_data_is_named():
    msg = _message(np.zeros((2, 3), np.float32))
    msg.data = b""
    with pytest.raises(TensorDataMissing, match="memory was recycled"):
        decode_tensor(msg, (2, 3), "float32")


def _torch(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    return torch


@pytest.fixture
def device_memory(monkeypatch):
    """Make message_tensor return torch tensors on ``device``, as CUDA-backed message data would."""

    def install(torch, device):
        host = decoders.message_tensor

        def on_device(data, dtype, shape, strides, byte_offset):
            view = host(data, dtype, shape, strides, byte_offset)
            return torch.from_numpy(np.array(view)).to(device)

        monkeypatch.setattr(decoders, "message_tensor", on_device)

    return install


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_tensor_stays_on_its_device(device_memory, device):
    torch = _torch(device)
    device_memory(torch, device)
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = decode_tensor(_message(arr), (2, 3), "float32")
    assert isinstance(out, torch.Tensor)
    assert out.device.type == device
    np.testing.assert_array_equal(to_host(out), arr)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_tensor_is_cast_on_its_device(device_memory, device):
    torch = _torch(device)
    device_memory(torch, device)
    arr = np.arange(6, dtype=np.float16).reshape(2, 3)
    out = decode_tensor(_message(arr), (2, 3), "float32")
    assert out.dtype == torch.float32
    assert out.device.type == device
    np.testing.assert_array_equal(to_host(out), arr.astype(np.float32))
