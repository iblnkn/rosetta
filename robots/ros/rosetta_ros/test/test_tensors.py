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

Most messages are SimpleNamespaces carrying a DLTensor's fields, so those
tests run without tensor_msgs installed, and torch tensors stand in for device
memory, on the CPU and, where available, on CUDA. The rest need the real
packages, built from ros2/rosidl_buffer_backends: a deserialized message and
a contract naming the type need tensor_msgs, and the parity cases compare
CUDA-published messages against torch_conversions itself.
"""

import importlib.util
import types

import numpy as np
import pytest
from codec_helpers import require_message

from rosetta.contract.schema import load_contract
from rosetta.contract.specs import iter_observation_specs
from rosetta.frames.arrays import to_host
from rosetta.frames.codecs import host_only_reason
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


# ---------- real messages (tensor_msgs from ros2/rosidl_buffer_backends) ----------

_TENSOR = "tensor_msgs/msg/ExperimentalTensor"


def _real_message(arr, **fields):
    require_message(_TENSOR)
    from tensor_msgs.msg import ExperimentalTensor

    code, bits = _CODES[arr.dtype.name]
    msg = ExperimentalTensor(dtype_code=code, dtype_bits=bits, dtype_lanes=1, shape=list(arr.shape), **fields)
    msg.data = arr.tobytes()
    return msg


def test_deserialized_message_decodes_like_the_stand_in():
    from rclpy.serialization import deserialize_message, serialize_message

    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    msg = _real_message(arr)
    received = deserialize_message(serialize_message(msg), type(msg))
    np.testing.assert_array_equal(decode_tensor(received, (3, 4), "float32"), arr)


def test_contract_with_a_tensor_stream_loads_on_the_receive_timeline(tmp_path):
    require_message(_TENSOR)
    text = f"""
robot_type: test
robot_interface: ros2
fps: 30
observations:
  observation.features:
    channel: {{topic: /features, type: {_TENSOR}, dtype: float16}}
    align: {{strategy: hold, timeline: receive}}
    shape: [4, 2, 2]
"""
    path = tmp_path / "c.yaml"
    path.write_text(text)
    (spec,) = iter_observation_specs(load_contract(path))
    assert spec.shape == (4, 2, 2)
    assert spec.dtype == "float16"
    assert host_only_reason(spec) is None


# ---------- parity with torch_conversions, in CUDA memory ----------

_HAS_TORCH_CONVERSIONS = importlib.util.find_spec("torch_conversions") is not None
_HAS_CUDA_BUFFER = importlib.util.find_spec("cuda_buffer") is not None


def _cuda_message(torch, flat_bytes, *, dtype_code, dtype_bits, shape, strides=(), byte_offset=0):
    """An ExperimentalTensor whose data a CUDA publisher wrote."""
    require_message(_TENSOR)
    from cuda_buffer import CudaBuffer
    from tensor_msgs.msg import ExperimentalTensor

    src = torch.from_numpy(np.frombuffer(flat_bytes, np.uint8).copy()).cuda()
    torch.cuda.synchronize()
    msg = ExperimentalTensor(
        dtype_code=dtype_code,
        dtype_bits=dtype_bits,
        dtype_lanes=1,
        shape=list(shape),
        strides=list(strides),
        byte_offset=byte_offset,
    )
    msg.data = CudaBuffer.allocate_buffer(src.numel())
    with CudaBuffer.from_output_buffer(msg.data, 1) as handle:
        CudaBuffer.to_buffer(src.data_ptr(), src.numel(), handle, 1)
    return msg


@pytest.mark.skipif(
    not (_HAS_TORCH_CONVERSIONS and _HAS_CUDA_BUFFER),
    reason="needs torch_conversions and cuda_buffer from ros2/rosidl_buffer_backends",
)
@pytest.mark.parametrize("dtype", sorted(_CODES))
@pytest.mark.parametrize(
    ("shape", "strides", "byte_offset_elements"),
    [((4, 6), (), 0), ((2, 3), (6, 2), 7), ((3,), (4,), 1), ((2, 1, 3), (), 5)],
    ids=["contiguous", "strided-offset", "every-fourth", "offset-3d"],
)
def test_cuda_tensor_matches_torch_conversions(dtype, shape, strides, byte_offset_elements):
    torch = _torch("cuda")
    import torch_conversions

    base = np.arange(24) % 2 if dtype == "bool" else np.arange(24)
    flat = base.astype(dtype)
    code, bits = _CODES[dtype]
    msg = _cuda_message(
        torch,
        flat.tobytes(),
        dtype_code=code,
        dtype_bits=bits,
        shape=shape,
        strides=strides,
        byte_offset=byte_offset_elements * flat.itemsize,
    )
    ours = decode_tensor(msg, shape, dtype)
    theirs = torch_conversions.from_input_tensor_msg(msg)
    assert ours.is_cuda
    assert ours.dtype == theirs.dtype
    torch.testing.assert_close(ours, theirs, rtol=0, atol=0)


@pytest.mark.skipif(
    not (_HAS_TORCH_CONVERSIONS and _HAS_CUDA_BUFFER),
    reason="needs torch_conversions and cuda_buffer from ros2/rosidl_buffer_backends",
)
def test_cuda_bfloat16_tensor_matches_torch_conversions_after_the_cast():
    torch = _torch("cuda")
    import torch_conversions

    values = torch.arange(12, dtype=torch.bfloat16)
    msg = _cuda_message(torch, values.view(torch.uint8).numpy().tobytes(), dtype_code=4, dtype_bits=16, shape=(3, 4))
    ours = decode_tensor(msg, (3, 4), "float32")
    theirs = torch_conversions.from_input_tensor_msg(msg).to(torch.float32)
    torch.testing.assert_close(ours, theirs, rtol=0, atol=0)


class _DeviceData(bytes):
    """Message data that says it lives in CUDA memory, for checks that run before any device import."""

    backend_type = "cuda"


def test_out_of_range_layouts_are_refused_before_touching_device_memory():
    from rosetta_ros.buffers import message_tensor

    with pytest.raises(ValueError, match="reads past its 24 bytes"):
        message_tensor(_DeviceData(24), "float32", (2, 3), (3, 1), 4)


def test_tensor_on_an_image_key_decodes_full_resolution_then_resizes(tmp_path):
    """An image key's shape and dtype are the resized 'video' frame; the decoder hands over the native RGB image."""
    require_message(_TENSOR)
    from rosetta.frames.codecs import decode_value

    text = f"""
robot_type: test
robot_interface: ros2
fps: 30
observations:
  observation.images.cam:
    channel: {{topic: /cam, type: {_TENSOR}}}
    align: {{strategy: hold, timeline: receive}}
    apply: [{{resize: [4, 6]}}]
"""
    path = tmp_path / "c.yaml"
    path.write_text(text)
    (spec,) = iter_observation_specs(load_contract(path))
    assert (spec.shape, spec.dtype) == ((4, 6, 3), "video")
    image = (np.arange(8 * 10 * 3) % 251).astype(np.uint8).reshape(8, 10, 3)
    out = to_host(decode_value(_message(image), spec))
    assert out.shape == (4, 6, 3)
    assert out.dtype == np.uint8


def test_tensor_on_an_image_key_must_be_rgb(tmp_path):
    require_message(_TENSOR)
    from rosetta.frames.codecs import decode_value

    path = tmp_path / "c.yaml"
    path.write_text(f"""
robot_type: test
robot_interface: ros2
fps: 30
observations:
  observation.images.cam:
    channel: {{topic: /cam, type: {_TENSOR}}}
    align: {{strategy: hold, timeline: receive}}
    apply: [{{resize: [4, 6]}}]
""")
    (spec,) = iter_observation_specs(load_contract(path))
    with pytest.raises(ValueError, match="RGB"):
        decode_value(_message(np.zeros((8, 10), np.uint8)), spec)
