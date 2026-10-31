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

"""Images in device memory: same bytes as on the host, and only where every step allows it.

The decoder and resize are checked on torch tensors standing in for device
memory, on the CPU and, where available, on CUDA, so they run wherever torch
is installed. The end-to-end case publishes CUDA-backed images on the RMW under test, which
must support buffer backends (rmw_fastrtps_cpp, rmw_zenoh_cpp 0.10.5+), and
needs CUDA plus cuda_buffer from ros2/rosidl_buffer_backends.
"""

import importlib.util
import types

import numpy as np
import pytest
from bridge_helpers import FakeLogger, bridge_on_graph, spin_until

from rosetta.arrays import to_host
from rosetta.codecs import DECODERS
from rosetta.contract.interfaces import discover_codecs
from rosetta.contract.schema import Align, Channel, Source
from rosetta.contract.specs import ObservationStreamSpec
from rosetta.operators import OperatorContext, build_operator
from rosetta_ros import decoders, topic_bridge
from rosetta_ros.buffers import message_bytes
from rosetta_ros.decoders import ImageDataMissing
from rosetta_ros.topic_bridge import TopicBridge


def _torch(device="cpu"):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    return torch


def _image(h, w, encoding, data, step=0):
    return types.SimpleNamespace(height=h, width=w, encoding=encoding, step=step, data=data)


def _decode(msg):
    discover_codecs()
    return DECODERS["sensor_msgs/msg/Image"](msg, spec=None)


def _spec(*, topic="/cam", decoder=None, resize=(4, 6)):
    operators = (build_operator("resize", list(resize), OperatorContext(is_image=True)),)
    return ObservationStreamSpec(
        key="observation.images.cam",
        names=(),
        fps=30,
        source=Source(
            channel=Channel(topic=topic, type="sensor_msgs/msg/Image", decoder=decoder), align=Align("hold", "receive")
        ),
        dtype="video",
        operators=operators,
        is_image=True,
        image_resize=tuple(resize),
    )


# ---------- decode ----------


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("encoding", ["rgb8", "bgr8", "rgba8", "bgra8", "mono8", "8uc1"])
@pytest.mark.parametrize("pad", [0, 5])
def test_device_memory_decodes_to_the_host_bytes(monkeypatch, device, encoding, pad):
    torch = _torch(device)
    h, w = 7, 9
    ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(encoding, 1)
    raw = np.random.default_rng(2).integers(0, 256, h * (w * ch + pad), dtype=np.uint8).tobytes()
    msg = _image(h, w, encoding, raw, step=w * ch + pad)
    host = _decode(msg)
    monkeypatch.setattr(
        decoders, "message_bytes", lambda data: torch.from_numpy(np.frombuffer(data, np.uint8).copy()).to(device)
    )
    out = _decode(msg)
    assert isinstance(out, torch.Tensor)
    assert out.device.type == device
    np.testing.assert_array_equal(to_host(out), host)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(("shape", "size"), [((5, 7, 3), (2, 3)), ((480, 640, 3), (240, 240)), ((3, 4), (9, 2))])
def test_device_resize_picks_the_host_pixels(device, shape, size):
    torch = _torch(device)
    img = np.random.default_rng(1).integers(0, 256, size=shape, dtype=np.uint8)
    resize = build_operator("resize", list(size), OperatorContext(is_image=True))
    out = resize.forward(torch.from_numpy(img).to(device))
    assert out.device.type == device
    np.testing.assert_array_equal(to_host(out), resize.forward(img))


def test_empty_image_data_is_named():
    with pytest.raises(ImageDataMissing, match="taken after its memory was recycled"):
        _decode(_image(2, 2, "rgb8", b""))


def test_host_data_is_a_numpy_view():
    out = message_bytes(bytes(range(4)))
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, [0, 1, 2, 3])


def test_unknown_device_memory_is_refused():
    data = types.SimpleNamespace(backend_type="qc")
    with pytest.raises(ValueError, match="'qc' memory"):
        message_bytes(data)


# ---------- which subscriptions accept device memory ----------


@pytest.fixture
def cuda_importable(monkeypatch):
    monkeypatch.setattr(topic_bridge, "device_memory_unavailable", lambda backend: None)


def test_no_buffer_backends_means_host_memory_only(cuda_importable):
    assert TopicBridge([], [], fps=30)._accepted_buffer_backends(_spec(), FakeLogger()) == ()


def test_device_capable_image_stream_accepts_the_backend(cuda_importable):
    logger = FakeLogger()
    accepted = TopicBridge([], [], fps=30, buffer_backends=("cuda",))._accepted_buffer_backends(_spec(), logger)
    assert accepted == ("cuda",)
    assert any("accepts cuda memory" in m for m in logger.infos)


def test_stream_with_a_host_only_step_takes_host_memory(cuda_importable):
    logger = FakeLogger()
    spec = _spec(decoder="rosetta_ros.decoders:decode_ros_image")
    bridge = TopicBridge([], [], fps=30, buffer_backends=("cuda",))
    assert bridge._accepted_buffer_backends(spec, logger) == ()
    assert any("takes host memory: inline decoder" in m for m in logger.infos)


def test_backend_this_process_cant_import_is_skipped_with_a_warning(monkeypatch):
    monkeypatch.setattr(topic_bridge, "device_memory_unavailable", lambda backend: "'cuda_buffer' is not installed")
    logger = FakeLogger()
    bridge = TopicBridge([], [], fps=30, buffer_backends=("cuda",))
    assert bridge._accepted_buffer_backends(_spec(), logger) == ()
    assert any("won't accept cuda memory: 'cuda_buffer' is not installed" in m for m in logger.warnings)


# ---------- end to end, on a GPU ----------

_HAS_CUDA_BUFFER = importlib.util.find_spec("cuda_buffer") is not None


@pytest.mark.skipif(not _HAS_CUDA_BUFFER, reason="needs cuda_buffer from ros2/rosidl_buffer_backends")
def test_cuda_published_image_reaches_the_frame_in_gpu_memory():
    torch = _torch("cuda")
    from cuda_buffer import CudaBuffer
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image

    h, w = 48, 64
    bgr = np.random.default_rng(3).integers(0, 256, (h, w, 3), dtype=np.uint8)
    spec = _spec(topic="/device_cam", resize=(24, 32))
    with bridge_on_graph("device_images", [spec], [], buffer_backends=("cuda",)) as (bridge, node, fixture, executor):
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        pub = fixture.create_publisher(Image, "/device_cam", qos)
        assert spin_until(executor, lambda: pub.get_subscription_count() > 0)

        src = torch.from_numpy(bgr.reshape(-1).copy()).cuda()
        torch.cuda.synchronize()
        msg = Image(height=h, width=w, encoding="bgr8", step=w * 3)
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.data = CudaBuffer.allocate_buffer(src.numel())
        with CudaBuffer.from_output_buffer(msg.data, 1) as handle:
            CudaBuffer.to_buffer(src.data_ptr(), src.numel(), handle, 1)
        pub.publish(msg)

        assert spin_until(executor, lambda: bridge.warmed_up), "the CUDA-backed image never arrived"
        frame = bridge.sample_frame()["observation.images.cam"]
        assert isinstance(frame, torch.Tensor)
        assert frame.is_cuda
        host = _decode(_image(h, w, "bgr8", bgr.tobytes()))
        expected = spec.operators[0].forward(host)
        np.testing.assert_array_equal(to_host(frame), expected)


_HAS_TENSOR_MSGS = importlib.util.find_spec("tensor_msgs") is not None


@pytest.mark.skipif(
    not (_HAS_CUDA_BUFFER and _HAS_TENSOR_MSGS),
    reason="needs cuda_buffer and tensor_msgs from ros2/rosidl_buffer_backends",
)
def test_cuda_published_tensor_reaches_the_frame_in_gpu_memory():
    torch = _torch("cuda")
    from cuda_buffer import CudaBuffer
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from tensor_msgs.msg import ExperimentalTensor

    features = np.random.default_rng(4).standard_normal((4, 6, 8)).astype(np.float16)
    spec = ObservationStreamSpec(
        key="observation.features",
        names=(),
        fps=30,
        source=Source(
            channel=Channel(topic="/device_features", type="tensor_msgs/msg/ExperimentalTensor"),
            align=Align("hold", "receive"),
            shape=features.shape,
        ),
        dtype="float32",
        image_resize=None,
    )
    with bridge_on_graph("device_tensors", [spec], [], buffer_backends=("cuda",)) as (bridge, _, fixture, executor):
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        pub = fixture.create_publisher(ExperimentalTensor, "/device_features", qos)
        assert spin_until(executor, lambda: pub.get_subscription_count() > 0)

        src = torch.from_numpy(features.reshape(-1).view(np.uint8).copy()).cuda()
        torch.cuda.synchronize()
        msg = ExperimentalTensor(dtype_code=2, dtype_bits=16, dtype_lanes=1, shape=list(features.shape))
        msg.data = CudaBuffer.allocate_buffer(src.numel())
        with CudaBuffer.from_output_buffer(msg.data, 1) as handle:
            CudaBuffer.to_buffer(src.data_ptr(), src.numel(), handle, 1)
        pub.publish(msg)

        assert spin_until(executor, lambda: bridge.warmed_up), "the CUDA-backed tensor never arrived"
        frame = bridge.sample_frame()["observation.features"]
        assert isinstance(frame, torch.Tensor)
        assert frame.is_cuda
        assert frame.dtype == torch.float32
        np.testing.assert_array_equal(to_host(frame), features.astype(np.float32))
