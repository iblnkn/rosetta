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

"""CudaBufferView's read handle: opened once, closed once, and only after its last user.

cuda_buffer and torch_conversions' DLPack bridge are faked, so these run
without CUDA. The fake bridge holds its owner the way the real one does (a
reference per capsule).
"""

import contextlib
import gc
import sys
import types

import pytest

from rosetta_ros import buffers


class _Handle:
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.device_ptr = 0x1000
        self.closes = 0

    def close(self):
        self.closes += 1


class _Capsule:
    def __init__(self, owner):
        self.owner = owner


@pytest.fixture
def handles(monkeypatch):
    opened = []

    def from_input_buffer(data, stream):
        handle = _Handle(getattr(data, "device_id", 0))
        opened.append(handle)
        return handle

    cuda_buffer = types.ModuleType("cuda_buffer")
    cuda_buffer.CudaBuffer = types.SimpleNamespace(from_input_buffer=from_input_buffer)
    bridge = types.ModuleType("torch_conversions._torch_bridge")
    bridge.make_dlpack_capsule = lambda *args: _Capsule(args[-1])
    monkeypatch.setitem(sys.modules, "cuda_buffer", cuda_buffer)
    monkeypatch.setitem(sys.modules, "torch_conversions", types.ModuleType("torch_conversions"))
    monkeypatch.setitem(sys.modules, "torch_conversions._torch_bridge", bridge)
    return opened


class _Data(bytes):
    backend_type = "cuda"


def test_a_view_never_exported_closes_its_handle(handles):
    view = buffers.CudaBufferView(_Data(b"abcd"))
    del view
    gc.collect()
    assert [h.closes for h in handles] == [1]


def test_two_exports_share_one_handle_and_close_it_after_the_last(handles):
    view = buffers.CudaBufferView(_Data(b"abcd"))
    first, second = view.__dlpack__(), view.__dlpack__()
    del view, first
    gc.collect()
    assert handles[0].closes == 0  # the second capsule still reads the memory
    del second
    gc.collect()
    assert [h.closes for h in handles] == [1]


def test_a_failed_import_closes_the_handle(handles, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "device", lambda idx: contextlib.nullcontext())

    def refuse(view):
        view.__dlpack__()
        raise BufferError("refused")

    monkeypatch.setattr(torch, "from_dlpack", refuse)
    with pytest.raises(BufferError):
        buffers.message_bytes(_Data(b"abcd"))
    gc.collect()
    assert [h.closes for h in handles] == [1]


def test_device_data_imports_with_its_own_device_current(handles, monkeypatch):
    torch = pytest.importorskip("torch")
    current = []

    @contextlib.contextmanager
    def device(idx):
        current.append(idx)
        yield

    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(torch, "from_dlpack", lambda view: ("tensor", current[-1]))
    data = _Data(b"abcd")
    data.device_id = 1
    assert buffers.message_bytes(data) == ("tensor", 1)
