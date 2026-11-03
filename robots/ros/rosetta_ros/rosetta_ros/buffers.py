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

"""
Device-backed message data: ROS 2 rosidl::Buffer backends as DLPack producers.

On ROS 2 Lyrical and newer, a ``uint8[]`` field can arrive in device memory
when the subscription accepts a buffer backend (rclpy's
``acceptable_buffer_backends``) and the publisher wrote it there. Rosetta
imports such a field without copying it: :func:`message_bytes` wraps it in a
DLPack view and imports that as a torch tensor, a device array (see
:mod:`rosetta.arrays`) any other library can take through DLPack.

Only the ``cuda`` backend is supported, through ``cuda_buffer`` and
``torch_conversions`` from ros2/rosidl_buffer_backends, which need torch.
They aren't in RoboStack, so they're built from source;
:func:`device_memory_unavailable` says what's missing when they aren't
installed. This module imports none of them at module level, so the decoders
that use it stay importable without ROS.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import Any

import numpy as np

_KDL_CUDA = 2
_CUDA_LEGACY_STREAM = 1


def _cuda_unavailable() -> str | None:
    """What the ``cuda`` backend needs: torch with a CUDA device, cuda_buffer and torch_conversions."""
    if importlib.util.find_spec("torch") is None:
        return "'torch' is not installed"
    import torch

    if not torch.cuda.is_available():
        return f"torch {torch.__version__} has no CUDA device available"
    for module in ("cuda_buffer", "torch_conversions"):
        if importlib.util.find_spec(module) is None:
            return f"'{module}' is not installed (build it from ros2/rosidl_buffer_backends)"
    return None


# Each backend checks its own requirements, so supporting another one adds an
# entry here rather than changing what the others need.
_BACKEND_REQUIREMENTS: dict[str, Callable[[], str | None]] = {"cuda": _cuda_unavailable}

SUPPORTED_BUFFER_BACKENDS = tuple(_BACKEND_REQUIREMENTS)
"""rosidl buffer backends whose memory rosetta imports."""


def device_memory_unavailable(buffer_backend: str) -> str | None:
    """Why this process can't import ``buffer_backend`` memory, or None if it can."""
    requirements = _BACKEND_REQUIREMENTS.get(buffer_backend)
    if requirements is None:
        return f"buffer backend '{buffer_backend}' is not supported (supported: {list(SUPPORTED_BUFFER_BACKENDS)})"
    return requirements()


DLPACK_DTYPES: dict[tuple[int, int], str] = {
    (1, 8): "uint8",
    (0, 8): "int8",
    (0, 16): "int16",
    (0, 32): "int32",
    (0, 64): "int64",
    (2, 16): "float16",
    (2, 32): "float32",
    (2, 64): "float64",
    (4, 16): "bfloat16",
    (6, 8): "bool",
}
"""DLPack ``(type code, bits)`` -> dtype name, for one-lane types. The set
torch_conversions reads, with the names numpy and torch both use. numpy has no
bfloat16, so a bfloat16 value can only be read in device memory."""

_ITEMSIZE = {name: bits // 8 for (_, bits), name in DLPACK_DTYPES.items()}


def message_bytes(data: Any) -> Any:
    """A message's ``uint8[]`` field as a flat uint8 array, without copying it.

    Host data becomes a read-only numpy view. Device data becomes a torch
    tensor on its device.
    """
    if _on_host(data):
        return np.frombuffer(data, dtype=np.uint8)
    return _device_tensor(data, "uint8", (len(data),), (1,), 0)


def message_tensor(data: Any, dtype: str, shape: tuple[int, ...], strides: tuple[int, ...], byte_offset: int) -> Any:
    """A message's ``uint8[]`` field viewed as a ``dtype`` array, without copying it.

    ``strides`` are in elements and non-negative, and ``byte_offset`` is where
    the first element starts, as in a DLTensor. Raises ValueError when they
    reach past ``data``. Host data becomes a read-only numpy view. Device
    data becomes a torch tensor on its device.
    """
    if byte_offset % _ITEMSIZE[dtype]:
        # A misaligned device pointer fails in the first kernel that reads it,
        # with a CUDA error that poisons the whole process.
        raise ValueError(
            f"A {dtype} tensor's byte offset {byte_offset} is not a multiple of its {_ITEMSIZE[dtype]}-byte items"
        )
    # Elements from the first to the last one the view reaches.
    span = 1 + sum((n - 1) * s for n, s in zip(shape, strides, strict=True)) if all(shape) else 0
    if byte_offset + span * _ITEMSIZE[dtype] > len(data):
        raise ValueError(
            f"A {dtype} tensor of shape {list(shape)} and strides {list(strides)} at byte offset "
            f"{byte_offset} reads past its {len(data)} bytes of data"
        )
    if not _on_host(data):
        return _device_tensor(data, dtype, shape, strides, byte_offset)
    if dtype == "bfloat16":
        raise ValueError("bfloat16 message data can only be read in device memory")
    flat = np.frombuffer(data, dtype=dtype, count=span, offset=byte_offset)
    return np.lib.stride_tricks.as_strided(flat, shape, tuple(s * flat.itemsize for s in strides), writeable=False)


def _on_host(data: Any) -> bool:
    """True for host data, False for CUDA data; raises for memory rosetta can't read."""
    backend_type = getattr(data, "backend_type", "cpu")
    if backend_type not in ("cpu", "cuda"):
        raise ValueError(f"Message data is in '{backend_type}' memory, which rosetta can't read")
    return backend_type == "cpu"


def _device_tensor(data: Any, dtype: str, shape: tuple[int, ...], strides: tuple[int, ...], byte_offset: int) -> Any:
    import torch

    view = CudaBufferView(data, dtype=dtype, shape=shape, strides=strides, byte_offset=byte_offset)
    # The legacy default stream the handle is ordered on is per device, so the
    # import runs with the data's device current, whichever GPU it's on.
    with torch.cuda.device(view.device_id):
        return torch.from_dlpack(view)


class CudaBufferView:
    """
    A DLPack producer over a CUDA-backed rosidl ``Buffer``.

    The read handle opens on CUDA's legacy default stream, which waits for the
    publisher's write. Rosetta decodes on threads that use that stream, and an
    import onto any other stream is refused rather than left unordered. The
    handle stays open until the view and the last array over its memory are
    gone, which keeps the publisher from reusing the block underneath it. One
    lease owns the handle: every export shares it, and a view that is never
    exported, or whose import fails, closes the handle when it's freed.

    The view is a flat uint8 array of the whole buffer unless ``dtype``,
    ``shape``, ``strides`` (in elements) and ``byte_offset`` describe a
    tensor inside it. The offset is folded into the data pointer, as
    torch_conversions does, so no consumer has to honor a DLPack byte_offset.

    TODO: replace with cuda_buffer's own ``__dlpack__`` once
    ros2/rosidl_buffer_backends exposes one on its read handle.
    """

    def __init__(
        self,
        data: Any,
        *,
        dtype: str = "uint8",
        shape: tuple[int, ...] | None = None,
        strides: tuple[int, ...] | None = None,
        byte_offset: int = 0,
    ) -> None:
        from cuda_buffer import CudaBuffer

        self._dtype = next(key for key, name in DLPACK_DTYPES.items() if name == dtype)
        self._shape = list(shape) if shape is not None else [len(data)]
        self._strides = list(strides) if strides is not None else [1]
        self._byte_offset = byte_offset
        self._handle = CudaBuffer.from_input_buffer(data, _CUDA_LEGACY_STREAM)
        self._lease = _ReadLease(self._handle, data)

    @property
    def device_id(self) -> int:
        return self._handle.device_id

    def __dlpack_device__(self) -> tuple[int, int]:
        return (_KDL_CUDA, self._handle.device_id)

    def __dlpack__(self, *, stream: int | None = None, max_version: Any = None, **_: Any) -> Any:
        if stream not in (None, _CUDA_LEGACY_STREAM):
            raise BufferError(
                f"CUDA message data is ordered on the legacy default stream; can't export to stream {stream}"
            )
        # Upstream's DLPack builder. The module is torch_conversions' own
        # torch-free bridge, though importing it imports the package's torch.
        from torch_conversions._torch_bridge import make_dlpack_capsule

        code, bits = self._dtype
        return make_dlpack_capsule(
            self._handle.device_ptr + self._byte_offset, _KDL_CUDA, self._handle.device_id,
            code, bits, 1,  # DLPack type code, bits, lanes
            self._shape, self._strides, 0, self._lease,
        )  # fmt: skip


class _ReadLease:
    """Closes the read handle when the last array over its memory is freed."""

    def __init__(self, handle: Any, data: Any) -> None:
        self._handle = handle
        self._data = data

    def __del__(self) -> None:
        self._handle.close()
