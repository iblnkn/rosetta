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
:mod:`rosetta.frames.arrays`) any other library can take through DLPack.

Only the ``cuda`` backend is supported, through ``cuda_buffer`` and
``torch_conversions`` from ros2/rosidl_buffer_backends, which need torch.
They aren't in RoboStack, so they're built from source;
:func:`device_memory_unavailable` says what's missing when they aren't
installed. This module imports none of them at module level, so the decoders
that use it stay importable without ROS.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

_KDL_CUDA = 2
_CUDA_LEGACY_STREAM = 1

SUPPORTED_BUFFER_BACKENDS = ("cuda",)
"""rosidl buffer backends whose memory rosetta imports."""


def device_memory_unavailable(buffer_backend: str) -> str | None:
    """Why this process can't import ``buffer_backend`` memory, or None if it can."""
    if buffer_backend not in SUPPORTED_BUFFER_BACKENDS:
        return f"buffer backend '{buffer_backend}' is not supported (supported: {list(SUPPORTED_BUFFER_BACKENDS)})"
    if importlib.util.find_spec("torch") is None:
        return "'torch' is not installed"
    for module in ("cuda_buffer", "torch_conversions"):
        if importlib.util.find_spec(module) is None:
            return f"'{module}' is not installed (build it from ros2/rosidl_buffer_backends)"
    return None


def message_bytes(data: Any) -> Any:
    """A message's ``uint8[]`` field as a flat uint8 array, without copying it.

    Host data becomes a read-only numpy view. Device data becomes a torch
    tensor on its device.
    """
    backend_type = getattr(data, "backend_type", "cpu")
    if backend_type == "cpu":
        return np.frombuffer(data, dtype=np.uint8)
    if backend_type != "cuda":
        raise ValueError(f"Message data is in '{backend_type}' memory, which rosetta can't read")
    import torch

    view = CudaBufferView(data)
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

    TODO: replace with cuda_buffer's own ``__dlpack__`` once
    ros2/rosidl_buffer_backends exposes one on its read handle.
    """

    def __init__(self, data: Any) -> None:
        from cuda_buffer import CudaBuffer

        self._size = len(data)
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

        return make_dlpack_capsule(
            self._handle.device_ptr, _KDL_CUDA, self._handle.device_id,
            1, 8, 1,  # uint8: DLPack type code, bits, lanes
            [self._size], [1], 0, self._lease,
        )  # fmt: skip


class _ReadLease:
    """Closes the read handle when the last array over its memory is freed."""

    def __init__(self, handle: Any, data: Any) -> None:
        self._handle = handle
        self._data = data

    def __del__(self) -> None:
        self._handle.close()
