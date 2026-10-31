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
Device arrays: images a transport delivered in device memory.

A frame carries DLPack arrays. Host arrays are numpy, the one library rosetta
creates arrays with: message bytes, decoded images, zero-filled images and
numeric vectors. An image whose data a transport delivered in device memory
(a ROS 2 message whose data is CUDA-backed, say) is whatever array the robot
interface imported it as, and stays on its device through decode and the
contract's operators. Rosetta never imports that array's library. It reads
the array through DLPack, its shape and dtype, basic slicing and indexing.

Rosetta's device work runs on the device's legacy default stream, so a host
copy, or any later work on that stream from any thread, sees it complete. A
consumer on another stream imports the array with its library's
``from_dlpack``, whose stream handshake orders it after that work.

Moving frames to the device a model runs on is the framework's job, as
normalization and batching are. A framework adapter whose API takes host
arrays brings each image to the host with :func:`to_host`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["is_device_array", "take", "to_host"]


def is_device_array(arr: Any) -> bool:
    """True if ``arr`` is another library's array rather than numpy: one a transport delivered."""
    return not isinstance(arr, np.ndarray) and hasattr(arr, "__dlpack__")


def take(arr: Any, indices: np.ndarray, axis: int) -> Any:
    """Gather ``indices`` along ``axis`` of any array, like ``np.take``. Returns a new array.

    Indices are host numpy, so index math stays in one place and a gather
    picks the same elements on every device, byte for byte. Device arrays
    gather by standard indexing. numpy uses ``np.take``, which is faster
    than indexing on camera-sized images.
    """
    if isinstance(arr, np.ndarray):
        return np.take(arr, indices, axis=axis)
    return arr[(slice(None),) * axis + (indices,)]


def to_host(arr: Any) -> np.ndarray:
    """``arr`` as a numpy array: numpy as is, a device array copied to the host.

    Raises TypeError if ``arr`` is not an array.
    """
    if isinstance(arr, np.ndarray):
        return arr
    if not hasattr(arr, "__dlpack__"):
        raise TypeError(f"to_host got {type(arr).__name__}, which is not a DLPack array")
    return np.from_dlpack(arr, device="cpu")
