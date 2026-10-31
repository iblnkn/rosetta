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

"""A device array with no accelerator behind it, for testing device images in core alone.

:class:`FakeDeviceArray` wraps a numpy array, so an image that stayed in device
memory is told apart from a host one by its type alone. It offers only what
device-capable code may use: ``shape``, ``dtype``, ``reshape``, slicing and
indexing, and DLPack. :class:`FakeDeviceMemory` stands in for a transport's
device buffer, which a device-capable decoder imports as a FakeDeviceArray.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class FakeDeviceArray:
    """An array in fake device memory."""

    def __init__(self, data: np.ndarray) -> None:
        self.data = data

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def reshape(self, *shape: Any) -> FakeDeviceArray:
        return FakeDeviceArray(self.data.reshape(*shape))

    def __getitem__(self, index: Any) -> FakeDeviceArray:
        return FakeDeviceArray(self.data[index])

    def __dlpack__(self, **kwargs: Any) -> Any:
        return self.data.__dlpack__(**kwargs)

    def __dlpack_device__(self) -> tuple[int, int]:
        return self.data.__dlpack_device__()


class FakeDeviceMemory:
    """A message's data in fake device memory, as a transport would hand it over."""

    def __init__(self, data: np.ndarray) -> None:
        self.data = data
