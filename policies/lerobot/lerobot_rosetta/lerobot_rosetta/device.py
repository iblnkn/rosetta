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

"""Policy-device resolution, done by the inference server on the host that runs the model."""

from collections.abc import Callable

from lerobot.utils.device_utils import is_torch_device_available


def resolve_policy_device(
    requested_device: str | None,
    is_available: Callable[[str], bool] = is_torch_device_available,
) -> str:
    """Resolve the requested policy device, falling back to CPU when its backend is unavailable.

    Availability is LeRobot's call (is_torch_device_available: cuda/mps/xpu/cpu). Devices
    LeRobot doesn't know (it raises ValueError) pass through untouched, for torch to
    accept or reject.
    """
    device = (requested_device or "").strip().lower()
    if not device:
        return "cpu"
    try:
        available = is_available(device.split(":", 1)[0])
    except ValueError:
        return device
    return device if available else "cpu"
