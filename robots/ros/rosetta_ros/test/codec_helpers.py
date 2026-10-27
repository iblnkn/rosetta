# Copyright 2026 Isaac Blankenau
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

"""Helpers shared by the codec tests."""

import pytest
from rosidl_runtime_py.utilities import get_message


def require_message(msg_type: str) -> None:
    """Skip the test when ``msg_type``'s package is not installed (e.g. control_msgs).

    Contract load introspects timelines via rosidl, and encoders build real
    messages, so both need the message package.
    """
    try:
        get_message(msg_type)
    except Exception:
        pytest.skip(f"{msg_type} message package unavailable")
