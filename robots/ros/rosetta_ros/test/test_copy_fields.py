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

"""Each feedback/result message the nodes fill with copy_fields mirrors its source.

A field added to a ``*.Feedback`` but not to the matching status message (or
with a different type) would break feedback at runtime; this catches the
interface drift at test time.
"""

import pytest

from rosetta_interfaces.action import ManageEpisode, RecordEpisode, RunPolicy
from rosetta_interfaces.msg import (
    ManageEpisodeStatus,
    RecordEpisodeResult,
    RecordEpisodeStatus,
    RunPolicyResult,
    RunPolicyStatus,
)
from rosetta_ros.node_utils import copy_fields


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        (RecordEpisodeStatus, RecordEpisode.Feedback),
        (RunPolicyStatus, RunPolicy.Feedback),
        (ManageEpisodeStatus, ManageEpisode.Feedback),
        (RecordEpisode.Result, RecordEpisodeResult),
        (RunPolicy.Result, RunPolicyResult),
    ],
    ids=lambda cls: cls.__qualname__,
)
def test_copy_fields_fills_every_destination_field(src, dst):
    assert set(dst.get_fields_and_field_types()) <= set(src.get_fields_and_field_types())
    copy_fields(src(), dst())  # rosidl setters reject a type mismatch
