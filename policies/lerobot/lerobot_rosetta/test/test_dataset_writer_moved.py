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

"""The dataset writer's old import path keeps working, with a warning, for one release."""

import importlib
import sys

import pytest

pytest.importorskip("lerobot")


def test_the_old_dataset_writer_import_still_works_and_warns():
    # Code written against lerobot_rosetta before the writer moved must not
    # break on upgrade without notice.
    sys.modules.pop("lerobot_rosetta.dataset_writer", None)
    with pytest.warns(DeprecationWarning, match="lerobot_dataset_rosetta.dataset_writer"):
        old = importlib.import_module("lerobot_rosetta.dataset_writer")

    from lerobot_dataset_rosetta.dataset_writer import LeRobotDatasetWriter, build_lerobot_features

    assert old.LeRobotDatasetWriter is LeRobotDatasetWriter
    assert old.build_lerobot_features is build_lerobot_features
