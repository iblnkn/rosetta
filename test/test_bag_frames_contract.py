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

"""read_bag_contract_text reads back what episode_recorder_node writes."""

import yaml
from rosetta.robots.ros2.bag_metadata import (
    BAG_CONTRACT_KEY,
    BAG_CUSTOM_DATA_KEY,
    BAG_METADATA_KEY,
)
from rosetta.robots.ros2.offline.bag_frames import read_bag_contract_text


def _write_metadata(bag_dir, custom_data: dict):
    bag_dir.mkdir(parents=True, exist_ok=True)
    meta = {BAG_METADATA_KEY: {BAG_CUSTOM_DATA_KEY: custom_data}}
    (bag_dir / "metadata.yaml").write_text(yaml.safe_dump(meta))


def test_reads_recorded_contract(tmp_path):
    bag_dir = tmp_path / "ep"
    _write_metadata(bag_dir, {BAG_CONTRACT_KEY: "robot_type: x\n"})
    assert read_bag_contract_text(bag_dir) == "robot_type: x\n"


def test_absent_contract_returns_empty_string(tmp_path):
    bag_dir = tmp_path / "ep"
    _write_metadata(bag_dir, {})
    assert read_bag_contract_text(bag_dir) == ""


def test_missing_metadata_file_returns_empty_string(tmp_path):
    bag_dir = tmp_path / "ep_no_meta"
    bag_dir.mkdir()
    assert read_bag_contract_text(bag_dir) == ""
