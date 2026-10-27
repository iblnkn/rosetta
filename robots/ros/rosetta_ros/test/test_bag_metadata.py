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

"""The recorder/porter metadata seam: one custom_data shape."""

import pytest
import yaml

from rosetta_ros.bag_metadata import (
    BAG_CONTRACT_KEY,
    BAG_CUSTOM_DATA_KEY,
    BAG_GOAL_ID_KEY,
    BAG_METADATA_KEY,
    BAG_PROMPT_KEY,
    read_bag_metadata,
    read_custom_field,
    update_custom_data,
    write_provenance,
)
from rosetta_ros.offline.bag_frames import read_bag_contract_text


def test_custom_data_round_trip(tmp_path):
    meta_path = tmp_path / "metadata.yaml"
    meta_path.write_text(yaml.safe_dump({BAG_METADATA_KEY: {"storage_identifier": "mcap"}}))

    update_custom_data(meta_path, {BAG_PROMPT_KEY: "pick", BAG_CONTRACT_KEY: "robot_type: x\n"})

    meta = read_bag_metadata(tmp_path)
    assert read_custom_field(meta, BAG_PROMPT_KEY) == "pick"
    assert read_bag_contract_text(tmp_path) == "robot_type: x\n"
    # rosbag2's own fields survive the read-modify-write.
    assert meta[BAG_METADATA_KEY]["storage_identifier"] == "mcap"


def test_custom_data_update_tolerates_null_block(tmp_path):
    # rosbag2 may emit `custom_data: null`; updating must not crash on it.
    meta_path = tmp_path / "metadata.yaml"
    meta_path.write_text(yaml.safe_dump({BAG_METADATA_KEY: {BAG_CUSTOM_DATA_KEY: None}}))
    update_custom_data(meta_path, {BAG_PROMPT_KEY: "go"})
    assert read_custom_field(read_bag_metadata(tmp_path), BAG_PROMPT_KEY) == "go"


def test_read_defaults_are_empty(tmp_path):
    # A bag recorded outside rosetta has no custom_data: readers answer ""
    # (and a missing metadata.yaml answers {}), never raise.
    assert read_bag_metadata(tmp_path) == {}
    assert read_custom_field({}, BAG_PROMPT_KEY) == ""
    assert read_bag_contract_text(tmp_path) == ""


def test_update_missing_file_raises(tmp_path):
    # Retry policy belongs to the caller (the recorder); the helper fails fast.
    with pytest.raises(OSError):
        update_custom_data(tmp_path / "metadata.yaml", {BAG_PROMPT_KEY: "x"})


_FIELDS = {
    "prompt": (BAG_PROMPT_KEY, "pick up cube"),
    "contract_text": (BAG_CONTRACT_KEY, "robot_type: x\n"),
    "goal_id": (BAG_GOAL_ID_KEY, "00010203-0405-0607-0809-0a0b0c0d0e0f"),
}


@pytest.mark.parametrize(
    "given",
    [("prompt",), ("contract_text",), ("goal_id",), ("prompt", "contract_text", "goal_id")],
    ids="+".join,
)
def test_writes_exactly_the_non_empty_fields(tmp_path, given):
    bag_dir = tmp_path / "ep"
    bag_dir.mkdir()
    (bag_dir / "metadata.yaml").write_text(yaml.safe_dump({BAG_METADATA_KEY: {}}))

    write_provenance(bag_dir, **{name: _FIELDS[name][1] for name in given})

    meta = yaml.safe_load((bag_dir / "metadata.yaml").read_text())
    assert meta[BAG_METADATA_KEY][BAG_CUSTOM_DATA_KEY] == dict(_FIELDS[name] for name in given)


def test_nothing_to_store_leaves_the_bag_untouched(tmp_path):
    bag_dir = tmp_path / "ep_no_meta"
    bag_dir.mkdir()  # deliberately no metadata.yaml -- must not try to write one
    write_provenance(bag_dir)
    assert not (bag_dir / "metadata.yaml").exists()
