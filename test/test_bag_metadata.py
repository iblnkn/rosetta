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

"""The recorder/porter metadata seam: one hash, one custom_data shape."""

import hashlib

import pytest
import yaml
from rosetta.robots.ros2.bag_metadata import (
    BAG_CONTRACT_HASH_KEY,
    BAG_CUSTOM_DATA_KEY,
    BAG_METADATA_KEY,
    BAG_PROMPT_KEY,
    contract_hash,
    read_bag_metadata,
    read_custom_field,
    update_custom_data,
)


def test_contract_hash_is_byte_exact(tmp_path):
    # Regression: the recorder used to hash read_text().encode() while the
    # porter hashed read_bytes(); universal-newline translation made the
    # identical CRLF file hash differently, producing spurious "recorded
    # with a different contract" warnings at port time.
    p = tmp_path / "c.yaml"
    p.write_bytes(b"robot_type: test\r\nfps: 30\r\n")
    assert contract_hash(p) == hashlib.sha256(p.read_bytes()).hexdigest()
    assert contract_hash(p) != hashlib.sha256(p.read_text().encode()).hexdigest()


def test_custom_data_round_trip(tmp_path):
    meta_path = tmp_path / "metadata.yaml"
    meta_path.write_text(yaml.safe_dump({BAG_METADATA_KEY: {"storage_identifier": "mcap"}}))

    update_custom_data(meta_path, {BAG_PROMPT_KEY: "pick", BAG_CONTRACT_HASH_KEY: "deadbeef"})

    meta = read_bag_metadata(tmp_path)
    assert read_custom_field(meta, BAG_PROMPT_KEY) == "pick"
    assert read_custom_field(meta, BAG_CONTRACT_HASH_KEY) == "deadbeef"
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


def test_update_missing_file_raises(tmp_path):
    # Retry policy belongs to the caller (the recorder); the helper fails fast.
    with pytest.raises(OSError):
        update_custom_data(tmp_path / "metadata.yaml", {BAG_PROMPT_KEY: "x"})
