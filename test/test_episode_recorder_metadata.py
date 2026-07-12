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

"""_write_metadata must round-trip both the operator prompt and the contract.

Covers the embed_contract addition: prompt-only (pre-existing behavior),
contract-only, both together, and neither (no-op, no file required).
"""

import yaml

from rosetta.robots.ros2.nodes.episode_recorder_node import (
    BAG_CONTRACT_HASH_KEY,
    BAG_CONTRACT_KEY,
    BAG_CUSTOM_DATA_KEY,
    BAG_METADATA_KEY,
    BAG_PROMPT_KEY,
    EpisodeRecorderNode,
)


class _FakeLogger:
    def debug(self, _msg):
        pass

    def error(self, _msg):
        pass


class _FakeSelf:
    """Enough of EpisodeRecorderNode for _write_metadata to run unbound."""

    def get_logger(self):
        return _FakeLogger()


def _init_bag_dir(tmp_path):
    bag_dir = tmp_path / "ep"
    bag_dir.mkdir()
    (bag_dir / "metadata.yaml").write_text(yaml.safe_dump({BAG_METADATA_KEY: {}}))
    return bag_dir


def _read_custom_data(bag_dir) -> dict:
    meta = yaml.safe_load((bag_dir / "metadata.yaml").read_text())
    return meta[BAG_METADATA_KEY].get(BAG_CUSTOM_DATA_KEY, {})


def test_prompt_only_unaffected_by_contract_fields(tmp_path):
    bag_dir = _init_bag_dir(tmp_path)
    EpisodeRecorderNode._write_metadata(_FakeSelf(), bag_dir, "pick up cube")
    custom = _read_custom_data(bag_dir)
    assert custom[BAG_PROMPT_KEY] == "pick up cube"
    assert BAG_CONTRACT_KEY not in custom
    assert BAG_CONTRACT_HASH_KEY not in custom


def test_contract_only_written_without_prompt(tmp_path):
    bag_dir = _init_bag_dir(tmp_path)
    EpisodeRecorderNode._write_metadata(_FakeSelf(), bag_dir, "", "robot_type: x\n", "deadbeef")
    custom = _read_custom_data(bag_dir)
    assert BAG_PROMPT_KEY not in custom
    assert custom[BAG_CONTRACT_KEY] == "robot_type: x\n"
    assert custom[BAG_CONTRACT_HASH_KEY] == "deadbeef"


def test_prompt_and_contract_both_written(tmp_path):
    bag_dir = _init_bag_dir(tmp_path)
    EpisodeRecorderNode._write_metadata(_FakeSelf(), bag_dir, "pick up cube", "robot_type: x\n", "deadbeef")
    custom = _read_custom_data(bag_dir)
    assert custom[BAG_PROMPT_KEY] == "pick up cube"
    assert custom[BAG_CONTRACT_KEY] == "robot_type: x\n"
    assert custom[BAG_CONTRACT_HASH_KEY] == "deadbeef"


def test_neither_prompt_nor_contract_is_a_noop(tmp_path):
    bag_dir = tmp_path / "ep_no_meta"
    bag_dir.mkdir()  # deliberately no metadata.yaml -- must not try to write one
    EpisodeRecorderNode._write_metadata(_FakeSelf(), bag_dir, "", "", "")
    assert not (bag_dir / "metadata.yaml").exists()
