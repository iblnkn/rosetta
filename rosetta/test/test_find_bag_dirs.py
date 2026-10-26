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

"""find_bag_dirs: discovery and the sharding contract parallel porting relies on.

convert_bags_parallel.sh fans one shard per process; shards must be
deterministic, disjoint, and jointly cover every bag. Finding no bags at all
is an error; a valid shard that happens to receive none is an empty list.
"""

import pytest

from rosetta.robots.ros2.offline.bag_frames import find_bag_dirs


def _make_bags(root, n):
    for i in range(n):
        d = root / f"bag_{i:02d}"
        d.mkdir(parents=True)
        (d / "metadata.yaml").write_text("{}")


def test_finds_all_bags_sorted(tmp_path):
    _make_bags(tmp_path, 3)
    dirs = find_bag_dirs(tmp_path)
    assert [d.name for d in dirs] == ["bag_00", "bag_01", "bag_02"]


def test_shards_are_deterministic_disjoint_and_cover(tmp_path):
    _make_bags(tmp_path, 5)
    shards = [find_bag_dirs(tmp_path, num_shards=3, shard_index=i) for i in range(3)]
    combined = [d for shard in shards for d in shard]
    assert sorted(combined) == find_bag_dirs(tmp_path)  # cover, no duplicates
    assert len(combined) == len(set(combined))


def test_valid_but_empty_shard_returns_empty_list(tmp_path):
    _make_bags(tmp_path, 2)
    assert find_bag_dirs(tmp_path, num_shards=8, shard_index=7) == []


def test_shard_index_required_with_num_shards(tmp_path):
    _make_bags(tmp_path, 1)
    with pytest.raises(ValueError, match="shard_index required"):
        find_bag_dirs(tmp_path, num_shards=2)


def test_shard_index_out_of_range_rejected(tmp_path):
    _make_bags(tmp_path, 1)
    with pytest.raises(ValueError, match=">= num_shards"):
        find_bag_dirs(tmp_path, num_shards=2, shard_index=2)


def test_no_bags_at_all_raises(tmp_path):
    with pytest.raises(RuntimeError, match="No bag directories"):
        find_bag_dirs(tmp_path)
