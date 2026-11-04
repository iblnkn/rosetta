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

"""Stats loaded from a stats layer."""

import numpy as np
import pytest

pytest.importorskip("rerun")

from rerun_rosetta.stats import _write_layer, load_stats, stats_layer_name

from rosetta.contract.schema import load_contract
from rosetta.contract.specs import iter_policy_specs
from rosetta.data.stats import sufficient_statistics, value_stats
from rosetta.frames.layout import FrameLayout

CONTRACT = """\
robot_type: stats_test
robot_interface: ros2
fps: 10
observations:
  observation.state:
    channel: {topic: /js, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.a, position.b]
actions:
  action:
    channel: {topic: /cmd, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.a, position.b]
"""


def test_a_segment_with_no_samples_adds_nothing_to_the_stats(catalog_url, tmp_path):
    # A short episode, or one whose camera never published, has no tick to
    # sample. Its layer records 0 frames, and training still gets stats from
    # every other segment.
    from rerun.catalog import CatalogClient

    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(CONTRACT)
    layout = FrameLayout(list(iter_policy_specs(load_contract(contract_path))))
    rng = np.random.default_rng(0)
    frames = [{"observation.state": rng.normal(size=2), "action": rng.normal(size=2)} for _ in range(6)]
    name = stats_layer_name(CONTRACT)
    _write_layer(tmp_path / "full.rrd", "full", name, *sufficient_statistics(frames, layout), CONTRACT)
    _write_layer(tmp_path / "empty.rrd", "empty", name, 0, {}, CONTRACT)
    dataset = CatalogClient(catalog_url).create_dataset("stats_with_an_empty_segment")
    dataset.register([(tmp_path / f).as_uri() for f in ("full.rrd", "empty.rrd")], layer_name=name).wait()

    # These segments hold only stats layers, so name them: by default
    # load_stats takes the segments with data for every observation stream.
    stats = load_stats(dataset, contract_path, segments=["full", "empty"])

    want = value_stats(np.stack([f["observation.state"] for f in frames]))
    assert stats["observation.state"]["count"].tolist() == [6]
    for field in ("mean", "std", "min", "max"):
        np.testing.assert_allclose(stats["observation.state"][field], want[field], err_msg=field)
