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


"""rosetta_port's exit status and argument checks.

main() runs end to end over real bags with only the writer stubbed. A port
where some bags failed still writes the rest, then exits non-zero naming
them, so a script can tell it from a complete one. Bad arguments are
argparse usage errors, before any bag is read.
"""

import sys

import pytest
from bridge_helpers import StubWriter, joint_state, write_bag

from rosetta_ros.offline import port as port_mod

CONTRACT = """
robot_type: test
robot_interface: ros2
fps: 30
observations:
  observation.state:
    channel: {topic: /joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.j1]
"""


@pytest.fixture
def writer(monkeypatch):
    writer = StubWriter()
    monkeypatch.setattr(port_mod, "load_dataset_writer", lambda _format: writer)
    return writer


@pytest.fixture
def contract(tmp_path):
    path = tmp_path / "contract.yaml"
    path.write_text(CONTRACT)
    return path


def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["rosetta_port", *map(str, argv)])
    port_mod.main()


def test_a_partial_port_writes_the_good_bags_and_exits_non_zero(tmp_path, contract, writer, monkeypatch):
    raw = tmp_path / "raw"
    write_bag(raw / "good", [("/joint_states", 1_000_000_000, joint_state(["j1"], [0.5]))])
    # The observation topic is registered but never published: no frame can be emitted.
    write_bag(raw / "empty", [], extra_topics={"/joint_states": "sensor_msgs/msg/JointState"})

    with pytest.raises(SystemExit) as exited:
        _main(monkeypatch, "--raw-dir", raw, "--contract", contract)

    assert str(exited.value) == "rosetta_port: 1 bag failed to port: empty. The dataset holds the rest."
    assert writer.calls.count("save_episode") == 1
    assert "finalize" in writer.calls


def test_a_complete_port_exits_normally(tmp_path, contract, writer, monkeypatch):
    write_bag(tmp_path / "raw" / "good", [("/joint_states", 1_000_000_000, joint_state(["j1"], [0.5]))])
    _main(monkeypatch, "--raw-dir", tmp_path / "raw", "--contract", contract)
    assert "finalize" in writer.calls


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (["--contract", "missing.yaml"], "--contract missing.yaml is not a file"),
        (["--raw-dir", "nope"], "--raw-dir nope is not a directory"),
        (["--format", "lerobt"], "invalid choice: 'lerobt'"),
        (["--num-shards", "2", "--shard-index", "2"], "--num-shards and --shard-index go together"),
        (["--num-shards", "2"], "--num-shards and --shard-index go together"),
        # Without --num-shards it was ignored, and every shard ported every bag.
        (["--shard-index", "1"], "--num-shards and --shard-index go together"),
    ],
    ids=[
        "missing-contract",
        "missing-raw-dir",
        "unknown-format",
        "shard-out-of-range",
        "shard-index-missing",
        "num-shards-missing",
    ],
)
def test_bad_arguments_are_usage_errors(tmp_path, contract, monkeypatch, capsys, args, error):
    (tmp_path / "raw").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(port_mod, "available_dataset_writers", lambda: ["lerobot"])
    base = {"--raw-dir": "raw", "--contract": str(contract)}
    for flag, value in zip(args[::2], args[1::2], strict=True):
        base[flag] = value
    with pytest.raises(SystemExit) as exited:
        _main(monkeypatch, *[x for pair in base.items() for x in pair])
    assert exited.value.code == 2
    assert error in capsys.readouterr().err
