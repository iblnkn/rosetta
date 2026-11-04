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

"""rosetta-lerobot-train runs lerobot-train on a frame reader, and the checkpoint carries its contract."""

import importlib.metadata
import json
import types

import pytest

pytest.importorskip("rosbag2_py")

from bag_helpers import image, joint_state, write_bag
from lerobot.scripts import lerobot_train
from lerobot_dataset_rosetta import sidecar
from lerobot_dataset_rosetta.sidecar import READER_DATASET_NAMESPACE, find_contract_for_pretrained
from lerobot_dataset_rosetta.train import PATCHED, main, split_rosetta_args

from rosetta.data import registry
from rosetta.data.registry import FRAME_READER_GROUP

FPS = 10
STEP_NS = int(1e9 / FPS)
T0 = 1_000_000_000

CONTRACT = """
robot_type: wrapper_bot
robot_interface: ros2
fps: 10
observations:
  observation.images.cam:
    channel: {topic: /cam, type: sensor_msgs/msg/Image}
    align: {strategy: hold, timeline: receive}
    apply: [{resize: [16, 16]}]
  observation.state:
    channel: {topic: /js, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.j1, position.j2]
actions:
  action:
    channel: {topic: /cmd, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: receive}
    select: [position.j1, position.j2]
"""


def _write_bag(bag_dir, n):
    events = []
    for t in range(n):
        ns = T0 + t * STEP_NS
        events += [
            ("/cam", ns, image(t * 10)),
            ("/js", ns, joint_state(t, -t)),
            ("/cmd", ns, joint_state(t + 1, -t - 1)),
        ]
    write_bag(bag_dir, events)


def test_split_rosetta_args():
    options, rest = split_rosetta_args(["--rosetta.reader=rerun", "--rosetta.catalog-url=/x", "--steps=2"])
    assert options == {"reader": "rerun", "catalog_url": "/x"}
    assert rest == ["--steps=2"]


def test_the_patched_names_exist_in_this_lerobot():
    """A LeRobot that renames them breaks the wrapper; this fails first."""
    for name in PATCHED:
        assert hasattr(lerobot_train, name), name


def _register_ported_reader(monkeypatch):
    """Register memory_reader.PortedFramesReader as the frame reader ``ported``, beside the installed ones."""
    ported = importlib.metadata.EntryPoint(
        name="ported", value="memory_reader:PortedFramesReader", group=FRAME_READER_GROUP
    )

    def entry_points(group=None):
        found = list(importlib.metadata.entry_points(group=group))
        return [*found, ported] if group == FRAME_READER_GROUP else found

    monkeypatch.setattr(registry, "_ilm", types.SimpleNamespace(entry_points=entry_points))


def _train(tmp_path, monkeypatch, *extra):
    """Run rosetta-lerobot-train for two ACT steps on two recordings; returns the output directory."""
    _register_ported_reader(monkeypatch)
    raw = tmp_path / "raw"
    for episode in range(2):
        _write_bag(raw / f"ep{episode}", 8)
    contract = tmp_path / "robot.yaml"
    contract.write_text(CONTRACT)
    out = tmp_path / "outputs"

    main(
        [
            "--rosetta.reader=ported",
            f"--rosetta.contract={contract}",
            f"--rosetta.raw_dir={raw}",
            "--policy.type=act",
            "--policy.chunk_size=3",
            "--policy.n_action_steps=3",
            "--policy.dim_model=32",
            "--policy.dim_feedforward=64",
            "--policy.n_encoder_layers=1",
            "--policy.n_decoder_layers=1",
            "--policy.pretrained_backbone_weights=null",
            "--policy.device=cpu",
            "--policy.push_to_hub=false",
            "--steps=2",
            "--batch_size=2",
            "--save_freq=2",
            "--log_freq=1",
            "--num_workers=0",
            "--wandb.enable=false",
            f"--output_dir={out}",
            *extra,
        ]
    )
    return out


def test_trains_act_from_a_frame_reader_and_keeps_the_contract(tmp_path, monkeypatch):
    out = _train(tmp_path, monkeypatch)

    pretrained = out / "checkpoints" / "last" / "pretrained_model"
    assert (pretrained / "rosetta_contract.yaml").read_text() == CONTRACT
    train_config = json.loads((pretrained / "train_config.json").read_text())
    assert train_config["dataset"]["streaming"] is False  # read by index, under LeRobot's sampler
    assert find_contract_for_pretrained(str(pretrained), warn=pytest.fail) == pretrained / "rosetta_contract.yaml"


def test_holds_out_episodes_and_computes_an_eval_loss(tmp_path, monkeypatch, capfd):
    """--dataset.eval_split works as it does on a ported dataset: one of the two episodes is held out."""
    _train(tmp_path, monkeypatch, "--dataset.eval_split=0.5", "--eval_steps=1")
    assert "eval_loss=" in capfd.readouterr().err


def test_block_shuffle_streams_and_refuses_an_eval_split(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="streaming has no eval split"):
        _train(tmp_path, monkeypatch, "--rosetta.shuffle=block", "--dataset.eval_split=0.5", "--eval_steps=1")


def test_streaming_an_indexed_reader_is_refused(tmp_path, monkeypatch):
    """lerobot-train would read an indexed view in episode order: no sampler, and shuffle off when streaming."""
    with pytest.raises(SystemExit, match="only for --rosetta.shuffle=block"):
        _train(tmp_path, monkeypatch, "--dataset.streaming=true")


def test_lerobot_train_is_restored_after_a_run(tmp_path, monkeypatch):
    originals = {name: getattr(lerobot_train, name) for name in PATCHED}
    _train(tmp_path, monkeypatch)
    assert {name: getattr(lerobot_train, name) for name in PATCHED} == originals


def test_a_model_without_its_contract_is_never_looked_up_on_the_hub(tmp_path, monkeypatch):
    """A pushed model has no rosetta_contract.yaml. Its placeholder dataset repo_id must not lead to the Hub."""
    pretrained = _train(tmp_path, monkeypatch) / "checkpoints" / "last" / "pretrained_model"
    (pretrained / "rosetta_contract.yaml").unlink()  # as the model LeRobot pushes
    repo_id = json.loads((pretrained / "train_config.json").read_text())["dataset"]["repo_id"]
    assert repo_id == f"{READER_DATASET_NAMESPACE}/ported"

    looked_up = []
    resolve = sidecar.resolve_repo_file

    def recording_resolve(ref, *args, **kwargs):
        looked_up.append(ref)
        return resolve(ref, *args, **kwargs)

    monkeypatch.setattr(sidecar, "resolve_repo_file", recording_resolve)
    warnings = []
    assert find_contract_for_pretrained(str(pretrained), warn=warnings.append) is None
    assert repo_id not in looked_up
    assert any("Pass contract_path" in w for w in warnings)
