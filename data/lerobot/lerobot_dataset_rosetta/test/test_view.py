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

"""LeRobotView serves a frame reader the way LeRobotDataset serves a ported dataset, by index."""

import numpy as np
import pytest
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.factory import make_policy
from lerobot_dataset_rosetta.view import LeRobotStreamView, LeRobotView, policy_windows, split_episodes
from memory_reader import MemoryReader, PortedFramesReader

from rosetta.data import FrameReader, ReaderInfo

LENGTHS = (4, 6)


def _frame(episode, t):
    return {
        "observation.images.cam": np.full((32, 32, 3), 10 * t + 100 * episode, dtype=np.uint8),
        "observation.state": np.array([0.1 * t + episode, 0.2 * t], dtype=np.float64),
        "action": np.array([1.0 * t + 10 * episode, 2.0 * t], dtype=np.float64),
        "task": f"task {episode}",
    }


@pytest.fixture
def episodes():
    return [[_frame(e, t) for t in range(n)] for e, n in enumerate(LENGTHS)]


def _act():
    return ACTConfig(chunk_size=3, n_action_steps=3, pretrained_backbone_weights=None)


def _diffusion():
    return DiffusionConfig(
        n_obs_steps=2, horizon=8, n_action_steps=4, pretrained_backbone_weights=None, crop_shape=None
    )


def test_memory_reader_is_a_frame_reader(episodes):
    assert isinstance(MemoryReader(episodes), FrameReader)


def test_windows_follow_the_policy_config(episodes):
    features = MemoryReader(episodes).features()
    assert policy_windows(_act(), features) == {"action": [0, 1, 2]}
    assert policy_windows(_diffusion(), features) == {
        "observation.images.cam": [-1, 0],
        "observation.state": [-1, 0],
        "action": list(range(-1, 7)),
    }


@pytest.mark.parametrize("config", [_act, _diffusion], ids=["act", "diffusion"])
def test_make_policy_accepts_the_views_metadata(episodes, config):
    if config is _diffusion:
        pytest.importorskip("diffusers")  # LeRobot's diffusion policy needs lerobot[diffusion]
    cfg = config()
    view = LeRobotView(MemoryReader(episodes), cfg)
    policy = make_policy(cfg, ds_meta=view.meta)
    assert set(policy.config.input_features) == {"observation.images.cam", "observation.state"}
    assert set(policy.config.output_features) == {"action"}


def test_metadata_and_stats(episodes):
    view = LeRobotView(MemoryReader(episodes), _act(), use_imagenet_stats=False)
    meta = view.meta
    assert (meta.fps, meta.total_episodes, meta.total_frames, meta.total_tasks) == (10, 2, 10, 2)
    assert list(meta.tasks.index) == ["task 0", "task 1"]
    assert meta.camera_keys == ["observation.images.cam"]
    cam = meta.stats["observation.images.cam"]
    assert cam["mean"].shape == (3, 1, 1)
    assert 0 <= float(cam["min"].min()) <= float(cam["max"].max()) <= 1
    states = np.stack([f["observation.state"] for e in episodes for f in e])
    np.testing.assert_allclose(meta.stats["observation.state"]["mean"].numpy(), states.mean(axis=0), rtol=1e-6)


def test_imagenet_stats_replace_camera_stats(episodes):
    meta = LeRobotView(MemoryReader(episodes), _act()).meta
    np.testing.assert_allclose(meta.stats["observation.images.cam"]["mean"].flatten().numpy(), [0.485, 0.456, 0.406])


def test_samples_are_lerobot_shaped(episodes):
    view = LeRobotView(MemoryReader(episodes), _diffusion())
    item = view[0]
    assert item["observation.images.cam"].shape == (2, 3, 32, 32)
    assert item["observation.images.cam"].dtype == torch.uint8
    assert item["observation.state"].dtype == torch.float32
    assert item["action"].shape == (8, 2)
    assert item["action_is_pad"].dtype == torch.bool
    assert item["task"] == "task 0"
    assert item["index"].dtype == torch.int64


class StreamingReader(MemoryReader):
    """Streams in an order fixed by (seed, epoch), as RerunReader does, so the order shows the epoch."""

    def stream(self, *, seed, epoch, worker):
        order = np.random.default_rng([seed, epoch]).permutation(self.info().num_frames)
        return iter(self.read([i for i in order if i % worker[1] == worker[0]]))


def _spawn_loader(dataset, **kwargs):
    import multiprocessing

    return torch.utils.data.DataLoader(
        dataset, num_workers=2, multiprocessing_context=multiprocessing.get_context("spawn"), **kwargs
    )


def test_lerobots_sampler_reads_every_sample_once_an_epoch(episodes):
    """What lerobot-train does without streaming: its EpisodeAwareSampler over the view, batches from workers."""
    from lerobot.datasets import EpisodeAwareSampler

    view = LeRobotView(MemoryReader(episodes), _act())
    sampler = EpisodeAwareSampler(
        view.meta.episodes["dataset_from_index"],
        view.meta.episodes["dataset_to_index"],
        episode_indices_to_use=view.episodes,
        shuffle=True,
        absolute_to_relative_idx=view.absolute_to_relative_idx,
    )
    loader = _spawn_loader(view, batch_size=3, sampler=sampler, persistent_workers=True)
    orders = [[int(i) for batch in loader for i in batch["index"]] for _ in range(2)]
    assert all(sorted(order) == list(range(sum(LENGTHS))) for order in orders)
    assert orders[0] != orders[1]


def test_a_view_over_some_episodes_reads_only_theirs(episodes):
    """As lerobot-train reads an episode-filtered dataset: the sampler's indices, mapped to positions."""
    from lerobot.datasets import EpisodeAwareSampler

    view = LeRobotView(MemoryReader(episodes), _act()).select([1])
    sampler = EpisodeAwareSampler(
        view.meta.episodes["dataset_from_index"],
        view.meta.episodes["dataset_to_index"],
        episode_indices_to_use=view.episodes,
        absolute_to_relative_idx=view.absolute_to_relative_idx,
    )
    items = view.__getitems__(list(sampler))
    assert len(view) == LENGTHS[1]
    assert [int(item["episode_index"]) for item in items] == [1] * LENGTHS[1]
    assert sorted(int(item["frame_index"]) for item in items) == list(range(LENGTHS[1]))


@pytest.mark.parametrize(
    ("tasks", "split", "held_out"),
    [
        (["a", "a", "a", "a"], 0.25, [3]),
        (["a", "b", "a", "b", "a"], 0.5, [2, 3, 4]),
        (["a", "a", "b"], 0.1, [1, 2]),
    ],
    ids=["one-task", "two-tasks", "ceil-per-task"],
)
def test_episodes_split_as_lerobot_splits_them(tasks, split, held_out):
    """Per task, the last ceil(n * split) episodes are held out (lerobot.datasets.factory.make_train_eval_datasets)."""
    info = ReaderInfo(10, "bot", (1,) * len(tasks), tuple(tasks), tuple(dict.fromkeys(tasks)), True)
    train, held = split_episodes(info, split)
    assert held == held_out
    assert sorted(train + held) == list(range(len(tasks)))


def test_a_split_that_leaves_no_training_episode_is_refused():
    info = ReaderInfo(10, "bot", (1, 1), ("a", "b"), ("a", "b"), True)
    with pytest.raises(ValueError, match="leaves 0 training episodes"):
        split_episodes(info, 1.0)


@pytest.mark.parametrize("persistent", [False, True], ids=["restarted", "persistent"])
def test_a_stream_reshuffles_every_epoch_with_workers(episodes, persistent):
    """lerobot-train keeps its workers across epochs; a stream's order must still change every epoch."""
    view = LeRobotStreamView(StreamingReader(episodes), _act())
    loader = _spawn_loader(view, batch_size=None, persistent_workers=persistent)
    orders = [[int(item["index"]) for item in loader] for _ in range(3)]
    assert all(sorted(order) == list(range(sum(LENGTHS))) for order in orders)
    assert len({tuple(order) for order in orders}) == 3


PARITY_CONTRACT = """
robot_type: parity_bot
robot_interface: ros2
fps: 10
observations:
  observation.images.cam:
    channel: {topic: /cam, type: sensor_msgs/msg/Image}
    align: {strategy: hold, timeline: receive}
    apply: [{resize: [32, 32]}]
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


def _parity_bag(bag_dir, episode, ticks):
    """A camera on every tick, joint states from 2.25 ticks in (so warmup skips three ticks), actions between ticks."""
    from bag_helpers import image, joint_state, write_bag

    step = 100_000_000
    events = []
    for t in range(ticks):
        at = 1_000_000_000 + t * step
        events.append(("/cam", at, image(10 * t + 100 * episode, size=32)))
        events.append(("/cmd", at + step // 2, joint_state(t + 10 * episode, -t)))
        if t >= 2:
            events.append(("/js", at + step // 4, joint_state(0.5 * t + episode, t * t)))
    write_bag(bag_dir, events, prompt=f"stack {episode}")


def test_a_view_gives_lerobot_what_a_ported_dataset_gives_it(tmp_path):
    """A view over the porter's frames serves LeRobot the samples and stats that rosetta_port's dataset gives it."""
    pytest.importorskip("rosbag2_py")
    from rosetta_ros.offline.port import port

    raw = tmp_path / "raw"
    for episode, ticks in enumerate((7, 10)):
        _parity_bag(raw / f"ep{episode}", episode, ticks)
    contract = tmp_path / "robot.yaml"
    contract.write_text(PARITY_CONTRACT)

    view = LeRobotView(PortedFramesReader(contract=contract, raw_dir=raw), _diffusion(), use_imagenet_stats=False)
    port(raw, "parity/ds", contract, root=tmp_path / "ported")
    delta_timestamps = {key: [i / 10 for i in offsets] for key, offsets in view.windows.items()}
    ported = LeRobotDataset(
        "parity/ds", root=tmp_path / "ported/parity/ds", delta_timestamps=delta_timestamps, return_uint8=True
    )

    # Read by index on both sides, as lerobot-train's sampler reads them.
    assert len(view) == len(ported)
    theirs = {int(ported[i]["index"]): ported[i] for i in range(len(ported))}
    ours = {int(view[i]["index"]): view[i] for i in range(len(view))}
    assert set(ours) == set(theirs) == set(range(len(view)))
    for index, mine in ours.items():
        other = theirs[index]
        assert mine["task"] == other["task"], index
        for key in ("episode_index", "frame_index"):
            assert int(mine[key]) == int(other[key]), (index, key)
        for key in ("observation.state", "action"):
            torch.testing.assert_close(mine[key], other[key].to(mine[key].dtype), msg=f"{index} {key}")
        for key in ("observation.state", "observation.images.cam", "action"):
            torch.testing.assert_close(mine[f"{key}_is_pad"], other[f"{key}_is_pad"], msg=f"{index} {key}_is_pad")
        # Video is lossy, but neighbouring ticks are 10 levels apart.
        cam, their_cam = mine["observation.images.cam"], other["observation.images.cam"]
        assert (cam.shape, cam.dtype) == (their_cam.shape, their_cam.dtype)
        assert (cam.int() - their_cam.int()).abs().max() <= 3, index

    # Quantiles differ by design: a ported dataset's are the frame-weighted mean
    # of each episode's quantiles, while rosetta's are exact over every frame.
    for key in ("observation.state", "action"):
        mine, other = view.meta.stats[key], ported.meta.stats[key]
        for name in ("count", "mean", "std", "min", "max"):
            np.testing.assert_allclose(
                np.asarray(mine[name]), np.asarray(other[name]), rtol=1e-6, err_msg=f"{key} {name}"
            )
