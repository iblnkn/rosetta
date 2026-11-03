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

"""Server-query frames must equal porter frames, frame for frame.

The parity guarantee, extended to the third frame producer: one rrd bag,
read once by ``iter_bag_frames`` (the offline porter — itself parity-tested
against the live bridge) and once by ``ContractFrames`` (the compiled
catalog query against a real ``rerun server``). Same contract, identical
frames, so a checkpoint trained off the server behaves exactly like one
trained off a ported dataset.

Needs the ``rerun`` binary and the rrd storage plugin; skips cleanly when
either is missing.
"""

from __future__ import annotations

import itertools
import textwrap

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("rerun")

import rosbag2_py  # noqa: E402
from rclpy.serialization import serialize_message  # noqa: E402
from rerun.catalog import CatalogClient  # noqa: E402
from rerun.experimental.dataloader import BlockShuffle, NoShuffle  # noqa: E402
from sensor_msgs.msg import CompressedImage, JointState  # noqa: E402

from rosetta.contract.schema import load_contract  # noqa: E402
from rosetta.contract.specs import iter_observation_specs, iter_specs  # noqa: E402
from rosetta_ros.offline.bag_frames import iter_bag_frames  # noqa: E402

FPS = 10
STEP_NS = 1_000_000_000 // FPS
T0 = 1_000_000_000_000_000_000  # arbitrary epoch-ish anchor

CONTRACT_YAML = textwrap.dedent(
    """
    robot_type: parity_test
    robot_interface: ros2
    fps: 10

    observations:
      observation.state:
        channel: {topic: /js_obs, type: sensor_msgs/msg/JointState,
                  qos: {reliability: best_effort, history: keep_last, depth: 10}}
        align: {strategy: hold, timeline: receive}
        select: [position.a, position.b]
        apply: [rad2deg]

      observation.images.cam:
        channel: {topic: /cam/compressed, type: sensor_msgs/msg/CompressedImage,
                  qos: {reliability: best_effort, history: keep_last, depth: 10}}
        align: {strategy: hold, timeline: receive}
        apply: [resize: [32, 32]]

    actions:
      action:
        channel: {topic: /js_cmd, type: sensor_msgs/msg/JointState,
                  qos: {reliability: best_effort, history: keep_last, depth: 10}}
        align: {strategy: hold, timeline: receive}
        select: [position.a, position.b]
    """
)


def _js(a: float, b: float) -> JointState:
    msg = JointState()
    msg.name = ["a", "b"]
    msg.position = [a, b]
    return msg


def _jpeg(seed: int) -> CompressedImage:
    rng = np.random.default_rng(seed)
    bgr = rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", bgr)
    assert ok
    msg = CompressedImage()
    msg.format = "jpeg"
    msg.data = buf.tobytes()
    return msg


# Deliberately tricky timing: images and observations start late (warmup
# skip), one observation lands exactly on a tick, actions are sparse.
EVENTS = [
    ("/js_cmd", T0 + int(0.05e9), _js(0.5, -0.5)),
    ("/cam/compressed", T0 + int(0.32e9), _jpeg(1)),
    ("/js_obs", T0 + int(0.35e9), _js(0.1, 0.2)),
    ("/js_obs", T0 + int(0.60e9), _js(0.3, 0.4)),  # exactly on tick 6
    ("/cam/compressed", T0 + int(0.81e9), _jpeg(2)),
    ("/js_obs", T0 + int(0.95e9), _js(0.5, 0.6)),
    ("/js_cmd", T0 + int(1.02e9), _js(0.7, -0.7)),
    ("/js_obs", T0 + int(1.49e9), _js(0.9, 1.0)),
]

TOPIC_TYPES = {
    "/js_obs": "sensor_msgs/msg/JointState",
    "/js_cmd": "sensor_msgs/msg/JointState",
    "/cam/compressed": "sensor_msgs/msg/CompressedImage",
}


def _write_rrd_bag(bag_dir, representation: str, events=EVENTS) -> None:
    config = bag_dir.parent / f"{representation}.yaml"
    config.write_text(f"representations: [{representation}]\n")
    writer = rosbag2_py.SequentialWriter()
    try:
        writer.open(
            rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="rrd", storage_config_uri=str(config)),
            rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
        )
    except RuntimeError as e:
        pytest.skip(f"rosbag2_storage_rrd plugin unavailable: {e}")
    for i, (topic, type_name) in enumerate(TOPIC_TYPES.items()):
        try:
            meta = rosbag2_py.TopicMetadata(id=i, name=topic, type=type_name, serialization_format="cdr")
        except TypeError:  # pre-Jazzy signature
            meta = rosbag2_py.TopicMetadata(name=topic, type=type_name, serialization_format="cdr")
        writer.create_topic(meta)
    for topic, ts, msg in events:
        writer.write(topic, serialize_message(msg), ts)


@pytest.fixture(scope="module")
def contract_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("contract") / "parity.yaml"
    path.write_text(CONTRACT_YAML)
    return path


@pytest.fixture(scope="module")
def contract(contract_path):
    return load_contract(contract_path)


@pytest.fixture(scope="module")
def bag_dir(tmp_path_factory):
    bag = tmp_path_factory.mktemp("bags_reflected") / "episode_0"
    _write_rrd_bag(bag, "reflected")
    return bag


def _register(catalog_url: str, name: str, bags_dir) -> str:
    """Register every rrd under ``bags_dir`` as ``name``: rerun's own API, no wrapper."""
    CatalogClient(catalog_url).create_dataset(name, exist_ok=True).register_prefix(f"file://{bags_dir}").wait()
    return name


@pytest.fixture(scope="module")
def dataset_name(bag_dir, catalog_url):
    return _register(catalog_url, f"parity_{bag_dir.parent.name}", bag_dir.parent)


def _porter_frames(bag_dir, contract):
    specs = list(iter_specs(contract))
    warmup_keys = {s.key for s in iter_observation_specs(contract)}
    frames = list(iter_bag_frames(bag_dir, specs, warmup_keys=warmup_keys))
    # Boundary markers and task are episode-stream concerns the dataset doesn't carry.
    for frame in frames:
        for extra in ("is_first", "is_last", "is_terminal", "task"):
            frame.pop(extra, None)
    return frames


def _assert_frame_equal(have, want, where):
    assert set(have) == set(want), f"{where}: keys differ"
    for key, value in want.items():
        assert have[key].dtype == value.dtype, f"{where} key {key} dtype: {have[key].dtype} != {value.dtype}"
        np.testing.assert_array_equal(have[key], value, err_msg=f"{where} key {key}")


def test_frames_match_porter(bag_dir, dataset_name, contract, catalog_url):
    from rerun_rosetta import contract_dataset

    frames = list(contract_dataset(catalog_url, dataset_name, contract, shuffle_strategy=NoShuffle()))
    expected = _porter_frames(bag_dir, contract)
    assert len(frames) == len(expected), f"sample count mismatch: query={len(frames)} porter={len(expected)}"
    for i, (have, want) in enumerate(zip(frames, expected, strict=True)):
        _assert_frame_equal(have, want, f"frame {i}")


def test_action_chunks_match_porter(bag_dir, dataset_name, contract, catalog_url):
    """Sample i's chunk offset p is porter frame i + p.

    The dataloader trims each episode so a chunk never runs past its end: the
    last ``chunk - 1`` ticks start no sample.
    """
    from rerun_rosetta import contract_dataset

    chunk = 3
    windows = {"action": (0, chunk - 1)}
    frames = list(contract_dataset(catalog_url, dataset_name, contract, windows=windows, shuffle_strategy=NoShuffle()))
    expected = _porter_frames(bag_dir, contract)
    assert len(frames) == len(expected) - (chunk - 1)
    for i, sample in enumerate(frames):
        assert sample["action"].shape == (chunk, 2)
        for p in range(chunk):
            np.testing.assert_array_equal(
                sample["action"][p], expected[i + p]["action"], err_msg=f"sample {i} offset {p}"
            )
        np.testing.assert_array_equal(sample["observation.images.cam"], expected[i]["observation.images.cam"])


def test_shuffle_is_a_permutation(dataset_name, contract, catalog_url):
    from rerun_rosetta import contract_dataset

    frames = contract_dataset(catalog_url, dataset_name, contract, seed=3)
    ordered = list(contract_dataset(catalog_url, dataset_name, contract, shuffle_strategy=NoShuffle()))
    shuffled = list(frames)
    assert len(shuffled) == len(ordered)
    key = lambda f: (f["observation.state"].tobytes(), f["action"].tobytes(), f["observation.images.cam"].tobytes())  # noqa: E731
    assert sorted(map(key, shuffled)) == sorted(map(key, ordered))


def test_features_match_layout(contract, dataset_name, catalog_url):
    from rerun_rosetta import contract_dataset

    from rosetta.contract.specs import iter_policy_specs
    from rosetta.frames.layout import FrameLayout

    frames = contract_dataset(catalog_url, dataset_name, contract)
    assert frames.features == FrameLayout(list(iter_policy_specs(contract))).lerobot_features()


def test_raw_only_recording_raises_with_the_fix(contract, catalog_url, tmp_path_factory):
    from rerun_rosetta import contract_dataset

    bag = tmp_path_factory.mktemp("bags_raw") / "episode_0"
    _write_rrd_bag(bag, "raw")
    name = _register(catalog_url, "parity_raw_only", bag.parent)
    with pytest.raises(ValueError, match="ros2 bag convert"):
        contract_dataset(catalog_url, name, contract)


def _as_list(samples):
    return samples


def test_spawned_workers_yield_the_same_samples(dataset_name, contract, catalog_url):
    """``num_workers > 0`` needs the dataset to pickle into spawned workers."""
    import torch.utils.data
    from rerun_rosetta import contract_dataset

    key = lambda f: (f["observation.state"].tobytes(), f["action"].tobytes(), f["observation.images.cam"].tobytes())  # noqa: E731
    expected = sorted(map(key, contract_dataset(catalog_url, dataset_name, contract, shuffle_strategy=NoShuffle())))

    frames = contract_dataset(catalog_url, dataset_name, contract, num_workers=2)
    loader = torch.utils.data.DataLoader(
        frames, batch_size=1, num_workers=2, multiprocessing_context="spawn", collate_fn=_as_list
    )
    assert sorted(key(frame) for batch in loader for frame in batch) == expected


def test_saved_manifest_replays_the_same_samples(dataset_name, contract, catalog_url, tmp_path):
    from rerun_rosetta import contract_dataset

    path = tmp_path / "manifest.parquet"
    built = list(contract_dataset(catalog_url, dataset_name, contract, shuffle_strategy=NoShuffle(), manifest=path))
    assert path.exists()
    loaded = list(contract_dataset(catalog_url, dataset_name, contract, shuffle_strategy=NoShuffle(), manifest=path))
    assert len(loaded) == len(built)
    for i, (have, want) in enumerate(zip(loaded, built, strict=True)):
        _assert_frame_equal(have, want, f"sample {i}")


def test_saved_manifest_for_other_windows_raises(dataset_name, contract, catalog_url, tmp_path):
    from rerun_rosetta import contract_dataset

    path = tmp_path / "manifest.parquet"
    contract_dataset(catalog_url, dataset_name, contract, manifest=path)
    with pytest.raises(ValueError, match="streams and windows"):
        contract_dataset(catalog_url, dataset_name, contract, windows={"action": (0, 2)}, manifest=path)
    with pytest.raises(ValueError, match="num_workers"):
        contract_dataset(catalog_url, dataset_name, contract, num_workers=2, manifest=path)


def test_stats_layer_matches_porter_frames(bag_dir, dataset_name, contract, contract_path, catalog_url, tmp_path):
    """Stats loaded from the layer equal numpy's over every porter frame, and a second write is a no-op."""
    from rerun_rosetta import load_stats, write_stats_layer
    from rerun_rosetta.stats import QUANTILES, quantile_name

    entry = CatalogClient(catalog_url).get_dataset(dataset_name)
    assert write_stats_layer(entry, contract_path, tmp_path) == entry.segment_ids()
    assert write_stats_layer(entry, contract_path, tmp_path) == []
    stats = load_stats(entry, contract_path)

    frames = _porter_frames(bag_dir, contract)
    expected = {
        "observation.state": np.stack([f["observation.state"] for f in frames]).astype(np.float64),
        "action": np.stack([f["action"] for f in frames]).astype(np.float64),
        "observation.images.cam": np.concatenate([f["observation.images.cam"].reshape(-1, 3) for f in frames]).astype(
            np.float64
        ),
    }
    assert set(stats) == set(expected)
    for key, values in expected.items():
        assert stats[key]["count"].tolist() == [len(frames)], key
        want = {
            "mean": values.mean(axis=0),
            "std": values.std(axis=0),
            "min": values.min(axis=0),
            "max": values.max(axis=0),
            **{quantile_name(q): v for q, v in zip(QUANTILES, np.quantile(values, QUANTILES, axis=0), strict=True)},
        }
        for name, value in want.items():
            np.testing.assert_allclose(stats[key][name], value, rtol=0, atol=1e-9, err_msg=f"{key} {name}")


# A longer recording for ordering tests: two episodes, the observed joint
# position is the tick index, so a sample's state names its tick.
TICKS = 150


def _ticking_events(episode: int) -> list:
    events = [("/js_cmd", T0 + int(0.05e9), _js(0.0, 0.0))]
    for tick in range(TICKS):
        t = T0 + (tick + 1) * STEP_NS
        events.append(("/js_obs", t, _js(float(tick), float(episode))))
        if tick % 5 == 0:
            events.append(("/cam/compressed", t, _jpeg(tick)))
    return sorted(events, key=lambda e: e[1])


@pytest.fixture(scope="module")
def ticking_dataset(tmp_path_factory, catalog_url):
    root = tmp_path_factory.mktemp("bags_ticking")
    for episode in range(2):
        _write_rrd_bag(root / f"episode_{episode}", "reflected", events=_ticking_events(episode))
    return _register(catalog_url, "parity_ticking", root)


def _tick(frame) -> tuple[int, int]:
    tick, episode = np.round(np.deg2rad(frame["observation.state"])).astype(int)
    return int(episode), int(tick)


def _adjacent_fraction(ticks: list[tuple[int, int]]) -> float:
    pairs = list(itertools.pairwise(ticks))
    return sum(a[0] == b[0] and abs(a[1] - b[1]) == 1 for a, b in pairs) / len(pairs)


def test_natural_order_is_adjacent(ticking_dataset, contract, catalog_url):
    from rerun_rosetta import contract_dataset

    ticks = [_tick(f) for f in contract_dataset(catalog_url, ticking_dataset, contract, shuffle_strategy=NoShuffle())]
    assert len(set(ticks)) == len(ticks) > 2 * (TICKS - 10)
    assert _adjacent_fraction(ticks) > 0.95  # the measure below detects correlated order


def test_default_order_is_decorrelated(ticking_dataset, contract, catalog_url):
    from rerun_rosetta import contract_dataset

    ordered = [_tick(f) for f in contract_dataset(catalog_url, ticking_dataset, contract, shuffle_strategy=NoShuffle())]
    ticks = [_tick(f) for f in contract_dataset(catalog_url, ticking_dataset, contract)]
    assert sorted(ticks) == sorted(ordered)
    assert _adjacent_fraction(ticks) < 0.05


def test_block_shuffle_without_buffer_raises(dataset_name, contract, catalog_url):
    from rerun_rosetta import contract_dataset

    with pytest.raises(ValueError, match="buffer_size"):
        contract_dataset(catalog_url, dataset_name, contract, shuffle_strategy=BlockShuffle())


def test_block_shuffle_buffer_decorrelates(ticking_dataset, contract, catalog_url):
    from rerun_rosetta import contract_dataset

    ordered = [_tick(f) for f in contract_dataset(catalog_url, ticking_dataset, contract, shuffle_strategy=NoShuffle())]
    strategy = BlockShuffle(buffer_size=1024)
    ticks = [_tick(f) for f in contract_dataset(catalog_url, ticking_dataset, contract, shuffle_strategy=strategy)]
    assert sorted(ticks) == sorted(ordered)
    assert _adjacent_fraction(ticks) < 0.05


def _epoch_ticks(frames, loader, epoch: int) -> list[tuple[int, int]]:
    frames.set_epoch(epoch)
    return [_tick(f) for batch in loader for f in batch]


def test_persistent_workers_reshuffle_each_epoch(ticking_dataset, contract, catalog_url):
    import torch.utils.data
    from rerun_rosetta import contract_dataset

    def loader(frames, persistent):
        return torch.utils.data.DataLoader(
            frames,
            batch_size=4,
            num_workers=2,
            multiprocessing_context="spawn",
            persistent_workers=persistent,
            collate_fn=_as_list,
        )

    persistent = contract_dataset(catalog_url, ticking_dataset, contract, num_workers=2)
    fresh = contract_dataset(catalog_url, ticking_dataset, contract, num_workers=2)
    persistent_loader, fresh_loader = loader(persistent, True), loader(fresh, False)
    epochs = [_epoch_ticks(persistent, persistent_loader, e) for e in range(2)]
    assert sorted(epochs[0]) == sorted(epochs[1])
    assert len(set(epochs[0])) == len(epochs[0])
    assert epochs[0] != epochs[1]
    assert epochs == [_epoch_ticks(fresh, fresh_loader, e) for e in range(2)]
