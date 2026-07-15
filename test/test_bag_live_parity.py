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

"""Bag-porter vs live-semantics oracle parity test.

The train/deploy parity contract: a frame the porter emits for tick t must be
exactly what the live bridge would serve at time t — the same StreamBuffers fed
with every message received at or before t, assembled through the same
FrameLayout. The oracle here IS the live machinery (StreamBuffer.push/sample +
FrameLayout.assemble) driven with those semantics directly; the porter's output
must match frame-for-frame.

Pinned regressions:
- frame 0 must not be zero-filled (old porter emitted the first tick before
  pushing the first message);
- a message received exactly at tick t belongs to frame t (old porter used an
  inclusive comparison and excluded it);
- warmup: ticks before every routed observation stream has data are skipped;
  late action streams do NOT gate warmup;
- a routed observation topic with zero messages -> RuntimeError;
- a spec whose type is encode-only (no decoder) -> RuntimeError before any
  message is read (porting it could only record zeros);
- an unstamped message on a header-timeline stream is dropped (warn-once)
  and a stamped one recovers, same as the live ingest guard.
"""

import logging

import numpy as np
import pytest
import rosbag2_py
from rclpy.serialization import serialize_message
from rosetta.contract.schema import Align, Channel, Source
from rosetta.contract.specs import ActionStreamSpec, ObservationStreamSpec
from rosetta.frames.codecs import decode_value
from rosetta.frames.layout import FrameLayout
from rosetta.frames.resample import StreamBuffer
from rosetta.robots.ros2.offline.bag_frames import iter_bag_frames
from sensor_msgs.msg import JointState

FPS = 10
STEP_NS = int(1e9 / FPS)
T0 = 1_000_000_000  # arbitrary bag epoch


def _obs(key, names, topic):
    return ObservationStreamSpec(
        key=key,
        names=list(names),
        fps=FPS,
        source=Source(
            channel=Channel(topic=topic, type="sensor_msgs/msg/JointState"),
            align=Align("hold", "receive"),
        ),
        is_image=False,
        image_resize=None,
        dtype="float64",
    )


def _act(key, names, topic):
    return ActionStreamSpec(
        key=key,
        names=list(names),
        fps=FPS,
        source=Source(
            channel=Channel(topic=topic, type="sensor_msgs/msg/JointState"),
            align=Align("hold", "receive"),
        ),
        dtype="float64",
    )


SPECS = [
    _obs("observation.state", ["position.j1", "position.j2"], "/obs_a"),
    _obs("observation.env", ["position.e1"], "/obs_b"),
    _act("action", ["position.a1"], "/cmd"),
]
WARMUP_KEYS = {"observation.state", "observation.env"}


def _js(names, positions):
    msg = JointState()
    msg.name = list(names)
    msg.position = [float(p) for p in positions]
    return msg


# (topic, receive_ns, msg). /obs_b starts late (warmup skips ticks 0 and 1);
# /obs_a has a message at EXACTLY tick 3 (T0 + 300ms); /cmd starts even later
# and must not gate warmup.
EVENTS = [
    ("/obs_a", T0, _js(["j1", "j2"], [1.0, 2.0])),
    ("/obs_a", T0 + 95_000_000, _js(["j1", "j2"], [3.0, 4.0])),
    ("/obs_b", T0 + 150_000_000, _js(["e1"], [7.0])),
    ("/obs_a", T0 + 300_000_000, _js(["j1", "j2"], [5.0, 6.0])),  # exact tick boundary
    ("/obs_b", T0 + 400_000_000, _js(["e1"], [8.0])),
    ("/cmd", T0 + 450_000_000, _js(["a1"], [9.0])),
]


def _write_bag(bag_dir, events, extra_topics=()):
    """Write events to a bag; extra_topics registers topics with no messages."""
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    for tid, topic in enumerate(sorted({t for t, _, _ in events} | set(extra_topics))):
        try:
            meta = rosbag2_py.TopicMetadata(
                id=tid,
                name=topic,
                type="sensor_msgs/msg/JointState",
                serialization_format="cdr",
            )
        except TypeError:  # pre-Jazzy signature without id
            meta = rosbag2_py.TopicMetadata(topic, "sensor_msgs/msg/JointState", "cdr")
        writer.create_topic(meta)
    for topic, ns, msg in sorted(events, key=lambda e: e[1]):
        writer.write(topic, serialize_message(msg), ns)
    del writer  # flush + write metadata.yaml


def _oracle_frames(specs, events, warmup_keys):
    """Live-semantics oracle: at each grid tick t, push every event with
    receive time <= t into the same StreamBuffers the bridge uses, then sample
    and assemble through the same FrameLayout. Ticks before all warmup buffers
    have data are skipped."""
    layout = FrameLayout(specs)
    buffers = []
    for spec in specs:
        if isinstance(spec, ObservationStreamSpec):
            buffers.append(StreamBuffer.from_spec(spec))
        else:
            buffers.append(StreamBuffer(policy="hold", step_ns=STEP_NS, tol_ns=0))

    events = sorted(events, key=lambda e: e[1])
    start_ns = events[0][1]
    end_ns = events[-1][1]
    n_ticks = max(1, int((end_ns - start_ns) // STEP_NS) + 1)

    frames = []
    ev = 0
    warmed_up = False
    for tick_idx in range(n_ticks):
        tick_ns = start_ns + tick_idx * STEP_NS
        while ev < len(events) and events[ev][1] <= tick_ns:
            topic, ns, msg = events[ev]
            for spec, buf in zip(specs, buffers, strict=False):
                if spec.source.channel.topic == topic:
                    buf.push(ns, decode_value(msg, spec))
            ev += 1
        if not warmed_up:
            warmed_up = all(
                buf.last_ts is not None for spec, buf in zip(specs, buffers, strict=False) if spec.key in warmup_keys
            )
            if not warmed_up:
                continue
        frame = layout.assemble([buf.sample(tick_ns) for buf in buffers])
        frame["_tick_ns"] = tick_ns
        frame["_is_last_tick"] = tick_idx == n_ticks - 1
        frames.append(frame)
    return frames


@pytest.fixture
def bag_dir(tmp_path):
    d = tmp_path / "bag"
    _write_bag(d, EVENTS)
    return d


def test_porter_matches_live_oracle(bag_dir):
    ported = list(iter_bag_frames(bag_dir, SPECS, warmup_keys=WARMUP_KEYS))
    oracle = _oracle_frames(SPECS, EVENTS, WARMUP_KEYS)

    assert len(ported) == len(oracle), f"porter emitted {len(ported)} frames, live oracle expects {len(oracle)}"
    for i, (p, o) in enumerate(zip(ported, oracle, strict=False)):
        for key in ("observation.state", "observation.env", "action"):
            np.testing.assert_array_equal(p[key], o[key], err_msg=f"frame {i} key {key}")
        assert p["is_first"][0] == (i == 0)
        assert p["is_last"][0] == o["_is_last_tick"]
        assert p["is_terminal"][0] == o["_is_last_tick"]


def test_warmup_skips_until_all_observations_present(bag_dir):
    # Grid: T0, +100, +200, +300, +400 ms. /obs_b first sample at +150ms ->
    # ticks 0 and 1 skipped; 3 frames emitted (+200, +300, +400).
    ported = list(iter_bag_frames(bag_dir, SPECS, warmup_keys=WARMUP_KEYS))
    assert len(ported) == 3
    # First emitted frame holds the latest data as of +200ms.
    np.testing.assert_array_equal(ported[0]["observation.state"], [3.0, 4.0])
    np.testing.assert_array_equal(ported[0]["observation.env"], [7.0])
    # Late /cmd stream did not gate warmup; it zero-fills until its message.
    np.testing.assert_array_equal(ported[0]["action"], [0.0])


def test_message_at_exact_tick_is_included(bag_dir):
    # /obs_a publishes [5, 6] with receive time exactly T0+300ms (tick 3):
    # live sample(t) includes ts <= t, so the ported frame for that tick must too.
    ported = list(iter_bag_frames(bag_dir, SPECS, warmup_keys=WARMUP_KEYS))
    frame_at_300 = ported[1]  # +200, +300, +400
    np.testing.assert_array_equal(frame_at_300["observation.state"], [5.0, 6.0])


def test_no_zero_filled_first_frame(bag_dir):
    ported = list(iter_bag_frames(bag_dir, SPECS, warmup_keys=WARMUP_KEYS))
    for key in ("observation.state", "observation.env"):
        assert not np.allclose(ported[0][key], 0), key


def test_observation_stream_with_no_messages_raises(tmp_path):
    # /obs_b topic exists in the bag but never publishes: warmup cannot
    # complete and the porter must fail loudly instead of emitting zeros.
    events = [e for e in EVENTS if e[0] != "/obs_b"]
    d = tmp_path / "bag_stale"
    _write_bag(d, events, extra_topics=("/obs_b",))
    with pytest.raises(RuntimeError, match="observation.env"):
        list(iter_bag_frames(d, SPECS, warmup_keys=WARMUP_KEYS))


def test_observation_topic_absent_from_bag_raises(tmp_path):
    # /obs_b not even registered in the bag (wrong topic name in the contract):
    # must fail fast with a clear message instead of silently training on
    # 100% fabricated zeros for that stream.
    events = [e for e in EVENTS if e[0] != "/obs_b"]
    d = tmp_path / "bag_absent"
    _write_bag(d, events)
    with pytest.raises(RuntimeError, match="missing from bag.*observation.env"):
        list(iter_bag_frames(d, SPECS, warmup_keys=WARMUP_KEYS))


def test_absent_action_topic_still_ports(tmp_path):
    # Non-warmup streams (actions) may legitimately be absent: warn + zero-fill.
    events = [e for e in EVENTS if e[0] != "/cmd"]
    d = tmp_path / "bag_no_cmd"
    _write_bag(d, events)
    ported = list(iter_bag_frames(d, SPECS, warmup_keys=WARMUP_KEYS))
    assert ported
    for frame in ported:
        np.testing.assert_array_equal(frame["action"], [0.0])


def test_encode_only_spec_fails_fast(bag_dir):
    # Encode-only channel types are valid live (actions are never decoded
    # there) but unportable: the porter records every spec, so it must refuse
    # up front rather than silently record an all-zero column.
    encode_only = ActionStreamSpec(
        key="action.aux",
        names=["g1"],
        fps=FPS,
        source=Source(
            channel=Channel(topic="/grip", type="test_msgs/msg/EncodeOnly"),
            align=Align("hold", "receive"),
        ),
        dtype="float64",
    )
    with pytest.raises(RuntimeError, match="cannot be decoded for porting"):
        next(iter_bag_frames(bag_dir, [*SPECS, encode_only], warmup_keys=WARMUP_KEYS))


def _js_stamped(names, positions, stamp_ns):
    msg = _js(names, positions)
    msg.header.stamp.sec = stamp_ns // 1_000_000_000
    msg.header.stamp.nanosec = stamp_ns % 1_000_000_000
    return msg


def test_header_timeline_drop_and_recovery_offline(tmp_path, caplog):
    # Offline twin of the live ingest guard: an unstamped message on a
    # header-timeline stream is dropped (warn-once), a stamped one recovers,
    # and porting continues instead of dying or fabricating a timestamp.
    spec = ObservationStreamSpec(
        key="observation.h",
        names=["position.j1"],
        fps=FPS,
        source=Source(
            channel=Channel(topic="/obs_h", type="sensor_msgs/msg/JointState"),
            align=Align("hold", "header"),
        ),
        is_image=False,
        image_resize=None,
        dtype="float64",
    )
    events = [
        ("/obs_h", T0, _js(["j1"], [1.0])),  # zero stamp -> dropped
        ("/obs_h", T0 + 50_000_000, _js_stamped(["j1"], [2.0], T0 + 50_000_000)),
        ("/obs_h", T0 + 150_000_000, _js_stamped(["j1"], [3.0], T0 + 150_000_000)),
    ]
    d = tmp_path / "bag_header"
    _write_bag(d, events)

    with caplog.at_level(logging.WARNING):
        frames = list(iter_bag_frames(d, [spec], warmup_keys={"observation.h"}))

    drops = [r for r in caplog.records if "missing its 'header' timeline" in r.getMessage()]
    assert len(drops) == 1
    # Tick 0 (T0) is a warmup skip (its only message was dropped); the first
    # emitted frame carries the recovered, stamped value.
    np.testing.assert_array_equal(frames[0]["observation.h"], [2.0])
