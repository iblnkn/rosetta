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

Pinned behaviors:
- a message received exactly at tick t belongs to frame t;
- warmup: ticks before every routed observation stream has data are skipped,
  and the first emitted frame carries real data, not zeros; late action
  streams do NOT gate warmup;
- a routed observation topic with zero messages -> RuntimeError;
- a spec whose type is encode-only (no decoder) -> RuntimeError before any
  message is read (porting it could only record zeros);
- an unstamped message on a header-timeline stream is dropped (warn-once)
  and a stamped one recovers, same as the live ingest guard.
"""

import logging

import numpy as np
import pytest
from bridge_helpers import act_spec, joint_state, obs_spec, write_bag

from rosetta.contract.schema import Align, Channel, Source
from rosetta.contract.specs import ActionStreamSpec
from rosetta.frames.codecs import decode_value
from rosetta.frames.layout import FrameLayout
from rosetta.frames.resample import StreamBuffer
from rosetta_ros.offline.bag_frames import iter_bag_frames

FPS = 10
STEP_NS = int(1e9 / FPS)
T0 = 1_000_000_000  # arbitrary bag epoch


SPECS = [
    obs_spec("observation.state", ["position.j1", "position.j2"], "/obs_a", fps=FPS),
    obs_spec("observation.env", ["position.e1"], "/obs_b", fps=FPS),
    act_spec("action", ["position.a1"], "/cmd", fps=FPS),
]
WARMUP_KEYS = {"observation.state", "observation.env"}


# (topic, receive_ns, msg). /obs_b starts late (warmup skips ticks 0 and 1);
# /obs_a has a message at EXACTLY tick 3 (T0 + 300ms); /cmd starts even later
# and must not gate warmup.
EVENTS = [
    ("/obs_a", T0, joint_state(["j1", "j2"], [1.0, 2.0])),
    ("/obs_a", T0 + 95_000_000, joint_state(["j1", "j2"], [3.0, 4.0])),
    ("/obs_b", T0 + 150_000_000, joint_state(["e1"], [7.0])),
    ("/obs_a", T0 + 300_000_000, joint_state(["j1", "j2"], [5.0, 6.0])),  # exact tick boundary
    ("/obs_b", T0 + 400_000_000, joint_state(["e1"], [8.0])),
    ("/cmd", T0 + 450_000_000, joint_state(["a1"], [9.0])),
]


def _oracle_frames(specs, events, warmup_keys):
    """Live-semantics oracle: at each grid tick t, push every event with
    receive time <= t into the same StreamBuffers the bridge uses, then sample
    and assemble through the same FrameLayout. Ticks before all warmup buffers
    have data are skipped."""
    layout = FrameLayout(specs)
    buffers = [StreamBuffer.from_spec(spec) for spec in specs]

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
    write_bag(d, EVENTS)
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


def test_observation_stream_with_no_messages_raises(tmp_path):
    # /obs_b topic exists in the bag but never publishes: warmup cannot
    # complete and the porter must fail loudly instead of emitting zeros.
    events = [e for e in EVENTS if e[0] != "/obs_b"]
    d = tmp_path / "bag_stale"
    write_bag(d, events, extra_topics={"/obs_b": "sensor_msgs/msg/JointState"})
    with pytest.raises(RuntimeError, match="observation.env"):
        list(iter_bag_frames(d, SPECS, warmup_keys=WARMUP_KEYS))


def test_observation_topic_absent_from_bag_raises(tmp_path):
    # /obs_b not even registered in the bag (wrong topic name in the contract):
    # must fail fast with a clear message instead of silently training on
    # 100% fabricated zeros for that stream.
    events = [e for e in EVENTS if e[0] != "/obs_b"]
    d = tmp_path / "bag_absent"
    write_bag(d, events)
    with pytest.raises(RuntimeError, match="missing from bag.*observation.env"):
        list(iter_bag_frames(d, SPECS, warmup_keys=WARMUP_KEYS))


def test_absent_action_topic_still_ports(tmp_path):
    # Non-warmup streams (actions) may legitimately be absent: warn + zero-fill.
    events = [e for e in EVENTS if e[0] != "/cmd"]
    d = tmp_path / "bag_no_cmd"
    write_bag(d, events)
    ported = list(iter_bag_frames(d, SPECS, warmup_keys=WARMUP_KEYS))
    assert ported
    for frame in ported:
        np.testing.assert_array_equal(frame["action"], [0.0])


def test_undecodable_spec_fails_fast(bag_dir):
    # A channel type with no registered decoder is valid live (actions are
    # never decoded there) but unportable: the porter records every spec, so it must refuse
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


def test_header_timeline_drop_and_recovery_offline(tmp_path, caplog):
    # Offline twin of the live ingest guard: an unstamped message on a
    # header-timeline stream is dropped (warn-once), a stamped one recovers,
    # and porting continues instead of dying or fabricating a timestamp.
    spec = obs_spec("observation.h", ["position.j1"], "/obs_h", fps=FPS, align=Align("hold", "header"))
    events = [
        ("/obs_h", T0, joint_state(["j1"], [1.0])),  # zero stamp -> dropped
        ("/obs_h", T0 + 50_000_000, joint_state(["j1"], [2.0], stamp_ns=T0 + 50_000_000)),
        ("/obs_h", T0 + 150_000_000, joint_state(["j1"], [3.0], stamp_ns=T0 + 150_000_000)),
    ]
    d = tmp_path / "bag_header"
    write_bag(d, events)

    with caplog.at_level(logging.WARNING):
        frames = list(iter_bag_frames(d, [spec], warmup_keys={"observation.h"}))

    drops = [r for r in caplog.records if "missing its 'header' timeline" in r.getMessage()]
    assert len(drops) == 1
    # Tick 0 (T0) is a warmup skip (its only message was dropped); the first
    # emitted frame carries the recovered, stamped value.
    np.testing.assert_array_equal(frames[0]["observation.h"], [2.0])
