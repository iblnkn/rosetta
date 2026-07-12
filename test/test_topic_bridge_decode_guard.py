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

"""Decode-failure guard in the live subscription callback.

Regression: a malformed message used to raise out of _on_observation, through
the executor, and kill the whole inference node mid-episode. Now it is dropped
with a once-per-stream warning, and the stream behaves like a missing stream.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from sensor_msgs.msg import JointState

from rosetta.contract.schema import Align, Channel, Source
from rosetta.contract.specs import ObservationStreamSpec
from rosetta.frames.resample import StreamBuffer
from rosetta.robots.ros2.topic_bridge import TopicBridge


class _FakeLogger:
    def __init__(self):
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def warning(self, msg):
        self.warnings.append(msg)

    def info(self, msg):
        self.infos.append(msg)


class _FakeNode:
    def __init__(self):
        self.logger = _FakeLogger()
        self._t = 1_000_000_000

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=self._t))

    def get_logger(self):
        return self.logger


def _spec():
    return ObservationStreamSpec(
        key="observation.state",
        names=["position.j1", "velocity.j1"],
        fps=30,
        source=Source(
            channel=Channel(topic="/js", type="sensor_msgs/msg/JointState"),
            align=Align("hold", "receive"),
        ),
        is_image=False,
        image_resize=None,
        dtype="float64",
    )


@pytest.fixture
def bridge_parts():
    spec = _spec()
    bridge = TopicBridge([spec], [], fps=30)
    node = _FakeNode()
    bridge._node = node
    buffer = StreamBuffer.from_spec(spec)
    return bridge, node, spec, buffer


def _bad_msg():
    # Names present, velocity array empty -> explicit decoder ValueError.
    msg = JointState()
    msg.name = ["j1"]
    msg.position = [0.5]
    msg.velocity = []
    return msg


def _good_msg():
    msg = JointState()
    msg.name = ["j1"]
    msg.position = [0.5]
    msg.velocity = [1.5]
    return msg


def test_malformed_message_dropped_with_single_warning(bridge_parts):
    bridge, node, spec, buffer = bridge_parts
    bridge._on_observation(_bad_msg(), spec=spec, buffer=buffer, index=0)  # must not raise
    assert buffer.last_val is None  # nothing pushed
    assert len(node.logger.warnings) == 1
    assert "Decode failed" in node.logger.warnings[0]

    bridge._on_observation(_bad_msg(), spec=spec, buffer=buffer, index=0)
    assert len(node.logger.warnings) == 1  # once per stream, not per message


def test_recovery_pushes_and_logs_once(bridge_parts):
    bridge, node, spec, buffer = bridge_parts
    bridge._on_observation(_bad_msg(), spec=spec, buffer=buffer, index=0)
    bridge._on_observation(_good_msg(), spec=spec, buffer=buffer, index=0)
    np.testing.assert_allclose(buffer.last_val, [0.5, 1.5])
    assert any("recovered" in m for m in node.logger.infos)

    # Warned-set cleared: a new failure warns again.
    bridge._on_observation(_bad_msg(), spec=spec, buffer=buffer, index=0)
    assert len(node.logger.warnings) == 2


def test_reset_state_rearms_warning(bridge_parts):
    bridge, node, spec, buffer = bridge_parts
    bridge._on_observation(_bad_msg(), spec=spec, buffer=buffer, index=0)
    assert len(node.logger.warnings) == 1
    bridge.reset_state()
    bridge._on_observation(_bad_msg(), spec=spec, buffer=buffer, index=0)
    assert len(node.logger.warnings) == 2


# ---------------------------------------------------------------------------
# Shared-key stream identity (regression: guards keyed by spec.key)
#
# Two specs may share one contract key; keying the warn/missing sets by key
# made spec B's success clear spec A's flag, flapping warn/recover (and
# missing/recovered — per tick) logs at message rate.
# ---------------------------------------------------------------------------


def _shared_key_specs():
    def spec(topic, names):
        return ObservationStreamSpec(
            key="observation.state",
            names=names,
            fps=30,
            source=Source(
                channel=Channel(topic=topic, type="sensor_msgs/msg/JointState"),
                align=Align("hold", "receive"),
            ),
            is_image=False,
            image_resize=None,
            dtype="float64",
        )

    return spec("/js_a", ["position.j1", "velocity.j1"]), spec("/js_b", ["position.j1"])


def test_shared_key_decode_guard_does_not_flap():
    spec_a, spec_b = _shared_key_specs()
    bridge = TopicBridge([spec_a, spec_b], [], fps=30)
    node = _FakeNode()
    bridge._node = node
    buf_a = StreamBuffer.from_spec(spec_a)
    buf_b = StreamBuffer.from_spec(spec_b)

    # A fails persistently while B (same key) keeps decoding fine.
    for _ in range(3):
        bridge._on_observation(_bad_msg(), spec=spec_a, buffer=buf_a, index=0)
        bridge._on_observation(_good_msg(), spec=spec_b, buffer=buf_b, index=1)

    assert len(node.logger.warnings) == 1  # A warned exactly once
    assert not any("recovered" in m for m in node.logger.infos)  # B never "recovers" A

    # A recovers: exactly one recovery log, naming A's topic.
    bridge._on_observation(_good_msg(), spec=spec_a, buffer=buf_a, index=0)
    recoveries = [m for m in node.logger.infos if "recovered" in m]
    assert len(recoveries) == 1
    assert "/js_a" in recoveries[0]


def test_shared_key_missing_stream_does_not_flap():
    spec_a, spec_b = _shared_key_specs()
    bridge = TopicBridge([spec_a, spec_b], [], fps=30)
    node = _FakeNode()
    bridge._node = node
    buf_a = StreamBuffer.from_spec(spec_a)
    buf_b = StreamBuffer.from_spec(spec_b)
    bridge._obs_buffers = [(spec_a, buf_a), (spec_b, buf_b)]

    # Only B publishes; A stays silent across several ticks.
    bridge._on_observation(_good_msg(), spec=spec_b, buffer=buf_b, index=1)
    for _ in range(3):
        bridge.sample_frame()

    missing = [m for m in node.logger.warnings if "missing" in m]
    assert len(missing) == 1  # A logged missing once, not per tick
    assert "/js_a" in missing[0]
    assert not any("recovered" in m for m in node.logger.infos)


def test_warmed_up_mirrors_bag_warmup_predicate():
    """warmed_up is the live twin of the bag porter's warmup gate: false until
    every observation stream has delivered at least one message."""
    spec_a, spec_b = _shared_key_specs()
    bridge = TopicBridge([spec_a, spec_b], [], fps=30)
    assert not bridge.warmed_up  # not set up: no node yet

    node = _FakeNode()
    bridge._node = node
    buf_a = StreamBuffer.from_spec(spec_a)
    buf_b = StreamBuffer.from_spec(spec_b)
    bridge._obs_buffers = [(spec_a, buf_a), (spec_b, buf_b)]
    assert not bridge.warmed_up  # cold: nothing delivered

    bridge._on_observation(_good_msg(), spec=spec_a, buffer=buf_a, index=0)
    assert not bridge.warmed_up  # one stream still silent

    bridge._on_observation(_good_msg(), spec=spec_b, buffer=buf_b, index=1)
    assert bridge.warmed_up
