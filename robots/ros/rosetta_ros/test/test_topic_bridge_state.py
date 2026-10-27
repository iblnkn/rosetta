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

"""TopicBridge per-stream state across episodes and ticks.

reset_state() clears episode data but keeps latched streams and re-arms the
ingest warn-once state; missing-stream logs are edge-triggered per stream,
not per key; warmed_up gates on every observation stream. The decode guard
itself is StreamIngest's (test_ingest_timeline.py); the bridge only forwards.
"""

from bridge_helpers import FakeNode, obs_spec
from sensor_msgs.msg import JointState

from rosetta.frames.resample import StreamBuffer
from rosetta_ros.topic_bridge import TopicBridge


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


def _bridge_with_buffers(specs):
    """Bridge bound to a FakeNode with the buffers setup() would create, no graph."""
    bridge = TopicBridge(specs, [], fps=30)
    node = FakeNode()
    bridge._node = node
    buffers = [StreamBuffer.from_spec(spec) for spec in specs]
    bridge._obs_buffers = list(zip(specs, buffers, strict=True))
    return bridge, node, buffers


def _shared_key_specs():
    # Distinct namespaces, as _derive_namespaces would assign for two topics
    # sharing a key (FrameLayout rejects colliding rendered names).
    return (
        obs_spec("observation.state", ["position.j1", "velocity.j1"], "/js_a", namespace="js_a"),
        obs_spec("observation.state", ["position.j1"], "/js_b", namespace="js_b"),
    )


def test_reset_state_preserves_latched_streams():
    # Latched (transient_local) data is not episode-scoped: DDS redelivers
    # latched samples only to NEW subscriptions, so clearing the buffer on a
    # publish-once topic would leave it empty forever.
    latched = obs_spec("observation.map", ["position.j1"], "/map", qos={"durability": "transient_local", "depth": 1})
    plain = obs_spec("observation.js", ["position.j1"], "/js")
    bridge, _, (buf_latched, buf_plain) = _bridge_with_buffers([latched, plain])
    buf_latched.push(1_000, 1.0)
    buf_plain.push(1_000, 2.0)

    bridge.reset_state()

    assert buf_latched.last_val == 1.0  # latched survives the episode boundary
    assert buf_plain.last_val is None  # episode-scoped data cleared


def test_reset_state_rearms_warning():
    spec = obs_spec("observation.state", ["position.j1", "velocity.j1"], "/js")
    bridge, node, (buffer,) = _bridge_with_buffers([spec])
    bridge._on_observation(_bad_msg(), spec=spec, buffer=buffer, index=0)
    assert len(node.logger.warnings) == 1
    bridge.reset_state()
    bridge._on_observation(_bad_msg(), spec=spec, buffer=buffer, index=0)
    assert len(node.logger.warnings) == 2


def test_shared_key_missing_stream_does_not_flap():
    # Two specs share one key, so missing/recovered state must be per stream:
    # keyed by key, B's data would clear A's flag and the logs flap per tick.
    spec_a, spec_b = _shared_key_specs()
    bridge, node, (_, buf_b) = _bridge_with_buffers([spec_a, spec_b])

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
    assert not TopicBridge([spec_a, spec_b], [], fps=30).warmed_up  # not set up: no node yet

    bridge, _, (buf_a, buf_b) = _bridge_with_buffers([spec_a, spec_b])
    assert not bridge.warmed_up  # cold: nothing delivered

    bridge._on_observation(_good_msg(), spec=spec_a, buffer=buf_a, index=0)
    assert not bridge.warmed_up  # one stream still silent

    bridge._on_observation(_good_msg(), spec=spec_b, buffer=buf_b, index=1)
    assert bridge.warmed_up
