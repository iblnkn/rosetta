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

"""Timeline enforcement at ingest, and align-driven buffers everywhere.

The contract chose a timeline per source (validated at load); a message that
still arrives without it must be dropped — never ingested on a fabricated
timeline. And every resolved spec (actions included) carries its own align,
so the bag path builds every buffer from the spec instead of fabricating
hold buffers for non-observations.
"""

import types

import pytest
from rosetta.contract.schema import Align, Channel, Source
from rosetta.contract.specs import ActionStreamSpec, ObservationStreamSpec
from rosetta.frames.resample import StreamBuffer
from rosetta.robots.ros2.ingest import StreamIngest
from rosetta.robots.ros2.ros2_utils import provided_timelines


def _obs(align):
    return ObservationStreamSpec(
        key="observation.state",
        names=["position.j1"],
        fps=30,
        source=Source(
            channel=Channel(topic="/joint_states", type="sensor_msgs/msg/JointState"),
            align=align,
        ),
        is_image=False,
        image_resize=None,
        dtype="float64",
    )


def _joint_state(stamp_ns=None):
    header = types.SimpleNamespace(
        stamp=types.SimpleNamespace(
            sec=0 if stamp_ns is None else stamp_ns // 1_000_000_000,
            nanosec=0 if stamp_ns is None else stamp_ns % 1_000_000_000,
        )
    )
    return types.SimpleNamespace(header=header, name=["j1"], position=[1.0])


def test_missing_timeline_drops_message_then_recovers():
    import rosetta.robots.ros2.decoders  # noqa: F401  register codecs

    spec = _obs(Align("hold", "header"))
    buffer = StreamBuffer.from_spec(spec)
    warns, infos = [], []
    ingest = StreamIngest(warn=warns.append, info=infos.append)

    # Uninitialized header stamp: dropped with one warning, nothing buffered.
    ingest.ingest(_joint_state(), spec, buffer, index=0, fallback_ns=5_000)
    ingest.ingest(_joint_state(), spec, buffer, index=0, fallback_ns=6_000)
    assert buffer.last_ts is None
    assert len(warns) == 1 and "header" in warns[0]

    # A stamped message recovers the stream (with a notice) and lands on the
    # header timeline, not the receive time.
    ingest.ingest(_joint_state(stamp_ns=3_000_000_000), spec, buffer, index=0, fallback_ns=7_000)
    assert buffer.last_ts == 3_000_000_000
    assert len(infos) == 1 and "recovered" in infos[0]


def test_receive_timeline_uses_receive_time():
    import rosetta.robots.ros2.decoders  # noqa: F401  register codecs

    spec = _obs(Align("hold", "receive"))
    buffer = StreamBuffer.from_spec(spec)
    ingest = StreamIngest(warn=lambda m: None, info=lambda m: None)

    # Header stamp present but ignored: the contract chose 'receive'.
    ingest.ingest(_joint_state(stamp_ns=3_000_000_000), spec, buffer, index=0, fallback_ns=42_000)
    assert buffer.last_ts == 42_000


def test_action_align_drives_its_buffer():
    """Actions carry real align now — no fabricated hold buffers anywhere."""
    spec = ActionStreamSpec(
        key="action",
        names=["position.j1"],
        fps=30,
        source=Source(
            channel=Channel(topic="/cmd", type="sensor_msgs/msg/JointState"),
            align=Align("drop", "receive"),
        ),
    )
    buffer = StreamBuffer.from_spec(spec)
    assert buffer.policy == "drop"

    # drop semantics: a sample older than one step is a gap, not held.
    step_ns = int(1e9 / 30)
    buffer.push(1_000_000_000, [0.5])
    assert buffer.sample(1_000_000_000) == [0.5]
    assert buffer.sample(1_000_000_000 + 2 * step_ns) is None


def test_provided_timelines_by_type():
    pytest.importorskip("rosidl_runtime_py")
    assert provided_timelines("sensor_msgs/msg/JointState") == {"receive", "header"}
    assert provided_timelines("std_msgs/msg/Float64") == {"receive"}
    with pytest.raises(ValueError, match="NoSuchThing"):
        provided_timelines("sensor_msgs/msg/NoSuchThing")
