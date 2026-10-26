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

"""StreamIngest policy: timeline enforcement, decode guarding, warn-once state.

The contract chose a timeline per source (validated at load); a message that
still arrives without it must be dropped — never ingested on a fabricated
timeline. A message that fails decode is likewise dropped (warn-once, with a
recovery notice) instead of killing the caller. And every resolved spec
(actions included) carries its own align, so the bag path builds every buffer
from the spec instead of fabricating hold buffers for non-observations.

Pure unit tests: fakes are SimpleNamespace, no rclpy anywhere (pinned by the
subprocess import-purity test at the bottom).
"""

import subprocess
import sys
import types

from rosetta.contract.schema import Align, Channel, Source
from rosetta.contract.specs import ActionStreamSpec, ObservationStreamSpec
from rosetta.frames.resample import StreamBuffer
from rosetta.robots.ros2.ingest import StreamIngest


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


def _joint_state(stamp_ns=None, joint="j1"):
    header = types.SimpleNamespace(
        stamp=types.SimpleNamespace(
            sec=0 if stamp_ns is None else stamp_ns // 1_000_000_000,
            nanosec=0 if stamp_ns is None else stamp_ns % 1_000_000_000,
        )
    )
    return types.SimpleNamespace(header=header, name=[joint], position=[1.0])


def _ingest():
    warns, infos = [], []
    return StreamIngest(warn=warns.append, info=infos.append), warns, infos


# -------------------- Timeline enforcement --------------------


def test_missing_timeline_drops_message_with_one_warning():
    spec = _obs(Align("hold", "header"))
    buffer = StreamBuffer.from_spec(spec)
    ingest, warns, _ = _ingest()

    # Uninitialized header stamps: dropped, nothing buffered, warned once.
    ingest.ingest(_joint_state(), spec, buffer, index=0, receive_ns=5_000)
    ingest.ingest(_joint_state(), spec, buffer, index=0, receive_ns=6_000)
    assert buffer.last_ts is None
    assert len(warns) == 1
    assert "header" in warns[0]


def test_stamped_message_recovers_and_lands_on_header_timeline():
    spec = _obs(Align("hold", "header"))
    buffer = StreamBuffer.from_spec(spec)
    ingest, _, infos = _ingest()

    ingest.ingest(_joint_state(), spec, buffer, index=0, receive_ns=5_000)
    # A stamped message recovers the stream (with a notice) and lands on the
    # header timeline, not the receive time.
    ingest.ingest(_joint_state(stamp_ns=3_000_000_000), spec, buffer, index=0, receive_ns=7_000)
    assert buffer.last_ts == 3_000_000_000
    assert len(infos) == 1
    assert "recovered" in infos[0]


def test_receive_timeline_uses_receive_time():
    spec = _obs(Align("hold", "receive"))
    buffer = StreamBuffer.from_spec(spec)
    ingest, _, _ = _ingest()

    # Header stamp present but ignored: the contract chose 'receive'.
    ingest.ingest(_joint_state(stamp_ns=3_000_000_000), spec, buffer, index=0, receive_ns=42_000)
    assert buffer.last_ts == 42_000


def test_reset_rearms_timeline_warning():
    spec = _obs(Align("hold", "header"))
    buffer = StreamBuffer.from_spec(spec)
    ingest, warns, _ = _ingest()

    ingest.ingest(_joint_state(), spec, buffer, index=0, receive_ns=5_000)
    ingest.reset()
    ingest.ingest(_joint_state(), spec, buffer, index=0, receive_ns=6_000)
    assert len(warns) == 2


# -------------------- Decode guarding --------------------
#
# The real registry JointState decoder raises when the selected joint is
# absent from the message — no mocks needed to drive the decode-failure path.


def test_decode_failure_drops_message_with_one_warning():
    spec = _obs(Align("hold", "receive"))
    buffer = StreamBuffer.from_spec(spec)
    ingest, warns, _ = _ingest()

    ingest.ingest(_joint_state(joint="other"), spec, buffer, index=0, receive_ns=1_000)
    ingest.ingest(_joint_state(joint="other"), spec, buffer, index=0, receive_ns=2_000)
    assert buffer.last_ts is None
    assert len(warns) == 1
    assert "Decode failed" in warns[0]


def test_decode_recovery_pushes_and_notices():
    spec = _obs(Align("hold", "receive"))
    buffer = StreamBuffer.from_spec(spec)
    ingest, _, infos = _ingest()

    ingest.ingest(_joint_state(joint="other"), spec, buffer, index=0, receive_ns=1_000)
    ingest.ingest(_joint_state(joint="j1"), spec, buffer, index=0, receive_ns=2_000)
    assert buffer.last_ts == 2_000
    assert len(infos) == 1
    assert "recovered" in infos[0]


def test_reset_rearms_decode_warning():
    spec = _obs(Align("hold", "receive"))
    buffer = StreamBuffer.from_spec(spec)
    ingest, warns, _ = _ingest()

    ingest.ingest(_joint_state(joint="other"), spec, buffer, index=0, receive_ns=1_000)
    ingest.reset()
    ingest.ingest(_joint_state(joint="other"), spec, buffer, index=0, receive_ns=2_000)
    assert len(warns) == 2


# -------------------- Align-driven buffers --------------------


def test_action_align_drives_its_buffer():
    """Actions carry real align now — no fabricated hold buffers anywhere."""
    spec = ActionStreamSpec(
        key="action",
        names=["position.j1"],
        fps=30,
        dtype="float64",
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


# -------------------- Import purity --------------------


def test_ingest_imports_without_rclpy():
    """StreamIngest and its whole import graph stay rclpy-free (CLAUDE.md unit
    standard). Runs in a subprocess: the shared pytest process has rclpy
    loaded by other test modules."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import rosetta.robots.ros2.ingest; assert 'rclpy' not in sys.modules",
        ],
        check=True,
    )
