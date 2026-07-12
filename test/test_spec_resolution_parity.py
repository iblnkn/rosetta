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

"""Resolution guard: computed fields derive correctly; declaration reads through.

Specs are composed, not copied: ``spec.source`` IS the declaration ``Source``,
so declaration facts (topic, qos, decoder, align, kind, ...) can never go
missing — there is no copy step to forget. What CAN still go wrong is the
compute job: dtype precedence, names-from-select, operator building, namespace
derivation, image geometry, and the projections' forced values. These tests
pin exactly that, plus the identity guarantee itself.
"""

from rosetta.contract.schema import (
    Align,
    Channel,
    Contract,
    FrameEntry,
    Source,
)
from rosetta.contract.specs import (
    iter_action_specs,
    iter_observation_specs,
    iter_reward_as_action_specs,
)


def make_contract(observations=(), actions=(), rewards=()):
    return Contract(
        robot_type="parity_bot",
        robot_interface="ros2",
        fps=37,
        observations=list(observations),
        actions=list(actions),
        tasks=[],
        adjunct=[],
        rewards=list(rewards),
        signals=[],
        info=[],
        complementary_data=[],
    )


STATE_SOURCE = Source(
    channel=Channel(
        topic="/joint_states",
        type="sensor_msgs/msg/JointState",
        qos={"depth": 7},
        dtype="float64",
        decoder="my.module:decode",
    ),
    align=Align(strategy="asof", timeline="header", tolerance_ms=41),
    select=["position.shoulder", "position.elbow"],
    apply=(("rad2deg", None),),
    kind="binary",
)
STATE_OBS = FrameEntry(key="observation.state", sources=(STATE_SOURCE,))

IMAGE_OBS = FrameEntry(
    key="observation.images.wrist",
    sources=(
        Source(
            channel=Channel(topic="/wrist/image_raw", type="sensor_msgs/msg/Image"),
            align=Align(strategy="hold", timeline="header"),
            apply=(("resize", [24, 32]),),
        ),
    ),
)

# Two sources with DIFFERENT aligns: per-source action alignment is real.
ACTION = FrameEntry(
    key="action",
    sources=(
        Source(
            channel=Channel(
                topic="/cmd",
                type="std_msgs/msg/Float64MultiArray",
                qos={"depth": 9},
                decoder="my.module:decode",
                encoder="my.module:encode",
                safety="hold",
            ),
            align=Align(strategy="asof", timeline="header", tolerance_ms=23),
            select=["data.0", "data.1"],
            apply=(("rad2deg", None),),
            kind="binary",
        ),
        Source(
            channel=Channel(
                topic="/gripper_cmd",
                type="std_msgs/msg/Float64MultiArray",
                safety="zeros",
            ),
            align=Align(strategy="drop", timeline="receive"),
            select=["data.2"],
        ),
    ),
)

REWARD = FrameEntry(
    key="reward.success",
    sources=(
        Source(
            channel=Channel(topic="/reward", type="std_msgs/msg/Float64MultiArray"),
            align=Align(strategy="hold", timeline="receive"),
            kind="binary",
        ),
    ),
)


def test_observation_source_reads_through_and_computed_fields_derive():
    state, image = iter_observation_specs(make_contract(observations=[STATE_OBS, IMAGE_OBS]))

    # Identity: the spec's source IS the declaration object — every
    # declaration fact (topic, qos, decoder, align, kind, ...) is reachable
    # with no copy in between.
    assert state.source is STATE_SOURCE
    assert state.source.channel.topic == "/joint_states"
    assert state.source.align.tolerance_ms == 41

    # Computed fields: each derivation, pinned against a non-default input.
    assert state.key == "observation.state"  # entry-level, carried onto the spec
    assert state.names == ["position.shoulder", "position.elbow"]  # from select
    assert state.fps == 37  # contract root
    assert state.dtype == "float64"  # explicit channel dtype wins precedence
    assert [o.name for o in state.operators] == ["rad2deg"]  # built from apply
    assert (state.is_image, image.is_image) == (False, True)  # from the key
    assert image.image_resize == (24, 32)  # from the resize operator
    assert image.dtype == "video"  # image precedence tier


def test_action_computed_fields_and_per_source_align():
    arm, gripper = iter_action_specs(make_contract(actions=[ACTION]))

    assert arm.source is ACTION.sources[0]
    assert gripper.source is ACTION.sources[1]

    assert arm.names == ["data.0", "data.1"]
    assert arm.dtype == "float64"  # custom-decoder tier of the precedence rule
    assert [o.name for o in arm.operators] == ["rad2deg"]

    # Per-source align survives: the second source resolves differently.
    assert gripper.source.align == Align("drop", "receive")
    assert gripper.source.channel.safety == "zeros"

    # Namespace derivation: two topics sharing the "action" key.
    assert (arm.namespace, gripper.namespace) == ("cmd", "gripper_cmd")


def test_reward_as_action_projection_forced_values():
    (spec,) = iter_reward_as_action_specs(make_contract(rewards=[REWARD]))

    # Projection overrides live in COMPUTED fields only:
    assert spec.key == "action"  # forced, != the reward entry's key
    assert spec.names == ["data"]  # synthesized when select is empty

    # Declaration facts read through untouched — including kind, which the
    # old flat design silently dropped (defaulted to "continuous"). A
    # declared kind on a reward channel is now honored.
    assert spec.source is REWARD.sources[0]
    assert spec.source.kind == "binary"
    # Reward channels cannot declare safety/encoder (rejected at load), so
    # reading through source matches the old forced "none"/None.
    assert spec.source.channel.safety == "none"
    assert spec.source.channel.encoder is None
