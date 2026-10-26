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

import pytest

from rosetta.contract.errors import ContractValidationError
from rosetta.contract.schema import (
    Align,
    Channel,
    Contract,
    FrameEntry,
    Source,
)
from rosetta.contract.specs import (
    _derive_namespaces,
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
    assert state.names == ("position.shoulder", "position.elbow")  # from select
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

    assert arm.names == ("data.0", "data.1")
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
    assert spec.names == ("data",)  # synthesized when select is empty

    # Declaration facts read through untouched — including kind, which the
    # old flat design silently dropped (defaulted to "continuous"). A
    # declared kind on a reward channel is now honored.
    assert spec.source is REWARD.sources[0]
    assert spec.source.kind == "binary"
    # Reward channels cannot declare safety/encoder (rejected at load), so
    # reading through source matches the old forced "none"/None.
    assert spec.source.channel.safety == "none"
    assert spec.source.channel.encoder is None


def test_depth_named_rgb_topic_loads():
    """No name heuristic: a depth-named topic under an image key is legal —
    actual depth images are rejected at decode time, where the message's
    encoding field is authoritative (see test_image_decoders)."""
    depth_named_obs = FrameEntry(
        key="observation.images.wrist",
        sources=(
            Source(
                channel=Channel(topic="/wrist/depth_module/rgb/image_raw", type="sensor_msgs/msg/Image"),
                align=Align("hold", "header"),
                apply=(("resize", [24, 32]),),
            ),
        ),
    )
    specs = list(iter_observation_specs(make_contract(observations=[depth_named_obs])))
    assert specs[0].dtype == "video"


def test_image_observation_without_resize_rejected():
    """Every image observation must declare its output geometry -- without one
    there is no static shape for the dataset feature/zero-fill, so this must
    fail at load, not at first frame assembly."""
    no_resize_obs = FrameEntry(
        key="observation.images.wrist",
        sources=(
            Source(
                channel=Channel(topic="/wrist/image_raw", type="sensor_msgs/msg/Image"),
                align=Align("hold", "header"),
            ),
        ),
    )
    with pytest.raises(ContractValidationError, match="resize"):
        list(iter_observation_specs(make_contract(observations=[no_resize_obs])))


def test_third_party_operator_with_output_hw_satisfies_image_gate():
    """The image-geometry gate reads Operator.output_hw, not the name 'resize',
    so any plugin declaring a fixed output size can fulfill it."""
    from rosetta.contract.operators import (
        OPERATOR_REGISTRY,
        Invertibility,
        Operator,
        register_operator,
    )

    @register_operator("_test_scale", kind=Invertibility.FORWARD_ONLY)
    class _Scale(Operator):
        def __init__(self, args, ctx):
            del args, ctx
            self.output_hw = (8, 8)

        def forward(self, arr):
            return arr

    scaled_obs = FrameEntry(
        key="observation.images.wrist",
        sources=(
            Source(
                channel=Channel(topic="/wrist/image_raw", type="sensor_msgs/msg/Image"),
                align=Align("hold", "header"),
                apply=(("_test_scale", None),),
            ),
        ),
    )
    try:
        (spec,) = iter_observation_specs(make_contract(observations=[scaled_obs]))
        assert spec.image_resize == (8, 8)
    finally:
        OPERATOR_REGISTRY.pop("_test_scale", None)


def test_resize_on_non_image_stream_rejected():
    """resize on a state vector would crash on every message at runtime
    (`h, w = img.shape[:2]` on a 1-D array); the operator's ctx.is_image gate
    turns it into a spec-resolution error instead."""
    state_obs = FrameEntry(
        key="observation.state",
        sources=(
            Source(
                channel=Channel(topic="/joint_states", type="sensor_msgs/msg/JointState"),
                align=Align("hold", "receive"),
                select=["position.j1"],
                apply=(("resize", [24, 32]),),
            ),
        ),
    )
    with pytest.raises(ContractValidationError, match="image"):
        list(iter_observation_specs(make_contract(observations=[state_obs])))


def test_last_geometry_operator_wins():
    """Pipelines run front-to-back, so the LAST declared output_hw is the
    stream's final geometry."""
    image_obs = FrameEntry(
        key="observation.images.wrist",
        sources=(
            Source(
                channel=Channel(topic="/wrist/image_raw", type="sensor_msgs/msg/Image"),
                align=Align("hold", "header"),
                apply=(("resize", [48, 64]), ("resize", [24, 32])),
            ),
        ),
    )
    (spec,) = iter_observation_specs(make_contract(observations=[image_obs]))
    assert spec.image_resize == (24, 32)


def test_string_stream_with_apply_rejected_at_load():
    """Operators transform numeric arrays; declaring `apply` on a string
    stream must fail at spec resolution, not silently skip the pipeline at
    decode time."""
    string_obs = FrameEntry(
        key="observation.environment_state",
        sources=(
            Source(
                channel=Channel(topic="/status", type="std_msgs/msg/String"),
                align=Align("hold", "receive"),
                apply=(("clamp", {"min": 0.0, "max": 1.0}),),
            ),
        ),
    )
    with pytest.raises(ContractValidationError, match="string"):
        list(iter_observation_specs(make_contract(observations=[string_obs])))


def test_derive_namespaces_compound_tier():
    """Three topics where no single segment distinguishes all of them (index
    0 is {a,a,b}, index 1 is {x,y,x}, index 2 is {1,1,2} -- each has a
    duplicate) must fall through to the depth-2 compound-prefix tier."""
    topics = ["/a/x/1", "/a/y/1", "/b/x/2"]
    assert _derive_namespaces(topics) == {
        "/a/x/1": "a.x",
        "/a/y/1": "a.y",
        "/b/x/2": "b.x",
    }


@pytest.mark.parametrize(
    ("topics", "expected"),
    [
        # First-segment tier: earliest segment index where all topics differ.
        (["/arm/state", "/base/state"], {"/arm/state": "arm", "/base/state": "base"}),
        (["/arm/pos", "/arm/vel"], {"/arm/pos": "pos", "/arm/vel": "vel"}),
        # Uneven depths: the short topic contributes an empty segment at the
        # winning index and ends up with no namespace ('' -> None downstream).
        (["/cmd_vel", "/cmd_vel/filtered"], {"/cmd_vel": "", "/cmd_vel/filtered": "filtered"}),
        # Single topic: no namespace needed.
        (["/only"], {"/only": ""}),
    ],
)
def test_derive_namespaces_tiers(topics, expected):
    assert _derive_namespaces(topics) == expected


def test_derive_namespaces_identical_normalization_rejected():
    """Distinct topic strings that normalize to the same segments ('/a/b' vs
    'a/b') have no unique namespace at any tier; colliding prefixes must be a
    load error, not silent duplicate feature names."""
    with pytest.raises(ContractValidationError, match="identical paths"):
        _derive_namespaces(["/a/b", "a/b"])
