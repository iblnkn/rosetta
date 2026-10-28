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

"""ensure_live_observation_layout: the contracts the live LeRobot stack would silently mangle.

The bundled-contract parity sweep lives in lerobot_rosetta/test/test_features_parity.py.
"""

import pytest

from lerobot_robot_rosetta.rosetta import ensure_live_observation_layout
from rosetta.contract.schema import Align, Channel, Source
from rosetta.contract.specs import ActionStreamSpec, ObservationStreamSpec


def _source(topic):
    return Source(
        channel=Channel(topic=topic, type="sensor_msgs/msg/JointState"),
        align=Align("hold", "receive"),
    )


def _obs(key, names, topic="/t"):
    return ObservationStreamSpec(
        key=key,
        names=list(names),
        fps=30,
        source=_source(topic),
        is_image=False,
        image_resize=None,
        dtype="float64",
    )


def _act(key, names, topic="/cmd"):
    return ActionStreamSpec(key=key, names=list(names), fps=30, source=_source(topic), dtype="float64")


@pytest.mark.parametrize(
    ("obs", "act", "named"),
    [
        (
            [_obs("observation.state", ["a", "b"]), _obs("observation.environment_state", ["e"], topic="/env")],
            [_act("action", ["x"])],
            "observation.environment_state",
        ),
        (
            [_obs("observation.state", ["a"])],
            [_act("action", ["x"]), _act("action.gripper", ["g"], topic="/grip")],
            "action.gripper",
        ),
        # No select: no names for the live stack to derive features from.
        (
            [_obs("observation.state", ["a"], topic="/arm"), _obs("observation.state", [], topic="/raw")],
            [_act("action", ["x"])],
            "/raw",
        ),
    ],
    ids=["second_numeric_observation_key", "second_action_key", "selectless_numeric_stream"],
)
def test_gate_rejects_layouts_live_lerobot_mangles(obs, act, named):
    with pytest.raises(ValueError, match=named):
        ensure_live_observation_layout(obs, act)


def test_gate_passes_multi_topic_single_key():
    # Multi-topic aggregation under ONE key is the supported live pattern.
    specs = [
        _obs("observation.state", ["a", "b"], topic="/arm"),
        _obs("observation.state", ["g"], topic="/gripper"),
    ]
    ensure_live_observation_layout(specs, [_act("action", ["x"])])
