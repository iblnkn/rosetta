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

"""Helpers shared by the codec tests."""

import pytest
from rosidl_runtime_py.utilities import get_message

from rosetta.contract.schema import load_contract
from rosetta.contract.specs import iter_action_specs, iter_observation_specs

JOINT_STATE = "sensor_msgs/msg/JointState"


def require_message(msg_type: str) -> None:
    """Skip the test when ``msg_type``'s package is not installed (e.g. control_msgs).

    Contract load introspects timelines via rosidl, and encoders build real
    messages, so both need the message package.
    """
    try:
        get_message(msg_type)
    except Exception:
        pytest.skip(f"{msg_type} message package unavailable")


def _entry(key, topic, msg_type, select, apply_block):
    lines = [
        f"  {key}:",
        f"    channel: {{topic: {topic}, type: {msg_type}}}",
        "    align: {strategy: hold, timeline: receive}",
    ]
    if select is not None:
        lines.append(f"    select: [{', '.join(select)}]")
    if apply_block is not None:
        lines.append(f"    apply: {apply_block}")
    return "\n".join(lines)


def load_specs(
    tmp_path,
    *,
    obs_type=JOINT_STATE,
    obs_select=("position.j1",),
    obs_apply=None,
    act_type=JOINT_STATE,
    act_select=("position.j1",),
    act_apply=None,
):
    """(action_spec, observation_spec) from a one-observation, one-action contract.

    A ``*_select`` of None leaves ``select:`` out. Loading runs the real
    load-time checks, so a bad channel raises ContractValidationError here.
    """
    text = "\n".join(
        [
            "robot_type: test",
            "robot_interface: ros2",
            "fps: 30",
            "observations:",
            _entry("observation.state", "/obs", obs_type, obs_select, obs_apply),
            "actions:",
            _entry("action", "/cmd", act_type, act_select, act_apply),
            "",
        ]
    )
    p = tmp_path / "c.yaml"
    p.write_text(text)
    contract = load_contract(p)
    return next(iter(iter_action_specs(contract))), next(iter(iter_observation_specs(contract)))
