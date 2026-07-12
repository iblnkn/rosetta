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

"""HIL manager teleop-event edge detection and episode exclusion.

Regressions pinned:
- level-triggered event handling: every Joy message reports every button's
  level, so the unpressed `failure` button's release branch cleared the
  reward override that a `success` press had just set (the override never
  stuck, and the classifier resumed overwriting the reward), and the
  unpressed `is_intervention` button forced the mux back to policy on every
  message, making the set_intervention service unusable while a joystick
  publishes;
- ManageEpisode goal accept raced (`_policy_goal_handle` set far downstream).
"""

from types import SimpleNamespace

import pytest
from rclpy.action import GoalResponse
from rosetta.robots.ros2.nodes.rosetta_hil_manager_node import RosettaHilManagerNode

# events_spec.select maps event name -> Joy selector. Button layout:
# 0=is_intervention, 1=terminate, 2=rerecord, 3=success, 4=failure.
EVENTS_SPEC = SimpleNamespace(
    select={
        "is_intervention": "buttons.0",
        "terminate_episode": "buttons.1",
        "rerecord_episode": "buttons.2",
        "success": "buttons.3",
        "failure": "buttons.4",
    }
)


def _joy(intervention=0, terminate=0, rerecord=0, success=0, failure=0):
    return SimpleNamespace(buttons=[intervention, terminate, rerecord, success, failure])


@pytest.fixture
def node(rclpy_ctx):
    n = RosettaHilManagerNode()
    yield n
    n.destroy_node()


def test_success_override_survives_unpressed_failure_button(node):
    # The regression: success pressed, failure unpressed in the SAME message.
    node._on_teleop_events(_joy(success=1), EVENTS_SPEC)
    assert node._human_reward_override is True
    assert node._current_reward == node.get_parameter("human_reward_positive").value


def test_override_latches_after_release(node):
    node._on_teleop_events(_joy(success=1), EVENTS_SPEC)
    node._on_teleop_events(_joy(), EVENTS_SPEC)  # release
    assert node._human_reward_override is True  # latched
    # Classifier output is still blocked while the override holds.
    msg = SimpleNamespace(data=0.25)
    node._on_reward_classifier_output(msg, "/reward")
    assert node._current_reward == node.get_parameter("human_reward_positive").value


def test_held_button_fires_once(node):
    published = []
    node._publish_human_reward = published.append
    for _ in range(3):
        node._on_teleop_events(_joy(success=1), EVENTS_SPEC)
    assert len(published) == 1  # press edge only, not per message


def test_failure_press_overrides_with_negative_reward(node):
    node._on_teleop_events(_joy(success=1), EVENTS_SPEC)
    node._on_teleop_events(_joy(failure=1), EVENTS_SPEC)
    assert node._current_reward == node.get_parameter("human_reward_negative").value
    assert node._human_reward_override is True


def test_intervention_edges_toggle_mux(node):
    assert node._control_source == "policy"
    node._on_teleop_events(_joy(intervention=1), EVENTS_SPEC)
    assert node._control_source == "teleop"
    node._on_teleop_events(_joy(intervention=1), EVENTS_SPEC)  # held: no change
    assert node._control_source == "teleop"
    node._on_teleop_events(_joy(), EVENTS_SPEC)  # release edge
    assert node._control_source == "policy"


def test_service_intervention_not_fought_by_unpressed_joystick(node):
    # set_intervention(True) with the joystick button unpressed: subsequent
    # Joy messages (no edge) must not force the mux back to policy.
    node._on_teleop_events(_joy(), EVENTS_SPEC)  # establish unpressed state
    with node._mux_lock:
        node._control_source = "teleop"  # as set by the service
    node._on_teleop_events(_joy(), EVENTS_SPEC)
    node._on_teleop_events(_joy(), EVENTS_SPEC)
    assert node._control_source == "teleop"


def test_terminate_press_sets_stop_once(node):
    node._on_teleop_events(_joy(terminate=1), EVENTS_SPEC)
    assert node._stop_requested is True


def test_episode_goal_accept_is_mutually_exclusive(node):
    node._accepting_goals = True
    assert node._on_goal(None) == GoalResponse.ACCEPT
    assert node._on_goal(None) == GoalResponse.REJECT
    node._episode_busy.release()
    assert node._on_goal(None) == GoalResponse.ACCEPT
    node._episode_busy.release()
