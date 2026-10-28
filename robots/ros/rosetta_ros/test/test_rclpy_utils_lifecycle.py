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

"""lifecycle_state_label reads rclpy's private lifecycle state machine.

Against a real rclpy LifecycleNode on purpose: rclpy exposes no public
synchronous state accessor, so the helper reaches into ``_state_machine``, and
only a real node notices when a ROS release reshapes it. The labels it returns
feed ``is_active``/``is_configured``, which the LeRobot adapters gate on.
"""

import pytest
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn

from rosetta_ros.rclpy_utils import LIFECYCLE_CONFIGURED_LABELS, lifecycle_state_label


class _ObservingNode(LifecycleNode):
    """Records the label seen from inside each transition callback."""

    def __init__(self):
        super().__init__("lifecycle_label_probe")
        self.seen: dict[str, str] = {}

    def on_activate(self, state):
        self.seen["activate"] = lifecycle_state_label(self)
        return super().on_activate(state)

    def on_deactivate(self, state):
        self.seen["deactivate"] = lifecycle_state_label(self)
        return super().on_deactivate(state)


@pytest.fixture
def node(rclpy_ctx):
    n = _ObservingNode()
    yield n
    n.destroy_node()


def test_labels_follow_the_transitions(node):
    labels = [lifecycle_state_label(node)]
    for trigger in (
        node.trigger_configure,
        node.trigger_activate,
        node.trigger_deactivate,
        node.trigger_cleanup,
        node.trigger_shutdown,
    ):
        assert trigger() == TransitionCallbackReturn.SUCCESS
        labels.append(lifecycle_state_label(node))
    assert labels == ["unconfigured", "inactive", "active", "inactive", "unconfigured", "finalized"]


def test_a_node_mid_transition_still_counts_as_configured(node):
    """Code running inside activate/deactivate sees a transitional label, and
    is_configured must not read that as "not configured"."""
    node.trigger_configure()
    node.trigger_activate()
    node.trigger_deactivate()
    assert node.seen == {"activate": "activating", "deactivate": "deactivating"}
    assert set(node.seen.values()) <= LIFECYCLE_CONFIGURED_LABELS
