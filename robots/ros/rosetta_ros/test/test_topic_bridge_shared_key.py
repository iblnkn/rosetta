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

"""Shared-key TopicBridge tests against a real rclpy graph.

Two specs reading the same topic (the README kind example) must both land in
the frame, and action frames must slice per spec, not per topic, publishing
all slices or none.
"""

import numpy as np
from bridge_helpers import act_spec, bridge_on_graph, obs_spec, spin_until, subscribe
from sensor_msgs.msg import JointState

from rosetta.contract.schema import SafetyBehavior


def test_two_specs_same_topic_both_sampled():
    """README kind example: two selectors on one topic concatenate in the frame."""
    specs = [
        obs_spec("observation.state", ["position.j1", "position.j2"], "/js"),
        obs_spec("observation.state", ["velocity.j1"], "/js"),
    ]
    with bridge_on_graph("obs", specs, [], activate=False) as (bridge, _node, fixture_node, executor):
        pub = fixture_node.create_publisher(JointState, "/js", 10)
        msg = JointState()
        msg.name = ["j1", "j2"]
        msg.position = [0.1, 0.2]
        msg.velocity = [1.5, 0.0]

        def frame_complete():
            pub.publish(msg)
            return np.allclose(bridge.sample_frame()["observation.state"], [0.1, 0.2, 1.5])

        assert spin_until(executor, frame_complete), f"never saw full concat, last frame: {bridge.sample_frame()}"


def test_shared_action_key_routes_slices_per_spec():
    """Two action specs sharing 'action' publish their own slice to their topic."""
    specs = [
        act_spec("action", ["position.j1", "position.j2"], "/arm_cmd"),
        act_spec("action", ["position.grip"], "/grip_cmd"),
    ]
    with bridge_on_graph("act", [], specs) as (bridge, _node, fixture_node, executor):
        received = subscribe(bridge, fixture_node, executor, ["/arm_cmd", "/grip_cmd"])

        bridge.publish_frame({"action": np.array([1.0, 2.0, 3.0])})

        assert spin_until(executor, lambda: received["/arm_cmd"] and received["/grip_cmd"])
        assert received["/arm_cmd"][0] == [1.0, 2.0]
        assert received["/grip_cmd"][0] == [3.0]


def test_publish_frame_drops_non_finite_frame_atomically():
    """A NaN command drops the whole frame (no partial publish), then recovers.

    The NaN sits in the second spec's slice, so publishing per spec as it
    encodes would already have commanded /arm_cmd. Nothing may reach the wire,
    the hold cache must not count the frame (a cached NaN would make the hold
    safety send fail instead of sending zeros), and a later finite frame
    publishes normally.
    """
    specs = [
        act_spec("action", ["position.j1", "position.j2"], "/arm_cmd", SafetyBehavior.HOLD),
        act_spec("action", ["position.grip"], "/grip_cmd", SafetyBehavior.HOLD),
    ]
    with bridge_on_graph("nan", [], specs) as (bridge, _node, fixture_node, executor):
        received = subscribe(bridge, fixture_node, executor, ["/arm_cmd", "/grip_cmd"])

        bridge.publish_frame({"action": np.array([1.0, 2.0, np.nan])})
        bridge.send_safety_action()  # nothing reached the wire to hold: zeros

        # Subscriptions are matched and per-publisher order holds, so this
        # sentinel arrives after anything published before it.
        bridge.publish_frame({"action": np.array([4.0, 5.0, 6.0])})
        assert spin_until(executor, lambda: len(received["/arm_cmd"]) >= 2 and len(received["/grip_cmd"]) >= 2)
        assert received["/arm_cmd"] == [[0.0, 0.0], [4.0, 5.0]]
        assert received["/grip_cmd"] == [[0.0], [6.0]]
