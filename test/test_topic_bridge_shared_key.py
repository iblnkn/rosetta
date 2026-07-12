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

Regression guards for the topic-keyed-buffer bugs: two specs reading the same
topic (the README kind example) must both land in the frame, and action frames
must slice per spec — not per topic — with length validation.
"""

import time

import numpy as np
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import LifecycleNode
from rosetta.contract.schema import Align, Channel, Source
from rosetta.contract.specs import ActionStreamSpec, ObservationStreamSpec
from rosetta.robots.ros2.topic_bridge import TopicBridge
from sensor_msgs.msg import JointState


def _obs(key, names, topic):
    return ObservationStreamSpec(
        key=key,
        names=list(names),
        fps=30,
        source=Source(
            channel=Channel(topic=topic, type="sensor_msgs/msg/JointState"),
            align=Align("hold", "receive"),
        ),
        is_image=False,
        image_resize=None,
        dtype="float64",
    )


def _act(key, names, topic):
    return ActionStreamSpec(
        key=key,
        names=list(names),
        fps=30,
        source=Source(
            channel=Channel(topic=topic, type="sensor_msgs/msg/JointState"),
            align=Align("hold", "receive"),
        ),
        dtype="float64",
    )


@pytest.fixture
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


def _spin_until(executor, predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.05)
        if predicate():
            return True
    return False


def test_two_specs_same_topic_both_sampled(ros_context):
    """README kind example: two selectors on one topic concatenate in the frame."""
    specs = [
        _obs("observation.state", ["position.j1", "position.j2"], "/js"),
        _obs("observation.state", ["velocity.j1"], "/js"),
    ]
    node = LifecycleNode("bridge_host_obs")
    pub_node = rclpy.create_node("fixture_pub")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(pub_node)
    try:
        bridge = TopicBridge(specs, [], fps=30)
        bridge.setup(node)

        pub = pub_node.create_publisher(JointState, "/js", 10)
        msg = JointState()
        msg.name = ["j1", "j2"]
        msg.position = [0.1, 0.2]
        msg.velocity = [1.5, 0.0]

        def frame_complete():
            pub.publish(msg)
            frame = bridge.sample_frame()
            return np.allclose(frame["observation.state"], [0.1, 0.2, 1.5])

        assert _spin_until(executor, frame_complete), f"never saw full concat, last frame: {bridge.sample_frame()}"
        bridge.teardown()
    finally:
        executor.remove_node(node)
        executor.remove_node(pub_node)
        node.destroy_node()
        pub_node.destroy_node()


def test_shared_action_key_routes_slices_per_spec(ros_context):
    """Two action specs sharing 'action' publish their own slice to their topic."""
    specs = [
        _act("action", ["position.j1", "position.j2"], "/arm_cmd"),
        _act("action", ["position.grip"], "/grip_cmd"),
    ]
    node = LifecycleNode("bridge_host_act")
    sub_node = rclpy.create_node("fixture_sub")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(sub_node)
    received: dict[str, list[float]] = {}
    try:
        bridge = TopicBridge([], specs, fps=30)
        bridge.setup(node)
        node.trigger_configure()
        node.trigger_activate()

        sub_node.create_subscription(
            JointState, "/arm_cmd", lambda m: received.__setitem__("arm", list(m.position)), 10
        )
        sub_node.create_subscription(
            JointState, "/grip_cmd", lambda m: received.__setitem__("grip", list(m.position)), 10
        )

        def both_received():
            bridge.publish_frame({"action": np.array([1.0, 2.0, 3.0])})
            return "arm" in received and "grip" in received

        assert _spin_until(executor, both_received)
        np.testing.assert_allclose(received["arm"], [1.0, 2.0])
        np.testing.assert_allclose(received["grip"], [3.0])
        bridge.teardown()
    finally:
        executor.remove_node(node)
        executor.remove_node(sub_node)
        node.destroy_node()
        sub_node.destroy_node()


def test_publish_frame_rejects_wrong_length(ros_context):
    specs = [
        _act("action", ["position.j1", "position.j2"], "/arm_cmd"),
        _act("action", ["position.grip"], "/grip_cmd"),
    ]
    node = LifecycleNode("bridge_host_len")
    try:
        bridge = TopicBridge([], specs, fps=30)
        bridge.setup(node)
        with pytest.raises(ValueError, match="length 3"):
            bridge.publish_frame({"action": np.array([1.0, 2.0])})
        bridge.teardown()
    finally:
        node.destroy_node()
