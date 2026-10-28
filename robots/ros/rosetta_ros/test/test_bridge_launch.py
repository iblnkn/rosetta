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

"""launch_testing integration exemplar: TopicBridge fed by an external publisher.

This is the repo's template for ROS2-idiomatic integration
tests (docs.ros.org Jazzy "Integration testing with launch_testing"): a launch
description brings up fixture processes and active tests run against the live
graph. Run via colcon test or, from the package root,
``PYTHONPATH=test:$PYTHONPATH launch_test test/test_bridge_launch.py`` (the
test dir must be importable for ``bridge_helpers``).

The scenario pins the shared-key path end to end: an external ``ros2 topic
pub`` process feeds one JointState topic read by two specs sharing
``observation.state``; the bridge frame must carry the concatenated vector.
"""

import time
import unittest

import launch
import launch_testing.actions
import numpy as np
import pytest
from bridge_helpers import obs_spec

TOPIC = "/launch_test/js"


@pytest.mark.launch_test
def generate_test_description():
    import os

    actions = []
    # Cross-process comms under rmw_zenoh need a running router; bring one up
    # as part of the test description so the test is self-contained.
    if os.environ.get("RMW_IMPLEMENTATION", "") == "rmw_zenoh_cpp":
        actions.append(
            launch.actions.ExecuteProcess(
                cmd=["ros2", "run", "rmw_zenoh_cpp", "rmw_zenohd"],
                output="screen",
            )
        )

    fixture_pub = launch.actions.ExecuteProcess(
        cmd=[
            "ros2",
            "topic",
            "pub",
            "-r",
            "30",
            TOPIC,
            "sensor_msgs/msg/JointState",
            "{name: [j1, j2], position: [0.1, 0.2], velocity: [1.5, 0.0]}",
        ],
        output="screen",
    )
    actions += [fixture_pub, launch_testing.actions.ReadyToTest()]
    return launch.LaunchDescription(actions)


class TestBridgeAgainstLiveGraph(unittest.TestCase):
    # launch_testing runs these cases outside pytest's fixture machinery, so
    # conftest's session-scoped rclpy_ctx never fires here: init ourselves,
    # but only when nobody else (e.g. a full-suite pytest run) already has.
    @classmethod
    def setUpClass(cls):
        import rclpy

        cls._owns_rclpy = not rclpy.ok()
        if cls._owns_rclpy:
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        import rclpy

        if cls._owns_rclpy:
            rclpy.try_shutdown()

    def test_shared_key_frame_from_external_publisher(self):
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.lifecycle import LifecycleNode

        from rosetta_ros.topic_bridge import TopicBridge

        specs = [
            obs_spec("observation.state", ["position.j1", "position.j2"], TOPIC),
            obs_spec("observation.state", ["velocity.j1"], TOPIC),
        ]
        node = LifecycleNode("launch_test_bridge_host")
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        bridge = TopicBridge(specs, [], fps=30)
        try:
            bridge.setup(node)

            frame = None
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.1)
                frame = bridge.sample_frame()
                if np.allclose(frame["observation.state"], [0.1, 0.2, 1.5]):
                    break

            np.testing.assert_allclose(frame["observation.state"], [0.1, 0.2, 1.5])
        finally:
            bridge.teardown()
            executor.remove_node(node)
            node.destroy_node()
