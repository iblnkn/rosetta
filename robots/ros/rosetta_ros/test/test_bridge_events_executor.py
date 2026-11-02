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

"""The bridge on the executor the nodes really spin: rclpy's EventsExecutor.

The other bridge tests spin a SingleThreadedExecutor. EventsExecutor runs
callbacks from C++ and supports less: a subscription callback that takes the
message info, for one, makes its spin raise on the first message. This test
fails if the bridge ever subscribes in a way the nodes' own executor can't
serve.
"""

import numpy as np
from bridge_helpers import bridge_on_graph, obs_spec, spin_until
from rclpy.experimental import EventsExecutor
from sensor_msgs.msg import JointState


def test_observations_reach_the_frame_on_the_events_executor():
    specs = [obs_spec("observation.state", ["position.j1"], "/js_events")]
    with bridge_on_graph("events", specs, [], activate=False, executor=EventsExecutor()) as (
        bridge,
        _node,
        fixture_node,
        executor,
    ):
        pub = fixture_node.create_publisher(JointState, "/js_events", 10)
        msg = JointState(name=["j1"], position=[0.25])

        def arrived():
            pub.publish(msg)
            return np.allclose(bridge.sample_frame()["observation.state"], [0.25])

        assert spin_until(executor, arrived), f"never saw the message, last frame: {bridge.sample_frame()}"
