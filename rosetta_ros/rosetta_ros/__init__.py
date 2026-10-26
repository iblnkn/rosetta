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

"""
The ROS2 robot interface for Rosetta.

Binds Rosetta's core (:mod:`rosetta.contract`, :mod:`rosetta.frames`,
:mod:`rosetta.policies`) to a ROS2 graph: message decoders/encoders
(registered into the codec registry), QoS / timestamp helpers, the live
:class:`~.topic_bridge.TopicBridge` (a :class:`~rosetta.frames.protocols.FrameIO`),
the ROS2 nodes and the node toolkit apps build on. Offline tooling (rosbag2
replay and the bag->dataset porter) lives in :mod:`.offline`.

Contract loading reaches this package through :class:`.interface.Ros2Interface`,
registered as the ``ros2`` robot interface. That module, :mod:`.decoders`,
:mod:`.encoders`, :mod:`.field_access` and :mod:`.timelines` import without
ROS (rclpy/rosidl are call-time dependencies), so finding the interface and
registering its codecs stays cheap; :mod:`.ingest` stays unit-testable
without a ROS install.
"""
