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
The robot side: pub/sub ecosystems adapted onto Rosetta's frame stream.

A robot, to Rosetta, is a messy pub/sub system (asynchronous topics, mixed
rates, ecosystem-specific message types) adapted onto two clean surfaces:

- live: a :class:`~rosetta.frames.protocols.FrameIO` — observations
  sampled out on the clock defined by the contract's ``fps``, actions
  published back in;
- recorded: an iterator of the same frame dicts replayed from the
  ecosystem's recording format.

:mod:`.protocols` is what contract loading needs from an interface
(message types, timelines, qos, built-in codecs), and :mod:`.registry` finds
the interface a contract names among the installed ``rosetta.robot_interfaces``
entry points. The ROS2 implementation (topic bridge, rosbag2 replay, nodes,
message codecs) is the ``rosetta_ros`` package. A future ecosystem (ROS1,
zenoh-native, MQTT, ...) is a sibling package implementing the same
surfaces; nothing on the policy side changes, because both sides speak only
frames.

This module imports only :mod:`rosetta.contract`. Interface packages build
on the rest of core as they need.
"""
