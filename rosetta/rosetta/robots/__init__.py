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

:mod:`.ros2` is the ROS2 implementation (topic bridge, rosbag2 replay,
nodes, message codecs). A future ecosystem (ROS1, zenoh-native, MQTT, ...)
slots in as a sibling package implementing the same surfaces; nothing on the
policy side changes, because both sides speak only frames.

This side imports only Rosetta's waist (:mod:`rosetta.contract`,
:mod:`rosetta.frames`) — never :mod:`rosetta.policies`.
"""
