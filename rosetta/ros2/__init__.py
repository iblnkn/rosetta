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
Rosetta ROS2 adapter layer.

Binds the framework-agnostic :mod:`rosetta.core` to ROS2: message decoders /
encoders (registered into the core codec registry), QoS / timestamp helpers,
bag IO, and the ROS2 nodes. Imports here may pull in rclpy / rosidl / rosbag2,
so keep this package out of the core import graph.
"""
