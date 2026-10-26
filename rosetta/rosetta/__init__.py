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
Rosetta: translation between pub/sub robots and policy-learning frameworks.

Robots are messy; policies want structure. A **contract**
(:mod:`rosetta.contract` — one YAML per robot) declares how the robot's
pub/sub topics become clean fixed-rate **frames**
(:mod:`rosetta.frames` — ``{contract_key: np.ndarray | str}``) and back.
The **robot side** (:mod:`rosetta.robots`) adapts each pub/sub ecosystem
onto that frame stream — ROS2 today; ROS1, zenoh, MQTT tomorrow. The
**policy side** (:mod:`rosetta.policies`) adapts each learning framework
(LeRobot, ...) to consume it, for dataset writing and
live policy execution. Either side swaps out without touching the other,
because both speak only frames.

The same frame machinery runs live inference and offline bag conversion, so
training data matches inference input sample-for-sample by construction.

Framework adapters live in their own packages and register via entry points
(see :mod:`rosetta.policies`), e.g. ``lerobot_rosetta``.

Public API. Other packages import only these modules, and only the names in
each module's ``__all__``; every other module is internal:

- Contract: ``contract.schema``, ``contract.specs``, ``contract.errors``,
  ``contract.sidecar``, ``contract.operators`` (operator plugins)
- Frames: ``frames.codecs`` (codecs and codec plugins), ``frames.layout``,
  ``frames.naming``, ``frames.protocols``
- Policies: ``policies``
- Robots: ``robots.protocols``, ``robots.registry`` (robot interface plugins)
- ROS 2 node toolkit: ``robots.ros2.rosetta_lifecycle_node``,
  ``robots.ros2.node_utils``, ``robots.ros2.child_goals``, ``robots.ros2.topic_bridge``,
  ``robots.ros2.node_host``, ``robots.ros2.rclpy_utils``,
  ``robots.ros2.field_access``, ``robots.ros2.launch_utils``

The contract, frames and policies modules import without ROS.

Usage::

    from rosetta.contract.schema import load_contract
    from rosetta.contract.specs import iter_observation_specs
"""
