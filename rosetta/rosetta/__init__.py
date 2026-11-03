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
(:mod:`rosetta.frames` — ``{contract_key: array | str}``) and back.
Three seams plug into that frame stream, and each swaps out without
touching the others, because all three speak only frames:

- **robots** (:mod:`rosetta.robots`): each pub/sub ecosystem adapted onto
  frames (ROS 2 today).
- **data** (:mod:`rosetta.data`): frames written as a dataset format for
  training (LeRobot, ...).
- **policies** (:mod:`rosetta.policies`): a trained policy run against live
  frames (LeRobot, ...).

This package holds the protocols and registries for all three and imports
no robot middleware and no learning framework. ``test/test_layering.py``
pins the module layers: base modules, then :mod:`rosetta.contract`, then
:mod:`rosetta.frames`, then the three seams, which never import each
other.

The same frame machinery runs live inference and offline bag conversion, so
training data matches inference input sample-for-sample by construction.

Implementations live in their own packages and register via entry points:
robot interfaces such as ``rosetta_ros`` (see :mod:`rosetta.robots`), dataset
writers such as ``lerobot_dataset_rosetta`` (see :mod:`rosetta.data`) and
policy runners such as ``lerobot_rosetta`` (see :mod:`rosetta.policies`).

Public API. Other packages import only these modules, and only the names in
each module's ``__all__``; every other module is internal:

- Base: ``errors``, ``plugins`` (entry-point loading), ``arrays`` (device
  arrays), ``operators`` (operator plugins), ``codecs`` (codec registry and
  codec plugins)
- Contract: ``contract.schema``, ``contract.specs``, ``contract.interfaces``
  (robot interfaces as contract loading sees them), ``contract.naming``,
  ``contract.sidecar``
- Frames: ``frames.codecs`` (decoding and encoding), ``frames.layout``,
  ``frames.protocols``, ``frames.resample`` (fixed-rate resampling)
- Data: ``data`` (dataset writer plugins)
- Policies: ``policies`` (policy runner plugins)
- Robots: ``robots.protocols``, ``robots.registry`` (robot interface plugins)

Usage::

    from rosetta.contract.schema import load_contract
    from rosetta.contract.specs import iter_observation_specs
"""
