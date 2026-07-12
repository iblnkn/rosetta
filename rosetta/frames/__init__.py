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
Frames: the interlingua both sides of Rosetta speak.

A *frame* is one synchronized sample of every contract key at a single
instant — a dict ``{contract_key: np.ndarray | str}`` covering all
modalities at once (images, joint state, ...), emitted once per tick of the
clock defined by the contract's ``fps``. This is the LeRobot sense of the
word (one row of a ``LeRobotDataset``, ``add_frame()``), NOT a ROS/tf
coordinate frame (``base_link``, ``map``, ...) — this package has nothing to
do with spatial transforms.

The frame dict is the only currency exchanged between the robot side and the
policy side. This package owns everything both sides must agree on, with no
ROS or policy-framework dependencies:

- :mod:`.layout` — ``FrameLayout``: contract keys <-> flat vectors, the
  single source of truth for frame assembly and splitting.
- :mod:`.resample` — ``StreamBuffer``: async messages -> fixed-rate frames.
  Live inference and offline bag conversion share it, so recorded and live
  frames match sample-for-sample by construction.
- :mod:`.codecs` — the encoder/decoder registry (message <-> array).
- :mod:`.protocols` — ``FrameSource``/``FrameSink``/``FrameIO``: the
  duck-typed surface a live robot presents to a policy.
- :mod:`.naming` — shared frame-key naming helpers.

Together with :mod:`rosetta.contract`, this package is Rosetta's waist: the
robot side (:mod:`rosetta.robots`) and the policy side
(:mod:`rosetta.policies`) import only these two packages and never each
other. Import from the concrete modules; this ``__init__`` stays import-free
to keep the schema <-> codec-registry module graph acyclic.
"""
