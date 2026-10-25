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
The contract: one YAML file per robot declaring what to translate.

- :mod:`.schema`: ``load_contract`` and the document types (from :mod:`.model`).
- :mod:`.specs`: the runtime ``StreamSpec`` family and ``iter_*_specs``.
- :mod:`.operators`, :mod:`.builtin_operators`: the ``apply`` pipeline.
- :mod:`.plugins`: entry-point loader for operators and codecs.
- :mod:`.sidecar`: finding a dataset's or checkpoint's contract file.

With :mod:`rosetta.frames`, this is the only code the robot side
(:mod:`rosetta.robots`) and the policy side (:mod:`rosetta.policies`) share.
This ``__init__`` imports nothing (an import here would create a cycle with
the codec registry); import from the modules directly.
"""
