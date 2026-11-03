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
- :mod:`.builtin_operators`: the built-in ``apply`` operators, registered on
  :mod:`rosetta.operators`.
- :mod:`.interfaces`: the robot interface a contract names, as contract
  loading sees it, and codec discovery.
- :mod:`.naming`: frame-key naming helpers.
- :mod:`.sidecar`: finding a dataset's or checkpoint's contract file.

With :mod:`rosetta.frames`, this is the code the three seams
(:mod:`rosetta.robots`, :mod:`rosetta.data`, :mod:`rosetta.policies`) share.
This package imports only core's base modules (:mod:`rosetta.errors`,
:mod:`rosetta.codecs` and the rest). This ``__init__`` imports nothing;
import from the modules directly.
"""
