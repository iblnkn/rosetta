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

A contract binds channels to frame keys. Every frame-clock entry reads
``channel (provides) -> align (chooses a timeline; mandatory) -> select
(field projection) -> apply (an operator pipeline) -> the mapping key`` — actions
are the same pipeline right-to-left, plus teleop roles, tasks, and safety.
Everything else in Rosetta executes what this package declares. (At runtime,
select/apply are pure per-message transforms and commute with alignment, so
they run once per message at ingest — observationally identical.)

Modules: :mod:`.schema` is what the contract *says* (the typed document
model of the YAML, validation, ``load_contract``); :mod:`.specs` is what the
runtime *consumes* (the ``StreamSpec`` family and the ``iter_*_specs``
resolution that produces them); :mod:`.operators` is the ``apply``-pipeline
framework (registry, invertibility tiers, round-trip gate) and
:mod:`.builtin_operators` the in-tree plugins (``rad2deg``, ``resize``,
``clamp``) registered into it; :mod:`.errors`.

Together with :mod:`rosetta.frames`, this package is Rosetta's waist: the
robot side (:mod:`rosetta.robots`) and the policy side
(:mod:`rosetta.policies`) import only these two packages and never each
other. Import from the concrete modules (``rosetta.contract.schema``,
``rosetta.contract.specs``); this ``__init__`` stays import-free to keep the
schema <-> codec-registry module graph acyclic.
"""
