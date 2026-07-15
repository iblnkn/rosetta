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

"""Contract exception types.

Dependency-free leaf of the waist: operators, codecs, and schema all raise
ContractValidationError, and putting it here (rather than in schema) keeps
the module import graph acyclic —
errors <- plugins <- operators <- frames.codecs <- schema <- specs / frames.layout.

Convention: ContractValidationError is for user-fixable problems — an invalid
contract, or an operator/codec plugin it depends on failing to load — raised
at contract load / spec resolution so mistakes fail fast. Plain ValueError
remains the type for internal invariants and for the runtime decode/encode
backstops that load-time validation normally makes unreachable.
"""


class ContractValidationError(ValueError):
    """Raised when a contract, or an operator/codec plugin it references, fails validation.

    Subclasses ValueError so callers catching the generic type still see it.
    """
