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
errors <- operators <- frames.codecs <- schema <- specs / frames.layout.
"""


class ContractValidationError(ValueError):
    """Raised when contract YAML is invalid."""
