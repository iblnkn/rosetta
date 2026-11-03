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

"""The contract sidecar: its file name, and auditing one before it loads.

A dataset carries its contract as a sidecar file next to the data, named
:data:`CONTRACT_SIDECAR_FILENAME`. Where a dataset format puts it, and how a
checkpoint leads back to it, belong to that format's writer and runner (see
:meth:`rosetta.policies.PolicyRunner.find_contract`).
"""

from __future__ import annotations

from pathlib import Path

import yaml

__all__ = ["CONTRACT_SIDECAR_FILENAME", "scan_inline_codec_paths"]

CONTRACT_SIDECAR_FILENAME = "rosetta_contract.yaml"
"""Bare sidecar filename; tar-shard dataset layouts put it at the output root."""


def scan_inline_codec_paths(contract_path: Path | str) -> list[tuple[str, str]]:
    """Raw-YAML scan for inline codec declarations, importing nothing.

    Returns every ``("decoder"|"encoder", path)`` pair with a non-empty string
    value, anywhere in the document. Meant to run before ``load_contract``,
    which imports those paths, so it ignores the schema: over-matching is
    fine, under-matching is not.
    """
    found: list[tuple[str, str]] = []

    def walk(obj) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in ("decoder", "encoder") and isinstance(value, str) and value.strip():
                    found.append((key, value))
                walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(yaml.safe_load(Path(contract_path).read_text()))
    return found
