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
Entry-point plugin loader shared by the operator and codec registries.

Each entry point names a module; importing it runs its ``@register_*``
decorators. Imports only :mod:`.errors`, so either registry can use it
without cycles.
"""

from __future__ import annotations

import importlib.metadata as _ilm
import threading

from .errors import ContractValidationError

_loaded_groups: set[str] = set()
_failed_groups: dict[str, ContractValidationError] = {}

# RLock: a plugin module may trigger discovery while being imported.
_scan_lock = threading.RLock()


def load_entry_point_plugins(group: str, noun: str) -> None:
    """
    Import every module advertised under entry-point ``group``, once per process.

    ``noun`` names the plugin kind in the error message. A plugin that fails
    to import raises ContractValidationError, and every later call re-raises
    the same error. Retrying would not help: a module that registered
    something before failing would hit the duplicate-registration guard and
    hide the real error.
    """
    with _scan_lock:
        if group in _loaded_groups:
            return
        if group in _failed_groups:
            raise _failed_groups[group]
        for ep in _ilm.entry_points(group=group):
            try:
                ep.load()
            except Exception as e:
                err = ContractValidationError(
                    f"Failed to load {noun} plugin '{ep.name}' ({ep.value}) from entry-point group '{group}': {e}"
                )
                _failed_groups[group] = err
                raise err from e
        _loaded_groups.add(group)
