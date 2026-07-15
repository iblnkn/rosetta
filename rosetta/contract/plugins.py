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
Shared entry-point plugin loader for Rosetta's open registries.

Both plugin surfaces -- operators (``rosetta.operators``, see
:mod:`.operators`) and codecs (``rosetta.codecs``, see
:mod:`rosetta.frames.codecs`) -- discover third-party extensions the same
way: each entry point's value is a module path, and importing that module
runs its ``@register_*`` decorators. This module is the one implementation
of that scan; the registries own only their group name.

Like :mod:`.errors`, this is a dependency-free leaf so every registry can
import it without cycles: ``errors <- plugins <- operators <- frames.codecs
<- schema <- specs``.
"""

from __future__ import annotations

import importlib.metadata as _ilm
import threading

from .errors import ContractValidationError

_loaded_groups: set[str] = set()
_failed_groups: dict[str, ContractValidationError] = {}

# RLock, not Lock: ep.load() imports arbitrary plugin modules, which may
# themselves touch discovery at import time -- a plain lock would deadlock.
_scan_lock = threading.RLock()


def load_entry_point_plugins(group: str, noun: str) -> None:
    """
    Import every module advertised under entry-point ``group``. Idempotent per group.

    A plugin that fails to import is a hard error, not a silent skip -- a
    half-registered plugin set would surface later as a confusing "Unknown
    operator"/"No decoder registered". A failure is latched just like a
    success: every subsequent call re-raises the original error, so a broken
    plugin environment is fatal and stable until the process restarts. (A
    rescan could not recover anyway: a plugin module that registered
    something before failing mid-import leaves its registrations behind, so
    re-executing it would hit the duplicate-registration guard and mask the
    real error.)

    Args:
    ----
        group: Entry-point group to scan (e.g. ``"rosetta.operators"``).
        noun: What a plugin in this group provides (e.g. ``"operator"``),
            used in the failure message.

    Raises:
    ------
        ContractValidationError: If any advertised plugin fails to import --
            the same error on every call for a group that has already failed.

    """
    with _scan_lock:
        if group in _loaded_groups:
            return
        if group in _failed_groups:
            raise _failed_groups[group]
        for ep in _ilm.entry_points(group=group):
            try:
                ep.load()  # imports the module -> runs its @register_* decorators
            except Exception as e:
                err = ContractValidationError(
                    f"Failed to load {noun} plugin '{ep.name}' ({ep.value}) from entry-point group '{group}': {e}"
                )
                _failed_groups[group] = err  # latch: fail the same way every time
                raise err from e
        _loaded_groups.add(group)
