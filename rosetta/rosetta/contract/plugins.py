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
Entry-point plugin loading shared by Rosetta's registries.

:func:`load_entry_point_plugins` imports module plugins (operators, codecs):
each entry point names a module, and importing it runs its ``@register_*``
decorators. :func:`instantiate_entry_point` resolves one named class plugin
(frameworks, robot interfaces) and checks it against a protocol. Imports
only :mod:`.errors`, so every registry can use it without cycles.
"""

from __future__ import annotations

import importlib.metadata as _ilm
import threading
from typing import Any

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


def _dist_name(ep: _ilm.EntryPoint) -> str:
    """Name of the distribution that registered ``ep``, best-effort."""
    dist = getattr(ep, "dist", None)
    return dist.name if dist is not None else "(unknown distribution)"


def resolve_entry_point(eps: list[_ilm.EntryPoint], group: str, name: str, noun: str) -> _ilm.EntryPoint:
    """Find the single entry point in ``eps`` (all of ``group``) registered as ``name``.

    ``noun`` names the plugin kind in errors. Each registry fetches ``eps``
    itself, so its tests can fake ``importlib.metadata`` on that registry alone.
    """
    matches = [ep for ep in eps if ep.name == name]
    if not matches:
        known = ", ".join(sorted({ep.name for ep in eps})) or "(none installed)"
        raise ValueError(f"No {noun} '{name}' registered under entry-point group '{group}'. Available: {known}.")
    # importlib.metadata dedups same-named distributions (first on sys.path
    # wins), so this guard only catches two DIFFERENTLY-named dists.
    if len(matches) > 1:
        dists = ", ".join(sorted(_dist_name(ep) for ep in matches))
        raise ValueError(
            f"{noun.capitalize()} '{name}' is registered more than once under entry-point group "
            f"'{group}' (by: {dists}) — resolution would be arbitrary; remove the stale distribution."
        )
    return matches[0]


def instantiate_entry_point(eps: list[_ilm.EntryPoint], group: str, name: str, protocol: type, noun: str) -> Any:
    """Load, instantiate, and structurally validate the ``name`` entry point.

    ``ep.load()`` and the zero-argument construction propagate raw: their
    tracebacks name the failing plugin module/class directly, which no
    wrapper improves on. The registry raises only for what it can diagnose
    better than a traceback: a non-class entry point and a non-conforming
    instance.
    """
    ep = resolve_entry_point(eps, group, name, noun)
    cls = ep.load()
    if not isinstance(cls, type):
        raise TypeError(
            f"Entry point '{name}' in group '{group}' ({_dist_name(ep)}) must be "
            f"a class; got {type(cls).__name__} ({cls!r})."
        )
    obj = cls()
    if not isinstance(obj, protocol):
        # isinstance is the gate; the sweep names the offenders — absent OR
        # present-but-non-callable (e.g. a member set to None fails isinstance
        # yet passes hasattr). __protocol_attrs__ is undocumented but the only
        # runtime member source before Python 3.13's typing.get_protocol_members;
        # Rosetta's plugin protocols are all-methods, so callability is the right test.
        problems = sorted(m for m in protocol.__protocol_attrs__ if not callable(getattr(obj, m, None)))
        detail = (
            f"missing or non-callable: {', '.join(problems)}"
            if problems
            else "all members present and callable; structural check failed on a non-method member"
        )
        raise TypeError(
            f"Entry point '{name}' in group '{group}' ({_dist_name(ep)}) does not "
            f"implement {protocol.__name__} ({detail})."
        )
    return obj
