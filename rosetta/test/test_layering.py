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

"""Core's layers: every import points down, the seams never meet, no cycles.

    L3 seams     rosetta.robots    rosetta.data    rosetta.policies
    L2 frames    rosetta.frames
    L1 contract  rosetta.contract
    L0 base      rosetta.errors, rosetta.plugins, rosetta.arrays,
                 rosetta.operators, rosetta.codecs

A module may import a module in a lower layer, or one in its own package
(``rosetta.contract.specs`` may import ``rosetta.contract.model``). The base
modules may import each other. Nothing imports upward, and no seam imports
another seam. The module graph has no cycles.

Every import counts: module level, inside functions, and under
``TYPE_CHECKING``. A lazy import is still a dependency.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "rosetta"

LAYERS = {
    "rosetta": 0,  # the package's __init__ only
    "rosetta.errors": 0,
    "rosetta.plugins": 0,
    "rosetta.arrays": 0,
    "rosetta.operators": 0,
    "rosetta.codecs": 0,
    "rosetta.contract": 1,
    "rosetta.frames": 2,
    "rosetta.robots": 3,
    "rosetta.data": 3,
    "rosetta.policies": 3,
}
"""Layer of each package or base module; a module takes its closest listed ancestor's."""

SEAMS = frozenset({"rosetta.robots", "rosetta.data", "rosetta.policies"})


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _modules() -> dict[str, Path]:
    return {_module_name(p): p for p in sorted(ROOT.rglob("*.py"))}


def _owner(module: str) -> str | None:
    """The closest ancestor (or the module itself) named in LAYERS.

    The ``rosetta`` package itself owns only its ``__init__``, so a new
    top-level module has no layer until LAYERS names it.
    """
    if module == "rosetta":
        return "rosetta"
    parts = module.split(".")
    for i in range(len(parts), 1, -1):
        candidate = ".".join(parts[:i])
        if candidate in LAYERS:
            return candidate
    return None


def _imports(module: str, path: Path, known: set[str]) -> set[str]:
    """Every rosetta module ``module`` imports, resolved to a module that exists."""
    is_pkg = path.name == "__init__.py"
    package = module if is_pkg else module.rpartition(".")[0]
    found: set[str] = set()

    def resolve(name: str) -> str | None:
        # The longest prefix that is a real module.
        parts = name.split(".")
        for i in range(len(parts), 0, -1):
            candidate = ".".join(parts[:i])
            if candidate in known:
                return candidate
        return None

    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "rosetta":
                    target = resolve(alias.name)
                    if target:
                        found.add(target)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                if node.level > 1:
                    base = base[: -(node.level - 1)]
                stem = ".".join(base + ([node.module] if node.module else []))
            else:
                stem = node.module or ""
            if stem.split(".")[0] != "rosetta":
                continue
            for alias in node.names:
                # `from pkg import submodule` imports the submodule.
                target = resolve(f"{stem}.{alias.name}") if f"{stem}.{alias.name}" in known else resolve(stem)
                if target:
                    found.add(target)
    found.discard(module)
    return found


def _graph() -> dict[str, set[str]]:
    modules = _modules()
    known = set(modules)
    return {m: _imports(m, p, known) for m, p in modules.items()}


def _violation(src: str, dst: str) -> str | None:
    """Why ``src`` may not import ``dst``, or None when it may."""
    src_owner, dst_owner = _owner(src), _owner(dst)
    if src_owner == dst_owner:
        return None
    src_layer, dst_layer = LAYERS[src_owner], LAYERS[dst_owner]
    if src_layer == 0 and dst_layer == 0:
        return None
    if dst_layer < src_layer:
        return None
    if src_owner in SEAMS and dst_owner in SEAMS:
        return "seams never import each other"
    if dst_layer == src_layer:
        return "same layer, different package"
    return "imports upward"


def _cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """One representative cycle per strongly connected component of size > 1."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[list[str]] = []
    counter = [0]

    def visit(v: str) -> None:
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in sorted(graph.get(v, ())):
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.append(w)
                if w == v:
                    break
            if len(component) > 1:
                components.append(sorted(component))

    for v in sorted(graph):
        if v not in index:
            visit(v)
    return components


def test_every_module_has_a_layer():
    unowned = sorted(m for m in _modules() if _owner(m) is None)
    assert not unowned, f"Modules with no layer; add them to LAYERS: {unowned}"


def test_imports_point_down():
    graph = _graph()
    found = sorted(
        f"{src} -> {dst} ({why})" for src, dsts in graph.items() for dst in dsts if (why := _violation(src, dst))
    )
    assert not found, "Layering violations:\n" + "\n".join(found)


def test_no_import_cycles():
    cycles = _cycles(_graph())
    assert not cycles, "Import cycles:\n" + "\n".join(" <-> ".join(c) for c in cycles)
