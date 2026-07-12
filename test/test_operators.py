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

"""Unit tests for the operator-plugin pipeline (rosetta.contract.operators)."""

import numpy as np
import pytest

from rosetta.contract.errors import ContractValidationError
from rosetta.contract.operators import (
    OPERATOR_REGISTRY,
    Invertibility,
    Operator,
    OperatorContext,
    build_operator,
    forward_pipeline,
    inverse_pipeline,
    register_operator,
)
from rosetta.contract.schema import _parse_apply

CTX = OperatorContext()


def test_registry_has_builtin_operators():
    # The operator set is intentionally small: value transforms (rad2deg, clamp) and
    # one lossy image transform (resize). Capability grows by adding plugins.
    assert {"rad2deg", "resize", "clamp"} <= set(OPERATOR_REGISTRY)


def test_rad2deg_roundtrip():
    op = build_operator("rad2deg", None, CTX)
    a = np.array([0.0, np.pi / 2, np.pi])
    assert np.allclose(op.forward(a), [0.0, 90.0, 180.0])
    # rad2deg is a true bijection: forward then inverse recovers the input.
    assert np.allclose(inverse_pipeline(forward_pipeline(a, [op]), [op]), a)


def test_resize_forward_shape_hwc_and_hw():
    op = build_operator("resize", [2, 2], CTX)
    img = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    assert op.forward(img).shape == (2, 2, 3)
    gray = np.arange(16, dtype=np.uint8).reshape(4, 4)
    assert op.forward(gray).shape == (2, 2)


def test_resize_is_noop_when_already_target_size():
    op = build_operator("resize", [4, 4], CTX)
    img = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    assert op.forward(img) is img


@pytest.mark.parametrize("args", [[0, 2], [2, -1], [2], "x"])
def test_resize_bad_args_raise(args):
    with pytest.raises(ContractValidationError):
        build_operator("resize", args, CTX)


def test_resize_is_forward_only():
    op = build_operator("resize", [2, 2], CTX)
    assert op.kind is Invertibility.FORWARD_ONLY
    assert op.kind.serveable is False
    with pytest.raises(ContractValidationError):
        op.inverse(np.zeros((2, 2, 3), dtype=np.uint8))


def test_clamp_clips_both_directions():
    op = build_operator("clamp", [-1.0, 1.0], CTX)
    a = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    expected = [-1.0, -0.5, 0.0, 0.5, 1.0]
    # Clamp bounds the value on both the build (forward) and serve (inverse)
    # directions -- the serve direction (encode, policy command -> ROS) is the
    # safety-critical one.
    assert np.allclose(op.forward(a), expected)
    assert np.allclose(op.inverse(a), expected)


def test_clamp_serve_direction_clips_via_pipeline():
    # inverse_pipeline is what the action/encode path runs; the out-of-range
    # command must come back inside the bound.
    op = build_operator("clamp", [0.0, 2.0], CTX)
    out = inverse_pipeline(np.array([-5.0, 1.0, 9.0]), [op])
    assert np.allclose(out, [0.0, 1.0, 2.0])


def test_clamp_is_bidirectional():
    # BIDIRECTIONAL == "runs in the serve direction but lossy"; clamp is.
    op = build_operator("clamp", [0.0, 1.0], CTX)
    assert op.kind is Invertibility.BIDIRECTIONAL
    assert op.kind.serveable is True


def test_clamp_dict_args():
    # {min, max} is the canonical contract spelling (see stone.yaml).
    op = build_operator("clamp", {"min": -0.5, "max": 0.5}, CTX)
    assert np.allclose(op.forward(np.array([-2.0, 0.1, 2.0])), [-0.5, 0.1, 0.5])


@pytest.mark.parametrize(
    "args",
    [
        [1.0],
        [1, 2, 3],
        "x",
        [2.0, 1.0],
        [float("inf"), 1.0],
        {"min": 0.0},
        {"max": 1.0},
        {"min": 0.0, "max": 1.0, "step": 0.1},
        {"lo": 0.0, "hi": 1.0},
        {"min": 1.0, "max": 0.0},
    ],
)
def test_clamp_bad_args_raise(args):
    with pytest.raises(ContractValidationError):
        build_operator("clamp", args, CTX)


def test_action_apply_allows_clamp():
    # clamp is BIDIRECTIONAL (serveable), so an action may carry it.
    ops = _parse_apply([{"clamp": [-1.0, 1.0]}], "action 'action'", require_serveable=True)
    assert ops == [("clamp", [-1.0, 1.0])]


def test_action_apply_rejects_forward_only_op_at_load():
    # An action's apply list must be fully serveable; resize is FORWARD_ONLY.
    with pytest.raises(ContractValidationError, match="resize"):
        _parse_apply([{"resize": [2, 2]}], "action 'action'", require_serveable=True)


def test_observation_apply_allows_resize():
    # Observations never run the serve direction, so resize is fine.
    ops = _parse_apply([{"resize": [2, 2]}], "observations[0]")
    assert ops == [("resize", [2, 2])]


def test_build_op_unknown_name_raises():
    with pytest.raises(ContractValidationError, match="Unknown operator"):
        build_operator("nope", None, CTX)


# --- Round-trip gate for BIJECTIVE operators -----------------------------------


def test_bijective_op_with_correct_inverse_builds():
    @register_operator("_test_negate", kind=Invertibility.BIJECTIVE)
    class _Negate(Operator):
        def forward(self, arr):
            return -arr

        def inverse(self, arr):
            return -arr

    try:
        op = build_operator("_test_negate", None, CTX)  # gate must pass
        assert np.allclose(op.inverse(op.forward(np.array([1.0, -2.0]))), [1.0, -2.0])
    finally:
        OPERATOR_REGISTRY.pop("_test_negate", None)


def test_bijective_op_with_wrong_inverse_fails_round_trip_gate():
    @register_operator("_test_broken", kind=Invertibility.BIJECTIVE)
    class _Broken(Operator):
        def forward(self, arr):
            return arr * 2.0

        def inverse(self, arr):
            return arr * 2.0  # wrong: should be / 2.0

    try:
        with pytest.raises(ContractValidationError, match="round-trip"):
            build_operator("_test_broken", None, CTX)
    finally:
        OPERATOR_REGISTRY.pop("_test_broken", None)


def test_bijective_op_honors_sample_input_domain():
    # A domain-restricted operator (log needs positive input) round-trips only when
    # sample_input stays in-domain; the default spread (with negatives) would
    # produce NaNs and fail the gate.
    @register_operator("_test_log", kind=Invertibility.BIJECTIVE)
    class _Log(Operator):
        def forward(self, arr):
            return np.log(arr)

        def inverse(self, arr):
            return np.exp(arr)

        def sample_input(self):
            return np.array([0.1, 1.0, 2.0, 10.0])

    try:
        op = build_operator("_test_log", None, CTX)  # gate passes on positive domain
        assert np.allclose(op.inverse(op.forward(np.array([3.0]))), [3.0])
    finally:
        OPERATOR_REGISTRY.pop("_test_log", None)


# --- Entry-point operator plugin discovery -------------------------------------


def test_discover_operators_loads_entry_point_plugins(monkeypatch):
    import rosetta.contract.operators as operators_mod

    class _FakeEP:
        name = "demo"
        value = "fake.mod"

        def load(self):
            @register_operator("_ep_demo", kind=Invertibility.BIDIRECTIONAL)
            class _Demo(Operator):
                def forward(self, arr):
                    return arr

                def inverse(self, arr):
                    return arr

    operators_mod._operators_discovered = False
    monkeypatch.setattr(operators_mod._ilm, "entry_points", lambda group=None: [_FakeEP()])
    try:
        operators_mod.discover_operators()
        assert "_ep_demo" in OPERATOR_REGISTRY  # plugin op resolvable by name
    finally:
        OPERATOR_REGISTRY.pop("_ep_demo", None)
        operators_mod._operators_discovered = False


def test_discover_operators_runs_once(monkeypatch):
    import rosetta.contract.operators as operators_mod

    calls = []
    operators_mod._operators_discovered = False
    monkeypatch.setattr(operators_mod._ilm, "entry_points", lambda group=None: calls.append(1) or [])
    try:
        operators_mod.discover_operators()
        operators_mod.discover_operators()
        assert len(calls) == 1  # idempotent: scanned once per process
    finally:
        operators_mod._operators_discovered = False


def test_discover_operators_raises_on_broken_plugin(monkeypatch):
    import rosetta.contract.operators as operators_mod

    class _BadEP:
        name = "bad"
        value = "bad.mod"

        def load(self):
            raise ImportError("boom")

    operators_mod._operators_discovered = False
    monkeypatch.setattr(operators_mod._ilm, "entry_points", lambda group=None: [_BadEP()])
    try:
        with pytest.raises(ContractValidationError, match="operator plugin"):
            operators_mod.discover_operators()
    finally:
        operators_mod._operators_discovered = False


def test_operators_has_no_contract_dependency():
    """operators must stay upstream of contract: no import of it, ever.

    Guards the errors.py cycle break — contract imports operators at module level
    now, so an operators -> contract import would recreate the circular-import
    problem the errors module exists to prevent. Static source check because
    the package __init__ pulls contract in any live-import test.
    """
    import ast
    from pathlib import Path

    import rosetta.contract.operators as operators_module

    tree = ast.parse(Path(operators_module.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "contract" not in (node.module or ""), ast.dump(node)
        elif isinstance(node, ast.Import):
            assert not any("contract" in a.name for a in node.names), ast.dump(node)
