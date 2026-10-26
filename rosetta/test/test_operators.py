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
IMG_CTX = OperatorContext(is_image=True)  # resize only builds on image streams


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
    op = build_operator("resize", [2, 2], IMG_CTX)
    img = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    assert op.forward(img).shape == (2, 2, 3)
    gray = np.arange(16, dtype=np.uint8).reshape(4, 4)
    assert op.forward(gray).shape == (2, 2)


def test_resize_downsample_pixel_values():
    # Non-square input pins the h/w axis order (a swap would still pass any
    # shape-only or square test) and the exact sampling grid: rows round to
    # [0, 3], cols to [0, 2, 5] (np.round on linspace, 2.5 rounds half-to-even).
    img = np.arange(24, dtype=np.uint8).reshape(4, 6)
    out = build_operator("resize", [2, 3], IMG_CTX).forward(img)
    assert np.array_equal(out, img[[0, 3]][:, [0, 2, 5]])


def test_resize_upsample_is_symmetric():
    # Symmetric rounding: 2x2 -> 4x4 repeats each source row/col evenly
    # ([0, 0, 1, 1]), not skewed toward the top-left ([0, 0, 0, 1]).
    img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    out = build_operator("resize", [4, 4], IMG_CTX).forward(img)
    assert np.array_equal(out, img[[0, 0, 1, 1]][:, [0, 0, 1, 1]])


def test_resize_is_noop_when_already_target_size():
    op = build_operator("resize", [4, 4], IMG_CTX)
    img = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    assert op.forward(img) is img


@pytest.mark.parametrize(
    "args",
    [
        [0, 2],  # zero dim
        [2, -1],  # negative dim
        [2],  # wrong arity
        "x",  # not a list
        [2.5, 3],  # float: rejected, never silently truncated
        [True, 2],  # bool is not an integer dimension
        ["2", "2"],  # strings: rejected, never coerced
        [9000, 10],  # above the sanity ceiling
    ],
)
def test_resize_bad_args_raise(args):
    with pytest.raises(ContractValidationError):
        build_operator("resize", args, IMG_CTX)


def test_resize_rejected_on_non_image_stream():
    # A resize on a state vector would crash on every message at runtime;
    # the ctx.is_image gate turns that into a contract-load error.
    with pytest.raises(ContractValidationError, match="image"):
        build_operator("resize", [2, 2], CTX)


def test_resize_declares_output_hw():
    # Geometry is a declared Operator capability (read at spec resolution),
    # not a name match; non-geometry operators leave it None.
    assert build_operator("resize", [2, 3], IMG_CTX).output_hw == (2, 3)
    assert build_operator("rad2deg", None, CTX).output_hw is None
    assert build_operator("clamp", {"min": 0.0, "max": 1.0}, CTX).output_hw is None


def test_resize_is_forward_only():
    # The base inverse raises NotImplementedError (mirroring forward); contracts
    # can never reach it because _parse_apply rejects FORWARD_ONLY on actions.
    op = build_operator("resize", [2, 2], IMG_CTX)
    assert op.kind is Invertibility.FORWARD_ONLY
    with pytest.raises(NotImplementedError):
        op.inverse(np.zeros((2, 2, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (Invertibility.FORWARD_ONLY, False),
        (Invertibility.BIDIRECTIONAL, True),
        (Invertibility.BIJECTIVE, True),
    ],
)
def test_invertibility_serveable(kind, expected):
    assert kind.serveable is expected


def test_clamp_clips_both_directions():
    op = build_operator("clamp", {"min": -1.0, "max": 1.0}, CTX)
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
    op = build_operator("clamp", {"min": 0.0, "max": 2.0}, CTX)
    out = inverse_pipeline(np.array([-5.0, 1.0, 9.0]), [op])
    assert np.allclose(out, [0.0, 1.0, 2.0])


def test_clamp_is_bidirectional():
    # BIDIRECTIONAL == "runs in the serve direction but lossy"; clamp is.
    op = build_operator("clamp", {"min": 0.0, "max": 1.0}, CTX)
    assert op.kind is Invertibility.BIDIRECTIONAL


def test_clamp_preserves_dtype():
    # Operators preserve input dtype: python-float bounds must not promote a
    # uint8 image to float64 (8x memory) or a float32 state to float64.
    op = build_operator("clamp", {"min": 10, "max": 200}, CTX)
    img = np.array([[0, 128, 255]], dtype=np.uint8)
    out = op.forward(img)
    assert out.dtype == np.uint8
    assert np.array_equal(out, [[10, 128, 200]])
    state = np.array([0.5, -3.0], dtype=np.float32)
    assert op.inverse(state).dtype == np.float32


def test_clamp_propagates_nan():
    # Clamp is a RANGE bound, not a finiteness guard: np.clip passes NaN
    # through. The serve path's finiteness gate (encode_value) is what refuses
    # non-finite commands -- pinned here so nobody "fixes" clamp into a
    # NaN scrubber and weakens that gate's rationale.
    op = build_operator("clamp", {"min": 0.0, "max": 1.0}, CTX)
    out = op.inverse(np.array([np.nan, 0.5, 9.0]))
    assert np.isnan(out[0]) and np.allclose(out[1:], [0.5, 1.0])


@pytest.mark.parametrize(
    "args",
    [
        [0.0, 1.0],  # list form removed: {min, max} is the one spelling
        "x",
        {"min": 0.0},
        {"max": 1.0},
        {"min": 0.0, "max": 1.0, "step": 0.1},
        {"lo": 0.0, "hi": 1.0},
        {"min": 1.0, "max": 0.0},  # lo > hi
        {"min": float("inf"), "max": 1.0},  # non-finite bound
        {"min": True, "max": 1.0},  # bool is not a number here
        {"min": "0", "max": 1.0},  # strings: rejected, never coerced
    ],
)
def test_clamp_bad_args_raise(args):
    with pytest.raises(ContractValidationError):
        build_operator("clamp", args, CTX)


def test_action_apply_allows_clamp():
    # clamp is BIDIRECTIONAL (serveable), so an action may carry it.
    ops = _parse_apply([{"clamp": {"min": -1.0, "max": 1.0}}], "action 'action'", require_serveable=True)
    assert ops == [("clamp", {"min": -1.0, "max": 1.0})]


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


def test_bare_operator_rejects_args():
    # rad2deg uses the base __init__, which rejects any payload -- a stray
    # `rad2deg: [2]` in a contract is a load error, not silently ignored args.
    with pytest.raises(ContractValidationError, match="takes no arguments"):
        build_operator("rad2deg", [2], CTX)


def test_inverse_pipeline_runs_back_to_front():
    # The encode path must undo the pipeline in reverse: with
    # apply: [rad2deg, {clamp: [0, 90]}] the dataset speaks clamped degrees,
    # so serving must clamp IN DEGREES first, then convert deg->rad.
    # Forward-order traversal would deg2rad(180)=pi first (unclamped, pi < 90)
    # -- numerically distinct, so this pins the reversed() in inverse_pipeline.
    ops = [build_operator("rad2deg", None, CTX), build_operator("clamp", {"min": 0.0, "max": 90.0}, CTX)]
    out = inverse_pipeline(np.array([180.0]), ops)
    assert np.allclose(out, [np.pi / 2])


# --- Registration invariants -----------------------------------------------------


def test_register_duplicate_operator_raises():
    @register_operator("_test_dup", kind=Invertibility.FORWARD_ONLY)
    class _First(Operator):
        def forward(self, arr):
            return arr

    try:
        with pytest.raises(ValueError, match="already registered"):

            @register_operator("_test_dup", kind=Invertibility.FORWARD_ONLY)
            class _Second(Operator):
                def forward(self, arr):
                    return arr

        assert OPERATOR_REGISTRY["_test_dup"] is _First  # first registration untouched
    finally:
        OPERATOR_REGISTRY.pop("_test_dup", None)


def test_register_operator_override_replaces():
    @register_operator("_test_dup", kind=Invertibility.FORWARD_ONLY)
    class _First(Operator):
        def forward(self, arr):
            return arr

    try:

        @register_operator("_test_dup", kind=Invertibility.FORWARD_ONLY, override=True)
        class _Second(Operator):
            def forward(self, arr):
                return arr

        assert OPERATOR_REGISTRY["_test_dup"] is _Second
    finally:
        OPERATOR_REGISTRY.pop("_test_dup", None)


@pytest.mark.parametrize("kind", [Invertibility.BIDIRECTIONAL, Invertibility.BIJECTIVE])
def test_serveable_operator_without_inverse_rejected_at_registration(kind):
    # A serveable tier promises a serve direction; forgetting inverse must fail
    # at plugin import, not mid-serve on the action/safety encode path.
    with pytest.raises(TypeError, match="inverse"):

        @register_operator("_test_no_inverse", kind=kind)
        class _NoInverse(Operator):
            def forward(self, arr):
                return arr

    assert "_test_no_inverse" not in OPERATOR_REGISTRY


def test_operator_without_forward_rejected_at_registration():
    with pytest.raises(TypeError, match="forward"):

        @register_operator("_test_no_forward", kind=Invertibility.FORWARD_ONLY)
        class _NoForward(Operator):
            pass

    assert "_test_no_forward" not in OPERATOR_REGISTRY


def test_forward_only_operator_without_inverse_allowed():
    @register_operator("_test_fwd_only", kind=Invertibility.FORWARD_ONLY)
    class _FwdOnly(Operator):
        def forward(self, arr):
            return arr

    try:
        assert OPERATOR_REGISTRY["_test_fwd_only"] is _FwdOnly
    finally:
        OPERATOR_REGISTRY.pop("_test_fwd_only", None)


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


def test_round_trip_gate_wraps_operator_exception():
    # A domain-restricted BIJECTIVE op that forgets to override sample_input
    # must fail with a ContractValidationError pointing at sample_input, not a
    # raw IndexError/FloatingPointError escaping build_operator.
    @register_operator("_test_gate_crash", kind=Invertibility.BIJECTIVE)
    class _Crash(Operator):
        def forward(self, arr):
            return arr[:, ::-1]  # needs 2-D input; default sample is 1-D

        def inverse(self, arr):
            return arr[:, ::-1]

    try:
        with pytest.raises(ContractValidationError, match="sample_input"):
            build_operator("_test_gate_crash", None, CTX)
    finally:
        OPERATOR_REGISTRY.pop("_test_gate_crash", None)


def test_round_trip_gate_rejects_empty_sample_input():
    # allclose on empty arrays is vacuously True; the gate must not be
    # satisfiable by verifying nothing.
    @register_operator("_test_gate_empty", kind=Invertibility.BIJECTIVE)
    class _Empty(Operator):
        def forward(self, arr):
            return arr

        def inverse(self, arr):
            return arr

        def sample_input(self):
            return np.array([])

    try:
        with pytest.raises(ContractValidationError, match="empty"):
            build_operator("_test_gate_empty", None, CTX)
    finally:
        OPERATOR_REGISTRY.pop("_test_gate_empty", None)


def test_round_trip_gate_rejects_shape_changing_round_trip():
    # A scalar return broadcasts against the sample in allclose; the shape
    # check must catch it instead of letting broadcasting mask the bug.
    @register_operator("_test_gate_shape", kind=Invertibility.BIJECTIVE)
    class _Collapse(Operator):
        def forward(self, arr):
            return arr

        def inverse(self, arr):
            return np.zeros(())  # allclose(0-d, x) broadcasts

        def sample_input(self):
            return np.array([0.0, 0.0])

    try:
        with pytest.raises(ContractValidationError, match="shape"):
            build_operator("_test_gate_shape", None, CTX)
    finally:
        OPERATOR_REGISTRY.pop("_test_gate_shape", None)


# --- Entry-point operator plugin discovery -------------------------------------


def _fake_ilm(monkeypatch, entry_points):
    """Point the shared plugin loader at a fake entry-point list, latches cleared."""
    import types

    import rosetta.contract.plugins as plugins_mod

    monkeypatch.setattr(plugins_mod, "_loaded_groups", set())
    monkeypatch.setattr(plugins_mod, "_failed_groups", {})
    monkeypatch.setattr(plugins_mod, "_ilm", types.SimpleNamespace(entry_points=lambda group=None: entry_points))


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

    _fake_ilm(monkeypatch, [_FakeEP()])
    try:
        operators_mod.discover_operators()
        assert "_ep_demo" in OPERATOR_REGISTRY  # plugin op resolvable by name
    finally:
        OPERATOR_REGISTRY.pop("_ep_demo", None)


def test_discover_operators_runs_once(monkeypatch):
    import rosetta.contract.operators as operators_mod

    calls = []

    class _CountingEP:
        name = "counter"
        value = "fake.mod"

        def load(self):
            calls.append(1)

    _fake_ilm(monkeypatch, [_CountingEP()])
    operators_mod.discover_operators()
    operators_mod.discover_operators()
    assert len(calls) == 1  # idempotent: scanned once per process


def test_discover_operators_raises_on_broken_plugin(monkeypatch):
    import rosetta.contract.operators as operators_mod

    class _BadEP:
        name = "bad"
        value = "bad.mod"

        def load(self):
            raise ImportError("boom")

    _fake_ilm(monkeypatch, [_BadEP()])
    with pytest.raises(ContractValidationError, match="operator plugin"):
        operators_mod.discover_operators()


def test_discover_operators_failure_is_latched(monkeypatch):
    # A failed scan latches: every later call re-raises the ORIGINAL error,
    # even if the entry-point list changes afterwards (see the codec
    # twin in test_codec_registry.py for the rationale).
    import rosetta.contract.operators as operators_mod

    class _BadEP:
        name = "bad"
        value = "bad.mod"

        def load(self):
            raise ImportError("boom")

    class _GoodEP:
        name = "good"
        value = "fake.mod"

        def load(self):
            @register_operator("_ep_recovered", kind=Invertibility.BIDIRECTIONAL)
            class _Recovered(Operator):
                def forward(self, arr):
                    return arr

                def inverse(self, arr):
                    return arr

    entry_points = [_BadEP()]
    _fake_ilm(monkeypatch, entry_points)
    try:
        with pytest.raises(ContractValidationError, match="boom") as first:
            operators_mod.discover_operators()
        entry_points[:] = [_GoodEP()]  # "fixing" the environment in-process...
        with pytest.raises(ContractValidationError) as second:
            operators_mod.discover_operators()  # ...does not help: same error, no rescan
        assert second.value is first.value
        assert "_ep_recovered" not in OPERATOR_REGISTRY  # never loaded
    finally:
        OPERATOR_REGISTRY.pop("_ep_recovered", None)


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
