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

pytestmark = pytest.mark.usefixtures("scratch_registries")

CTX = OperatorContext()
IMG_CTX = OperatorContext(is_image=True)  # resize only builds on image streams


def test_rad2deg_roundtrip():
    op = build_operator("rad2deg", None, CTX)
    a = np.array([0.0, np.pi / 2, np.pi])
    assert np.allclose(op.forward(a), [0.0, 90.0, 180.0])
    # rad2deg is a true bijection: forward then inverse recovers the input.
    assert np.allclose(inverse_pipeline(forward_pipeline(a, [op]), [op]), a)


@pytest.mark.parametrize("shape", [(4, 6), (4, 6, 3)], ids=["hw", "hwc"])
def test_resize_downsample_pixel_values(shape):
    # Non-square input pins the h/w axis order (a swap would still pass any
    # shape-only or square test) and the exact sampling grid: rows round to
    # [0, 3], cols to [0, 2, 5] (np.round on linspace, 2.5 rounds half-to-even).
    img = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    out = build_operator("resize", [2, 3], IMG_CTX).forward(img)
    assert np.array_equal(out, img[[0, 3]][:, [0, 2, 5]])


def test_resize_upsample_is_symmetric():
    # Symmetric rounding: 2x2 -> 4x4 repeats each source row/col evenly
    # ([0, 0, 1, 1]), not skewed toward the top-left ([0, 0, 0, 1]).
    img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    out = build_operator("resize", [4, 4], IMG_CTX).forward(img)
    assert np.array_equal(out, img[[0, 0, 1, 1]][:, [0, 0, 1, 1]])


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ([0, 2], "must be in"),  # zero dim
        ([2, -1], "must be in"),  # negative dim
        ([2], "expects"),  # wrong arity
        ("x", "expects"),  # not a list
        ([2.5, 3], "must be integers"),  # float: rejected, never silently truncated
        ([True, 2], "must be integers"),  # bool is not an integer dimension
        (["2", "2"], "must be integers"),  # strings: rejected, never coerced
        ([9000, 10], "must be in"),  # above the sanity ceiling
    ],
)
def test_resize_bad_args_raise(args, match):
    with pytest.raises(ContractValidationError, match=f"resize operator.*{match}"):
        build_operator("resize", args, IMG_CTX)


def test_clamp_clips_both_directions():
    op = build_operator("clamp", {"min": -1.0, "max": 1.0}, CTX)
    a = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    expected = [-1.0, -0.5, 0.0, 0.5, 1.0]
    # Clamp bounds the value on both the build (forward) and serve (inverse)
    # directions -- the serve direction (encode, policy command -> ROS) is the
    # safety-critical one.
    assert np.allclose(op.forward(a), expected)
    assert np.allclose(inverse_pipeline(a, [op]), expected)


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


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ([0.0, 1.0], "expects"),
        ("x", "expects"),
        ({"min": 0.0}, "expects"),
        ({"max": 1.0}, "expects"),
        ({"min": 0.0, "max": 1.0, "step": 0.1}, "expects"),
        ({"lo": 0.0, "hi": 1.0}, "expects"),
        ({"min": 1.0, "max": 0.0}, "requires lo <= hi"),
        ({"min": float("inf"), "max": 1.0}, "must be finite"),
        ({"min": True, "max": 1.0}, "must be numbers"),  # bool is not a number here
        ({"min": "0", "max": 1.0}, "must be numbers"),  # strings: rejected, never coerced
    ],
)
def test_clamp_bad_args_raise(args, match):
    with pytest.raises(ContractValidationError, match=f"clamp operator.*{match}"):
        build_operator("clamp", args, CTX)


def test_bare_operator_rejects_args():
    # rad2deg uses the base __init__, which rejects any payload -- a stray
    # `rad2deg: [2]` in a contract is a load error, not silently ignored args.
    with pytest.raises(ContractValidationError, match="takes no arguments"):
        build_operator("rad2deg", [2], CTX)


# --- Registration invariants -----------------------------------------------------


def test_register_duplicate_operator_raises():
    @register_operator("_test_dup", kind=Invertibility.FORWARD_ONLY)
    class _First(Operator):
        def forward(self, arr):
            return arr

    with pytest.raises(ValueError, match="already registered"):

        @register_operator("_test_dup", kind=Invertibility.FORWARD_ONLY)
        class _Second(Operator):
            def forward(self, arr):
                return arr

    assert OPERATOR_REGISTRY["_test_dup"] is _First  # first registration untouched


def test_register_operator_override_replaces():
    @register_operator("_test_dup", kind=Invertibility.FORWARD_ONLY)
    class _First(Operator):
        def forward(self, arr):
            return arr

    @register_operator("_test_dup", kind=Invertibility.FORWARD_ONLY, override=True)
    class _Second(Operator):
        def forward(self, arr):
            return arr

    assert OPERATOR_REGISTRY["_test_dup"] is _Second


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


# --- Round-trip gate for BIJECTIVE operators -----------------------------------


def test_bijective_op_with_wrong_inverse_fails_round_trip_gate():
    @register_operator("_test_broken", kind=Invertibility.BIJECTIVE)
    class _Broken(Operator):
        def forward(self, arr):
            return arr * 2.0

        def inverse(self, arr):
            return arr * 2.0  # wrong: should be / 2.0

    with pytest.raises(ContractValidationError, match="round-trip"):
        build_operator("_test_broken", None, CTX)


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

    op = build_operator("_test_log", None, CTX)  # gate passes on positive domain
    assert np.allclose(op.inverse(op.forward(np.array([3.0]))), [3.0])


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

    with pytest.raises(ContractValidationError, match="sample_input"):
        build_operator("_test_gate_crash", None, CTX)


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

    with pytest.raises(ContractValidationError, match="empty"):
        build_operator("_test_gate_empty", None, CTX)


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

    with pytest.raises(ContractValidationError, match="shape"):
        build_operator("_test_gate_shape", None, CTX)
