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

"""Unit tests for the op-plugin pipeline (rosetta.core.ops)."""

import numpy as np
import pytest

from rosetta.core.contract import ContractValidationError, _parse_apply
from rosetta.core.ops import (
    build_op,
    forward_pipeline,
    inverse_pipeline,
    Invertibility,
    Op,
    OP_REGISTRY,
    OpContext,
    register_op,
)

CTX = OpContext()


def test_registry_has_builtin_ops():
    # The op set is intentionally small: value transforms (rad2deg, clamp) and
    # one lossy image transform (resize). Capability grows by adding plugins.
    assert {'rad2deg', 'resize', 'clamp'} <= set(OP_REGISTRY)


def test_rad2deg_roundtrip():
    op = build_op('rad2deg', None, CTX)
    a = np.array([0.0, np.pi / 2, np.pi])
    assert np.allclose(op.forward(a), [0.0, 90.0, 180.0])
    # rad2deg is a true bijection: forward then inverse recovers the input.
    assert np.allclose(inverse_pipeline(forward_pipeline(a, [op]), [op]), a)


def test_resize_forward_shape_hwc_and_hw():
    op = build_op('resize', [2, 2], CTX)
    img = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    assert op.forward(img).shape == (2, 2, 3)
    gray = np.arange(16, dtype=np.uint8).reshape(4, 4)
    assert op.forward(gray).shape == (2, 2)


def test_resize_is_noop_when_already_target_size():
    op = build_op('resize', [4, 4], CTX)
    img = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    assert op.forward(img) is img


@pytest.mark.parametrize('args', [[0, 2], [2, -1], [2], 'x'])
def test_resize_bad_args_raise(args):
    with pytest.raises(ContractValidationError):
        build_op('resize', args, CTX)


def test_resize_is_forward_only():
    op = build_op('resize', [2, 2], CTX)
    assert op.kind is Invertibility.FORWARD_ONLY
    assert op.kind.serveable is False
    with pytest.raises(ContractValidationError):
        op.inverse(np.zeros((2, 2, 3), dtype=np.uint8))


def test_clamp_clips_both_directions():
    op = build_op('clamp', [-1.0, 1.0], CTX)
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
    op = build_op('clamp', [0.0, 2.0], CTX)
    out = inverse_pipeline(np.array([-5.0, 1.0, 9.0]), [op])
    assert np.allclose(out, [0.0, 1.0, 2.0])


def test_clamp_is_bidirectional():
    # BIDIRECTIONAL == "runs in the serve direction but lossy"; clamp is.
    op = build_op('clamp', [0.0, 1.0], CTX)
    assert op.kind is Invertibility.BIDIRECTIONAL
    assert op.kind.serveable is True


@pytest.mark.parametrize('args', [[1.0], [1, 2, 3], 'x', [2.0, 1.0], [float('inf'), 1.0]])
def test_clamp_bad_args_raise(args):
    with pytest.raises(ContractValidationError):
        build_op('clamp', args, CTX)


def test_action_apply_allows_clamp():
    # clamp is BIDIRECTIONAL (serveable), so an action may carry it.
    ops = _parse_apply([{'clamp': [-1.0, 1.0]}], "action 'action'", require_serveable=True)
    assert ops == [('clamp', [-1.0, 1.0])]


def test_action_apply_rejects_forward_only_op_at_load():
    # An action's apply list must be fully serveable; resize is FORWARD_ONLY.
    with pytest.raises(ContractValidationError, match='resize'):
        _parse_apply([{'resize': [2, 2]}], "action 'action'", require_serveable=True)


def test_observation_apply_allows_resize():
    # Observations never run the serve direction, so resize is fine.
    ops = _parse_apply([{'resize': [2, 2]}], 'observations[0]')
    assert ops == [('resize', [2, 2])]


def test_build_op_unknown_name_raises():
    with pytest.raises(ContractValidationError, match='Unknown op'):
        build_op('nope', None, CTX)


# --- Round-trip gate for BIJECTIVE ops -----------------------------------


def test_bijective_op_with_correct_inverse_builds():
    @register_op('_test_negate', kind=Invertibility.BIJECTIVE)
    class _Negate(Op):
        def forward(self, arr):
            return -arr

        def inverse(self, arr):
            return -arr

    try:
        op = build_op('_test_negate', None, CTX)  # gate must pass
        assert np.allclose(op.inverse(op.forward(np.array([1.0, -2.0]))), [1.0, -2.0])
    finally:
        OP_REGISTRY.pop('_test_negate', None)


def test_bijective_op_with_wrong_inverse_fails_round_trip_gate():
    @register_op('_test_broken', kind=Invertibility.BIJECTIVE)
    class _Broken(Op):
        def forward(self, arr):
            return arr * 2.0

        def inverse(self, arr):
            return arr * 2.0  # wrong: should be / 2.0

    try:
        with pytest.raises(ContractValidationError, match='round-trip'):
            build_op('_test_broken', None, CTX)
    finally:
        OP_REGISTRY.pop('_test_broken', None)


def test_bijective_op_honors_sample_input_domain():
    # A domain-restricted op (log needs positive input) round-trips only when
    # sample_input stays in-domain; the default spread (with negatives) would
    # produce NaNs and fail the gate.
    @register_op('_test_log', kind=Invertibility.BIJECTIVE)
    class _Log(Op):
        def forward(self, arr):
            return np.log(arr)

        def inverse(self, arr):
            return np.exp(arr)

        def sample_input(self):
            return np.array([0.1, 1.0, 2.0, 10.0])

    try:
        op = build_op('_test_log', None, CTX)  # gate passes on positive domain
        assert np.allclose(op.inverse(op.forward(np.array([3.0]))), [3.0])
    finally:
        OP_REGISTRY.pop('_test_log', None)


# --- Entry-point op plugin discovery -------------------------------------


def test_discover_ops_loads_entry_point_plugins(monkeypatch):
    import rosetta.core.ops as ops_mod

    class _FakeEP:
        name = 'demo'
        value = 'fake.mod'

        def load(self):
            @register_op('_ep_demo', kind=Invertibility.BIDIRECTIONAL)
            class _Demo(Op):
                def forward(self, arr):
                    return arr

                def inverse(self, arr):
                    return arr

    ops_mod._ops_discovered = False
    monkeypatch.setattr(ops_mod._ilm, 'entry_points', lambda group=None: [_FakeEP()])
    try:
        ops_mod.discover_ops()
        assert '_ep_demo' in OP_REGISTRY  # plugin op resolvable by name
    finally:
        OP_REGISTRY.pop('_ep_demo', None)
        ops_mod._ops_discovered = False


def test_discover_ops_runs_once(monkeypatch):
    import rosetta.core.ops as ops_mod

    calls = []
    ops_mod._ops_discovered = False
    monkeypatch.setattr(
        ops_mod._ilm, 'entry_points', lambda group=None: calls.append(1) or []
    )
    try:
        ops_mod.discover_ops()
        ops_mod.discover_ops()
        assert len(calls) == 1  # idempotent: scanned once per process
    finally:
        ops_mod._ops_discovered = False


def test_discover_ops_raises_on_broken_plugin(monkeypatch):
    import rosetta.core.ops as ops_mod

    class _BadEP:
        name = 'bad'
        value = 'bad.mod'

        def load(self):
            raise ImportError('boom')

    ops_mod._ops_discovered = False
    monkeypatch.setattr(ops_mod._ilm, 'entry_points', lambda group=None: [_BadEP()])
    try:
        with pytest.raises(ContractValidationError, match='op plugin'):
            ops_mod.discover_ops()
    finally:
        ops_mod._ops_discovered = False
