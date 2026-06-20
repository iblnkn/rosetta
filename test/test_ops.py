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
    OP_REGISTRY,
    OpContext,
    build_op,
    forward_pipeline,
    inverse_pipeline,
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


def test_resize_is_not_invertible():
    op = build_op('resize', [2, 2], CTX)
    assert op.invertible is False
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


def test_clamp_is_invertible_flag():
    # invertible == "can run in the serve direction"; clamp can.
    assert build_op('clamp', [0.0, 1.0], CTX).invertible is True


@pytest.mark.parametrize('args', [[1.0], [1, 2, 3], 'x', [2.0, 1.0], [float('inf'), 1.0]])
def test_clamp_bad_args_raise(args):
    with pytest.raises(ContractValidationError):
        build_op('clamp', args, CTX)


def test_action_apply_allows_clamp():
    # clamp is invertible, so an action may carry it.
    ops = _parse_apply([{'clamp': [-1.0, 1.0]}], "action 'action'", require_invertible=True)
    assert ops == [('clamp', [-1.0, 1.0])]


def test_action_apply_rejects_noninvertible_op_at_load():
    # An action's apply list must be fully serveable; resize is not.
    with pytest.raises(ContractValidationError, match='resize'):
        _parse_apply([{'resize': [2, 2]}], "action 'action'", require_invertible=True)


def test_observation_apply_allows_resize():
    # Observations never run the serve direction, so resize is fine.
    ops = _parse_apply([{'resize': [2, 2]}], 'observations[0]')
    assert ops == [('resize', [2, 2])]


def test_build_op_unknown_name_raises():
    with pytest.raises(ContractValidationError, match='Unknown op'):
        build_op('nope', None, CTX)
