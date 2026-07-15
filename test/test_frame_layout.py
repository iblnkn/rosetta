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

"""Tests for FrameLayout: the canonical shared-key concatenation layout.

Pins the flagship contract behavior (multiple specs sharing one key
concatenate into a flat vector) and its inverse, including the README's
same-topic-twice example, non-float dtypes, zero-fill shapes, and the
validation rules for shared keys. Pure Python — no ROS required.
"""

import numpy as np
import pytest
from rosetta.contract.errors import ContractValidationError
from rosetta.contract.schema import Align, Channel, Source
from rosetta.contract.specs import ActionStreamSpec, ObservationStreamSpec
from rosetta.frames.layout import FrameLayout


def obs_spec(key, names, *, topic="/t", dtype="float32", namespace=None, **over):
    kwargs = dict(
        key=key,
        names=list(names),
        fps=30,
        source=Source(
            channel=Channel(topic=topic, type="sensor_msgs/msg/JointState"),
            align=Align("hold", "receive"),
        ),
        is_image=False,
        image_resize=None,
        dtype=dtype,
        namespace=namespace,
    )
    kwargs.update(over)
    return ObservationStreamSpec(**kwargs)


def image_spec(key, *, topic="/cam", resize=(48, 64)):
    return obs_spec(key, [], topic=topic, dtype="video", is_image=True, image_resize=resize)


def act_spec(key, names, *, topic="/cmd", dtype="float64", namespace=None):
    return ActionStreamSpec(
        key=key,
        names=list(names),
        fps=30,
        source=Source(
            channel=Channel(topic=topic, type="sensor_msgs/msg/JointState"),
            align=Align("hold", "receive"),
        ),
        dtype=dtype,
        namespace=namespace,
    )


# ---------- assemble: shared-key concatenation ----------


def test_shared_float_key_concatenates_in_declaration_order():
    specs = [
        obs_spec("observation.state", ["j1", "j2", "j3"], topic="/arm"),
        obs_spec("observation.state", ["grip"], topic="/gripper"),
    ]
    layout = FrameLayout(specs)
    frame = layout.assemble([np.array([1.0, 2.0, 3.0]), np.array([4.0])])
    np.testing.assert_array_equal(frame["observation.state"], np.array([1, 2, 3, 4], dtype=np.float32))


def test_same_topic_two_specs_readme_example():
    # The README kind example: position and orientation slices of /ee_pose.
    specs = [
        obs_spec("observation.state", ["x", "y", "z"], topic="/ee_pose"),
        obs_spec("observation.state", ["qx", "qy", "qz", "qw"], topic="/ee_pose"),
        obs_spec("observation.state", ["grip"], topic="/gripper"),
    ]
    layout = FrameLayout(specs)
    frame = layout.assemble([np.arange(3.0), np.arange(10.0, 14.0), np.array([0.5])])
    np.testing.assert_array_equal(
        frame["observation.state"],
        np.array([0, 1, 2, 10, 11, 12, 13, 0.5], dtype=np.float32),
    )


@pytest.mark.parametrize("dtype", ["int32", "int64", "bool"])
def test_shared_nonfloat_keys_concatenate(dtype):
    # Regression: the old aggregate_frame kept only the first stream for
    # bool/int keys.
    specs = [
        obs_spec("signal", ["a", "b"], topic="/s1", dtype=dtype),
        obs_spec("signal", ["c"], topic="/s2", dtype=dtype),
    ]
    layout = FrameLayout(specs)
    frame = layout.assemble([np.array([1, 0]), np.array([1])])
    assert frame["signal"].shape == (3,)
    np.testing.assert_array_equal(np.asarray(frame["signal"], dtype=np.int64), [1, 0, 1])


def test_interleaved_keys_are_key_major():
    # Declaration A, B, A: key order is first occurrence; A's slices stay
    # together in declaration order.
    specs = [
        obs_spec("a", ["a1"], topic="/1"),
        obs_spec("b", ["b1"], topic="/2"),
        obs_spec("a", ["a2"], topic="/3"),
    ]
    layout = FrameLayout(specs)
    assert layout.keys == ("a", "b")
    assert [(s.start, s.dim) for s in layout["a"].slices] == [(0, 1), (1, 1)]
    frame = layout.assemble([np.array([1.0]), np.array([2.0]), np.array([3.0])])
    np.testing.assert_array_equal(frame["a"], [1, 3])
    np.testing.assert_array_equal(frame["b"], [2])


def test_missing_stream_zero_fills_at_spec_dim():
    # Regression: bool/int zero-fill used to be np.zeros(1) regardless of names.
    specs = [
        obs_spec("signal", ["a", "b"], topic="/s1", dtype="int32"),
        obs_spec("signal", ["c"], topic="/s2", dtype="int32"),
    ]
    layout = FrameLayout(specs)
    frame = layout.assemble([None, np.array([7])])
    np.testing.assert_array_equal(frame["signal"], [0, 0, 7])
    assert frame["signal"].dtype == np.int32


def test_assemble_shapes_match_features():
    specs = [
        obs_spec("observation.state", ["j1", "j2"], topic="/arm"),
        obs_spec("observation.state", ["grip"], topic="/gripper"),
        obs_spec("observation.env", [], topic="/scalar"),  # select-less single
        image_spec("observation.images.cam"),
        obs_spec("task2", [], topic="/str", dtype="string"),
    ]
    layout = FrameLayout(specs)
    feats = layout.lerobot_features()
    frame = layout.assemble([None] * len(specs))  # all missing -> zero-fill

    for key, feat in feats.items():
        if feat["dtype"] == "string":
            assert frame[key] == ""
            continue
        assert frame[key].shape == tuple(feat["shape"]), key

    assert feats["observation.state"]["names"] == ["j1", "j2", "grip"]
    assert feats["observation.images.cam"]["shape"] == (48, 64, 3)


def test_duplicate_rendered_names_rejected():
    # Uniqueness must hold on the RENDERED name: (namespace=None, "b.data")
    # and (namespace="b", "data") are distinct pairs that flatten identically.
    specs = [
        obs_spec("observation.state", ["b.data"], topic="/r"),
        obs_spec("observation.state", ["data"], topic="/r/b", namespace="b"),
    ]
    with pytest.raises(ContractValidationError, match="b.data"):
        FrameLayout(specs)


def test_namespaced_names_in_features():
    specs = [
        obs_spec("observation.state", ["pos"], topic="/arm/state", namespace="arm"),
        obs_spec("observation.state", ["pos"], topic="/base/state", namespace="base"),
    ]
    feats = FrameLayout(specs).lerobot_features()
    assert feats["observation.state"]["names"] == ["arm.pos", "base.pos"]


def test_string_and_image_pass_through():
    specs = [
        image_spec("observation.images.cam"),
        obs_spec("task2", [], topic="/str", dtype="string"),
    ]
    layout = FrameLayout(specs)
    img = np.ones((48, 64, 3), dtype=np.uint8)
    frame = layout.assemble([img, "pick up the block"])
    np.testing.assert_array_equal(frame["observation.images.cam"], img)
    assert frame["task2"] == "pick up the block"


# ---------- split: the inverse ----------


def test_split_round_trips_assemble():
    specs = [
        act_spec("action", ["j1", "j2", "j3"], topic="/arm_cmd"),
        act_spec("action", ["grip"], topic="/grip_cmd"),
        act_spec("action.aux", ["x", "y"], topic="/aux_cmd"),
    ]
    layout = FrameLayout(specs)
    values = [np.array([1.0, 2.0, 3.0]), np.array([4.0]), np.array([5.0, 6.0])]
    frame = layout.assemble(values)
    parts = layout.split(frame)
    assert len(parts) == len(specs)
    for part, value in zip(parts, values, strict=False):
        np.testing.assert_array_equal(part, value)


def test_split_validates_total_length():
    # Regression: short/long vectors used to slice silently.
    specs = [
        act_spec("action", ["j1", "j2"], topic="/arm_cmd"),
        act_spec("action", ["grip"], topic="/grip_cmd"),
    ]
    layout = FrameLayout(specs)
    with pytest.raises(ValueError, match="length 3"):
        layout.split({"action": np.zeros(2)})
    with pytest.raises(ValueError, match="length 3"):
        layout.split({"action": np.zeros(5)})


def test_split_missing_key_raises():
    layout = FrameLayout([act_spec("action", ["j1"])])
    with pytest.raises(KeyError, match="action"):
        layout.split({"other": np.zeros(1)})


def test_split_selectless_single_spec_is_scalar():
    # A select-less numeric spec is a scalar by contract (spec.dim == 1);
    # split enforces that instead of passing dynamic widths through.
    layout = FrameLayout([act_spec("action", [])])
    (part,) = layout.split({"action": np.array([1.5])})
    np.testing.assert_array_equal(part, [1.5])
    with pytest.raises(ValueError, match="length 1"):
        layout.split({"action": np.array([1.0, 2.0, 3.0])})


def test_split_rejects_unknown_keys():
    # A policy adapter emitting a misnamed key must fail loudly, not be
    # silently part-consumed.
    layout = FrameLayout([act_spec("action", ["j1"])])
    with pytest.raises(ValueError, match="unknown key"):
        layout.split({"action": np.zeros(1), "action.typo": np.zeros(1)})


def test_split_emits_float64_preserving_nonfinite():
    # Regression: split() used to cast to the key's dtype BEFORE
    # encode_value's finiteness gate — for an int32 action key that
    # laundered a policy's NaN into INT_MIN, finite garbage that sailed
    # through the gate onto hardware. split always emits float64; encoders
    # do the wire-type cast after the gate.
    layout = FrameLayout([act_spec("action", ["j1", "j2"], dtype="int32")])
    (part,) = layout.split({"action": np.array([np.nan, 1.5])})
    assert part.dtype == np.float64
    assert np.isnan(part[0])  # survives to the gate
    assert part[1] == 1.5  # not truncated by an int cast

    layout32 = FrameLayout([act_spec("action", ["j1"], dtype="float32")])
    (part32,) = layout32.split({"action": np.array([1.0])})
    assert part32.dtype == np.float64


# ---------- validation ----------


def test_shared_key_mixed_dtype_rejected():
    specs = [
        obs_spec("observation.state", ["a"], topic="/1", dtype="float32"),
        obs_spec("observation.state", ["b"], topic="/2", dtype="float64"),
    ]
    with pytest.raises(ContractValidationError, match="different dtypes"):
        FrameLayout(specs)


def test_shared_key_selectless_spec_rejected():
    specs = [
        obs_spec("observation.state", ["a"], topic="/1"),
        obs_spec("observation.state", [], topic="/2"),
    ]
    with pytest.raises(ContractValidationError, match="no select"):
        FrameLayout(specs)


def test_shared_string_key_rejected():
    specs = [
        obs_spec("task2", [], topic="/1", dtype="string"),
        obs_spec("task2", [], topic="/2", dtype="string"),
    ]
    with pytest.raises(ContractValidationError, match="all-numeric"):
        FrameLayout(specs)


def test_shared_image_key_rejected():
    specs = [
        image_spec("observation.images.cam", topic="/cam1"),
        image_spec("observation.images.cam", topic="/cam2"),
    ]
    with pytest.raises(ContractValidationError, match="all-numeric"):
        FrameLayout(specs)


def test_unsupported_numeric_dtype_rejected():
    # Backstop for hand-built specs; contract-loaded dtypes are already
    # vocabulary-checked in schema._validate_dtype.
    with pytest.raises(ContractValidationError, match="Unsupported dtype"):
        FrameLayout([obs_spec("observation.state", ["a"], dtype="float16")])


def test_assemble_wrong_value_count_rejected():
    # The guard on the positional contract (values[i] pairs with specs[i]).
    layout = FrameLayout([obs_spec("observation.state", ["a", "b"])])
    with pytest.raises(ValueError, match="got 2 values for 1 specs"):
        layout.assemble([np.zeros(2), np.zeros(2)])


def test_assemble_wrong_slice_length_rejected():
    # A decoder emitting the wrong length must not silently shift every
    # later slice of a shared key.
    specs = [
        obs_spec("observation.state", ["j1", "j2", "j3"], topic="/arm"),
        obs_spec("observation.state", ["grip"], topic="/gripper"),
    ]
    layout = FrameLayout(specs)
    with pytest.raises(ValueError, match="/arm"):
        layout.assemble([np.array([1.0, 2.0]), np.array([4.0])])


def test_assemble_selectless_stream_must_be_scalar():
    # Select-less numeric streams are scalars; a multi-element decoder
    # output errors with guidance instead of varying the key's width.
    layout = FrameLayout([obs_spec("observation.env", [], topic="/scalar")])
    frame = layout.assemble([np.array([3.5])])
    assert frame["observation.env"].shape == (1,)
    with pytest.raises(ValueError, match="add select"):
        layout.assemble([np.array([1.0, 2.0, 3.0])])


def test_np_dtype_rejects_non_numeric_keys():
    layout = FrameLayout([obs_spec("task2", [], topic="/str", dtype="string")])
    with pytest.raises(ValueError, match="numeric keys only"):
        _ = layout["task2"].np_dtype


def test_action_spec_default_dtype_builds():
    # Actions carry a resolved dtype (default float64) so custom-decoder
    # actions no longer need a registered decoder to build a layout.
    layout = FrameLayout([act_spec("action", ["j1"])])
    assert layout["action"].dtype == "float64"


def test_mixed_role_shared_key_rejected():
    """A key fed by both observation-shaped and action-shaped specs has no
    single role — routing (state vs action column) would depend on which
    slice a consumer inspects. Reachable from YAML: sections are parsed
    independently, so observations.foo + actions.foo used to merge here."""
    with pytest.raises(ContractValidationError, match="observation-shaped.*action-shaped"):
        FrameLayout([obs_spec("foo", ["a"]), act_spec("foo", ["b"])])


def test_key_layout_is_action():
    """Role comes from the spec's TYPE (which section declared it), never the
    key's spelling: an observation named 'action_history' is not an action."""
    layout = FrameLayout(
        [
            obs_spec("action_history", ["a", "b"]),
            act_spec("cmd_vel", ["vx", "wz"]),
        ]
    )
    assert layout["action_history"].is_action is False
    assert layout["cmd_vel"].is_action is True
