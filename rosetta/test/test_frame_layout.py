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

"""FrameLayout: shared-key concatenation, its inverse (split), zero-fill and features.

Multiple specs sharing one key concatenate into a flat vector in declaration
order; split undoes it for actions. The specs are built by hand, so the type
strings are labels only.
"""

import numpy as np
import pytest

from rosetta.contract.schema import Align, Channel, Source
from rosetta.contract.specs import ActionStreamSpec, ObservationStreamSpec
from rosetta.errors import ContractValidationError
from rosetta.frames.layout import FrameLayout, _feature


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


@pytest.mark.parametrize("dtype", ["int32", "int64", "bool"])
def test_shared_nonfloat_keys_concatenate(dtype):
    # bool/int keys concatenate every stream, like float keys.
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
    # bool/int zero-fill must match the spec's width, not default to one value.
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
    feats = layout.features()
    frame = layout.assemble([None] * len(specs))  # all missing -> zero-fill

    for key, feat in feats.items():
        if feat["dtype"] == "string":
            assert frame[key] == ""
            continue
        assert frame[key].shape == tuple(feat["shape"]), key

    assert feats["observation.state"]["names"] == ["j1", "j2", "grip"]
    assert feats["observation.images.cam"]["shape"] == (48, 64, 3)
    assert frame["observation.images.cam"].dtype == np.uint8


def test_string_feature_is_a_lerobot_string():
    feats = FrameLayout([obs_spec("task2", [], topic="/str", dtype="string")]).features()
    assert feats["task2"] == {"dtype": "string", "shape": (1,), "names": None}


def test_build_feature_rejects_numeric_specs():
    # A numeric key's feature must cover the whole concatenated vector, which
    # one spec cannot know; only FrameLayout.features() builds it.
    with pytest.raises(ValueError, match="FrameLayout.features"):
        _feature(obs_spec("observation.state", ["a", "b"]))


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
    feats = FrameLayout(specs).features()
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
    # Short or long vectors must raise, not slice silently.
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
    # A select-less numeric spec is a scalar by contract (spec.dim == 1).
    layout = FrameLayout([act_spec("action", [])])
    (part,) = layout.split({"action": np.array([1.5])})
    np.testing.assert_array_equal(part, [1.5])


def test_split_rejects_unknown_keys():
    # A policy adapter emitting a misnamed key must fail loudly, not be
    # silently part-consumed.
    layout = FrameLayout([act_spec("action", ["j1"])])
    with pytest.raises(ValueError, match="unknown key"):
        layout.split({"action": np.zeros(1), "action.typo": np.zeros(1)})


def test_split_emits_float64_preserving_nonfinite():
    # split must emit float64 and leave the wire-type cast to encoders, after
    # encode_value's finiteness gate. Casting an int32 key here would turn a
    # policy's NaN into INT_MIN, finite garbage that passes the gate.
    layout = FrameLayout([act_spec("action", ["j1", "j2"], dtype="int32")])
    (part,) = layout.split({"action": np.array([np.nan, 1.5])})
    assert part.dtype == np.float64
    assert np.isnan(part[0])  # survives to the gate
    assert part[1] == 1.5  # not truncated by an int cast


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


@pytest.mark.parametrize(
    "spec",
    [
        lambda topic: obs_spec("observation.x", [], topic=topic, dtype="string"),
        lambda topic: image_spec("observation.x", topic=topic),
    ],
    ids=["string", "image"],
)
def test_shared_non_numeric_key_rejected(spec):
    with pytest.raises(ContractValidationError, match="all-numeric"):
        FrameLayout([spec("/1"), spec("/2")])


def test_unsupported_numeric_dtype_rejected():
    # Backstop for hand-built specs; contract-loaded dtypes are already
    # vocabulary-checked in schema._validate_dtype.
    with pytest.raises(ContractValidationError, match="Unsupported dtype"):
        FrameLayout([obs_spec("observation.state", ["a"], dtype="complex64")])


def test_assemble_wrong_value_count_rejected():
    # The guard on the positional contract (values[i] pairs with specs[i]).
    layout = FrameLayout([obs_spec("observation.state", ["a", "b"])])
    with pytest.raises(ValueError, match="1 specs"):
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


def test_mixed_role_shared_key_rejected():
    """A key fed by both observation-shaped and action-shaped specs has no
    single role — routing (state vs action column) would depend on which
    slice a consumer inspects. Reachable from YAML: sections are parsed
    independently, so observations.foo + actions.foo meet here."""
    with pytest.raises(ContractValidationError, match="observation-shaped.*action-shaped"):
        FrameLayout([obs_spec("foo", ["a"]), act_spec("foo", ["b"])])


def test_role_comes_from_section_not_key_name():
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


# ---------- tensors: declared shapes and device values ----------


def tensor_spec(key, shape, *, topic="/features", dtype="float32"):
    source = Source(
        channel=Channel(topic=topic, type="sensor_msgs/msg/JointState"), align=Align("hold", "receive"), shape=shape
    )
    return obs_spec(key, [], topic=topic, dtype=dtype, source=source)


def test_numeric_values_are_copies_of_the_held_sample():
    # A hold strategy serves the same array every tick until the next message,
    # so a consumer that normalizes in place must not change it.
    layout = FrameLayout([obs_spec("observation.state", ["a", "b"])])
    held = np.array([1.0, 2.0], dtype=np.float32)
    layout.assemble([held])["observation.state"] -= 10
    np.testing.assert_array_equal(layout.assemble([held])["observation.state"], [1.0, 2.0])


def test_tensor_key_keeps_its_declared_shape():
    layout = FrameLayout([tensor_spec("observation.features", (2, 3))])
    frame = layout.assemble([np.arange(6, dtype=np.float64)])
    assert frame["observation.features"].shape == (2, 3)
    assert frame["observation.features"].dtype == np.float32
    assert layout["observation.features"].shape == (2, 3)


def test_missing_tensor_zero_fills_at_its_shape():
    frame = FrameLayout([tensor_spec("observation.features", (2, 3))]).assemble([None])
    np.testing.assert_array_equal(frame["observation.features"], np.zeros((2, 3), np.float32))


def test_tensor_value_of_the_wrong_size_is_rejected():
    with pytest.raises(ValueError, match="observation.features"):
        FrameLayout([tensor_spec("observation.features", (2, 3))]).assemble([np.zeros(5)])


def test_tensor_feature_declares_its_shape():
    feats = FrameLayout([tensor_spec("observation.features", (4, 2, 2))]).features()
    assert feats["observation.features"] == {"dtype": "float32", "shape": (4, 2, 2), "names": None}


def test_device_value_alone_in_its_key_stays_on_its_device():
    from fake_arrays import FakeDeviceArray

    val = FakeDeviceArray(np.zeros((2, 3), np.float32))
    assert FrameLayout([tensor_spec("observation.features", (2, 3))]).assemble([val])["observation.features"] is val


def test_device_value_of_another_dtype_is_refused_rather_than_copied():
    from fake_arrays import FakeDeviceArray

    with pytest.raises(ValueError, match="device dtype float64, not float32"):
        FrameLayout([tensor_spec("observation.features", (2, 3))]).assemble([FakeDeviceArray(np.zeros((2, 3)))])


def test_shared_key_with_a_device_part_concatenates_on_the_host():
    from fake_arrays import FakeDeviceArray

    specs = [obs_spec("observation.state", ["a", "b"], topic="/arm"), obs_spec("observation.state", ["g"], topic="/g")]
    frame = FrameLayout(specs).assemble([FakeDeviceArray(np.array([1.0, 2.0], np.float32)), np.array([3.0])])
    assert isinstance(frame["observation.state"], np.ndarray)
    np.testing.assert_array_equal(frame["observation.state"], np.array([1, 2, 3], np.float32))
