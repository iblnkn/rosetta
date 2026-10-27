# Copyright 2026 Isaac Blankenau
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

"""The ``kind`` semantic tag: its dimension rule and the untagged-rotation warning."""

import warnings

import pytest

from rosetta.contract.errors import ContractValidationError
from rosetta.contract.schema import _validate_kind

QUATERNION_FIELDS = ["orientation.x", "orientation.y", "orientation.z", "orientation.w"]


def test_default_is_continuous():
    assert _validate_kind(None, ["a", "b"], "ctx") == "continuous"


def test_invalid_kind_errors():
    with pytest.raises(ContractValidationError):
        _validate_kind("rotation_matrix", ["a"], "ctx")


@pytest.mark.parametrize(
    ("kind", "n", "ok"),
    [
        ("quaternion", 4, True),
        ("quaternion", 3, False),
        ("euler_rpy", 3, True),
        ("euler_rpy", 4, False),
        ("rotation_6d", 6, True),
        ("binary", 1, True),
        ("continuous", 10, True),
    ],
)
def test_dim_count_enforced(kind, n, ok):
    names = [f"f{i}" for i in range(n)]
    if ok:
        assert _validate_kind(kind, names, "ctx") == kind
    else:
        with pytest.raises(ContractValidationError):
            _validate_kind(kind, names, "ctx")


@pytest.mark.parametrize(
    ("kind", "names"),
    [("continuous", QUATERNION_FIELDS), (None, ["pose.quat_w"])],
)
def test_rotation_smell_warns_when_untagged(kind, names):
    with pytest.warns(UserWarning):
        _validate_kind(kind, names, "ctx")


@pytest.mark.parametrize(
    ("kind", "names"),
    [("continuous", ["position.j1", "position.j2", "position.j3"]), ("quaternion", QUATERNION_FIELDS)],
)
def test_no_warning_for_plain_joints_or_tagged_rotations(kind, names):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _validate_kind(kind, names, "ctx")
