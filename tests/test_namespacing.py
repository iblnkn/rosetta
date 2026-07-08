"""Tests for the shared topic-namespacing policy (_topic_namespace_map).

The dataset writer (_apply_namespaces, via port_bags) and the ROS-free
validator path (_aggregate_namespaced_names, via contract_interface) must
derive identical names, or the validators flag drift on datasets their own
contract produced. These tests pin the shared policy on the pure-function
level (no decoder registry / rclpy needed).
"""

import pytest

from rosetta.common.contract import ContractValidationError
from rosetta.common.contract_utils import _aggregate_namespaced_names


def test_multi_topic_key_gets_prefixed():
    out = _aggregate_namespaced_names(
        [
            ("observation.state", "/left_arm/joint_states", ["position.j1"]),
            ("observation.state", "/right_arm/joint_states", ["position.j1"]),
        ]
    )
    assert out["observation.state"] == [
        "left_arm.position.j1",
        "right_arm.position.j1",
    ]


def test_isolated_single_topic_key_stays_bare():
    out = _aggregate_namespaced_names(
        [("observation.state", "/joint_states", ["position.j1"])]
    )
    assert out["observation.state"] == ["position.j1"]


def test_shared_topic_prefixes_single_topic_key_like_the_writer():
    """A topic in a multi-topic group carries its namespace into every key
    it feeds — including single-topic keys. This mirrors the dataset
    writer's global by-topic application; datasets already on disk store
    the prefixed form, so the validator must derive it too."""
    out = _aggregate_namespaced_names(
        [
            ("observation.state", "/left_arm/joint_states", ["position.j1"]),
            ("observation.state", "/right_arm/joint_states", ["position.j1"]),
            ("observation.environment_state", "/left_arm/joint_states", ["effort.e1"]),
        ]
    )
    assert out["observation.environment_state"] == ["left_arm.effort.e1"]


def test_conflicting_namespaces_across_groups_rejected():
    """/a/x/js derives namespace 'a' in one group and 'x' in the other;
    the old silent last-group-wins overwrite made dataset names depend on
    key order, so the ambiguity is now a load-time error."""
    with pytest.raises(ContractValidationError) as exc:
        _aggregate_namespaced_names(
            [
                ("k1", "/a/x/js", ["n1"]),
                ("k1", "/b/x/js", ["n2"]),
                ("k2", "/a/x/js", ["n3"]),
                ("k2", "/a/y/imu", ["n4"]),
            ]
        )
    assert "conflicting derived namespaces" in str(exc.value)
