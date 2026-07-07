"""Tests for inline {per-role: {...}} value maps in unified contracts.

The single mechanism for role-specific settings: a field's value may be
``{per-role: {record: ..., inference: ...}}`` and the loader substitutes the
active role's value. Legacy ``roles:`` delta blocks are rejected.

ROS-free (rosetta.common.contract deliberately avoids rclpy), so this runs in
the training env / CI without a ROS installation.
"""

import textwrap

import pytest

from rosetta.common.contract import (
    ContractValidationError,
    PER_ROLE_SENTINEL,
    is_unified_contract,
    load_contract,
    load_processor_spec,
    load_unified_contract,
)

MAP_CONTRACT = textwrap.dedent(
    """\
    robot_type: testbot
    fps: 30
    observations:
      - key: observation.state
        topic: /joint_states
        type: sensor_msgs/msg/JointState
        dtype: float64
        selector: { names: [position.j1] }
        align:
          strategy: hold
          stamp: { per-role: { record: header, inference: receive } }
    actions:
      - key: action
        publish:
          topic:
            per-role:
              record: /traj_controller/joint_trajectory
              inference: /mlc/joint_traj
          type: std_msgs/msg/Float64MultiArray
        selector: { names: [j1] }
        safety_behavior: { per-role: { record: none, inference: hold } }
    """
)


def _write(tmp_path, text):
    path = tmp_path / "contract.yaml"
    path.write_text(text)
    return path


def test_map_resolves_per_role(tmp_path):
    path = _write(tmp_path, MAP_CONTRACT)
    rec = load_unified_contract(path, "record")
    inf = load_unified_contract(path, "inference")
    assert rec.observations[0].align.stamp == "header"
    assert inf.observations[0].align.stamp == "receive"
    assert rec.actions[0].publish_topic == "/traj_controller/joint_trajectory"
    assert inf.actions[0].publish_topic == "/mlc/joint_traj"
    assert rec.actions[0].safety_behavior == "none"
    assert inf.actions[0].safety_behavior == "hold"


def test_structural_fork_whole_section(tmp_path):
    """A per-role map value may be a whole stream list (structural fork)."""
    path = _write(
        tmp_path,
        textwrap.dedent(
            """\
            robot_type: testbot
            fps: 30
            observations:
              - key: observation.state
                topic: /joint_states
                type: sensor_msgs/msg/JointState
                dtype: float64
                selector: { names: [position.j1] }
                align: { strategy: hold, stamp: receive }
            actions:
              per-role:
                record:
                  - key: action
                    publish:
                      topic: /traj_controller/joint_trajectory
                      type: trajectory_msgs/msg/JointTrajectory
                    selector: { names: [position.j1] }
                inference:
                  - key: action
                    publish:
                      topic: /dmp/cmd
                      type: std_msgs/msg/Float64MultiArray
                    selector: { names: [g_j1, alpha] }
            """
        ),
    )
    rec = load_unified_contract(path, "record")
    inf = load_unified_contract(path, "inference")
    assert rec.actions[0].publish_topic == "/traj_controller/joint_trajectory"
    assert rec.actions[0].type == "trajectory_msgs/msg/JointTrajectory"
    assert inf.actions[0].publish_topic == "/dmp/cmd"
    assert inf.actions[0].type == "std_msgs/msg/Float64MultiArray"
    assert len(inf.actions[0].selector["names"]) == 2


def test_map_missing_role_raises(tmp_path):
    replaced = MAP_CONTRACT.replace("          inference: /mlc/joint_traj\n", "")
    assert replaced != MAP_CONTRACT  # guard: pattern must have matched
    path = _write(tmp_path, replaced)
    with pytest.raises(ContractValidationError) as exc:
        load_unified_contract(path, "inference")
    msg = str(exc.value)
    assert "no value for role 'inference'" in msg
    assert "actions[key=action].publish.topic" in msg


def test_map_unknown_role_key_raises(tmp_path):
    path = _write(
        tmp_path,
        MAP_CONTRACT.replace("inference: /mlc/joint_traj", "infrence: /mlc/joint_traj"),
    )
    with pytest.raises(ContractValidationError) as exc:
        load_unified_contract(path, "record")
    assert "unknown role name(s) ['infrence']" in str(exc.value)


def test_map_with_sibling_keys_raises(tmp_path):
    replaced = MAP_CONTRACT.replace(
        "      stamp: { per-role: { record: header, inference: receive } }",
        "      stamp: { per-role: { record: header, inference: receive }, extra: 1 }",
    )
    assert "extra: 1" in replaced  # guard: pattern must have matched
    path = _write(tmp_path, replaced)
    with pytest.raises(ContractValidationError) as exc:
        load_unified_contract(path, "record")
    assert "must be the only key" in str(exc.value)


def test_bare_sentinel_string_raises(tmp_path):
    replaced = MAP_CONTRACT.replace(
        "      stamp: { per-role: { record: header, inference: receive } }",
        "      stamp: per-role",
    )
    path = _write(tmp_path, replaced)
    with pytest.raises(ContractValidationError) as exc:
        load_unified_contract(path, "record")
    msg = str(exc.value)
    assert "bare" in msg
    assert "observation.state" in msg


def test_nonempty_roles_delta_raises(tmp_path):
    path = _write(
        tmp_path,
        MAP_CONTRACT
        + textwrap.dedent(
            """\
            roles:
              record: {}
              inference:
                actions: { safety_behavior: hold }
            """
        ),
    )
    with pytest.raises(ContractValidationError) as exc:
        load_unified_contract(path, "record")
    msg = str(exc.value)
    assert "no longer supported" in msg
    assert "['inference']" in msg


def test_empty_roles_block_tolerated(tmp_path):
    path = _write(tmp_path, MAP_CONTRACT + "roles: { record: {}, inference: {} }\n")
    assert load_unified_contract(path, "record").actions[0].safety_behavior == "none"


def test_processor_block_rejects_per_role_marker(tmp_path):
    """The processor block has no role view; markers inside it must not
    leak through load_processor_spec unresolved."""
    path = _write(
        tmp_path,
        MAP_CONTRACT
        + textwrap.dedent(
            """\
            processor:
              steps:
                - registry_name: numpy_image_crop_resize
                  config:
                    resize_size: { per-role: { record: [224, 224], inference: [256, 256] } }
            """
        ),
    )
    with pytest.raises(ContractValidationError) as exc:
        load_processor_spec(path)
    msg = str(exc.value)
    assert "role-independent" in msg
    assert "processor.steps[0].config.resize_size" in msg
    # A marker-free processor block still loads fine.
    clean = _write(tmp_path, MAP_CONTRACT + "processor: { steps: [] }\n")
    assert load_processor_spec(clean) == {"steps": []}


def test_plain_loader_redirects_on_map(tmp_path):
    path = _write(tmp_path, MAP_CONTRACT)
    with pytest.raises(ContractValidationError) as exc:
        load_contract(path)
    assert "load_unified_contract" in str(exc.value)


def test_is_unified_detects_markers_without_roles_block(tmp_path):
    path = _write(tmp_path, MAP_CONTRACT)
    assert is_unified_contract(path)


def test_plain_contract_not_unified_and_loads(tmp_path):
    plain = MAP_CONTRACT.replace(
        "      stamp: { per-role: { record: header, inference: receive } }",
        "      stamp: header",
    ).replace(
        "      topic:\n"
        "        per-role:\n"
        "          record: /traj_controller/joint_trajectory\n"
        "          inference: /mlc/joint_traj",
        "      topic: /traj_controller/joint_trajectory",
    ).replace(
        "    safety_behavior: { per-role: { record: none, inference: hold } }\n", ""
    )
    assert PER_ROLE_SENTINEL not in plain  # guard: all three replaces matched
    path = _write(tmp_path, plain)
    assert not is_unified_contract(path)
    assert load_contract(path).observations[0].align.stamp == "header"
