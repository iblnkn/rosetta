"""Tests for the ``python -m rosetta.validate`` CLI surface.

ROS-free (validate.py deliberately avoids rclpy), so this runs in the
training env / CI without a ROS installation.
"""

import textwrap

from rosetta.validate import main

UNIFIED = textwrap.dedent(
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
          topic: /cmd
          type: std_msgs/msg/Float64MultiArray
        selector: { names: [j1] }
    """
)

LEGACY_RECORD = UNIFIED.replace(
    "      stamp: { per-role: { record: header, inference: receive } }",
    "      stamp: header",
)


def test_migrate_without_legacy_paths_is_an_error(tmp_path, capsys):
    """--unified alone runs zero comparisons and must not report [OK]."""
    unified = tmp_path / "unified.yaml"
    unified.write_text(UNIFIED)
    rc = main(["migrate", "--unified", str(unified)])
    assert rc == 2
    captured = capsys.readouterr()
    assert "checks nothing" in captured.err
    assert "[OK]" not in captured.out


def test_migrate_with_matching_record_passes(tmp_path):
    unified = tmp_path / "unified.yaml"
    unified.write_text(UNIFIED)
    record = tmp_path / "record.yaml"
    record.write_text(LEGACY_RECORD)
    rc = main(["migrate", "--unified", str(unified), "--record", str(record)])
    assert rc == 0
