"""Tests for contract_interface's parity with the dataset writer.

The validators compare contract_interface's derived layout against real
artifacts, so its dims must match what build_features/port_bags actually
write — including the `len(names) or 1` fallback for nameless numeric
streams (selector is optional; decoders support the nameless mode).
"""

from rosetta.common.contract import _contract_from_dict
from rosetta.common.contract_utils import contract_interface


def test_nameless_numeric_stream_dim_matches_writer():
    contract = _contract_from_dict(
        {
            "robot_type": "t",
            "fps": 30,
            "observations": [
                {
                    "key": "observation.state",
                    "topic": "/scalar",
                    "type": "std_msgs/msg/Float64",
                }
            ],
            "actions": [
                {
                    "key": "action",
                    "publish": {
                        "topic": "/cmd",
                        "type": "std_msgs/msg/Float64MultiArray",
                    },
                }
            ],
        }
    )
    intf = contract_interface(contract)
    # The writer stores these as shape (1,) (`len(all_names) or 1`), so the
    # validator must expect dim 1, not 0.
    assert intf["state"]["observation.state"] == {"names": [], "dim": 1}
    assert intf["actions"]["action"] == {"names": [], "dim": 1}


def test_named_stream_dim_is_name_count():
    contract = _contract_from_dict(
        {
            "robot_type": "t",
            "fps": 30,
            "observations": [
                {
                    "key": "observation.state",
                    "topic": "/joint_states",
                    "type": "sensor_msgs/msg/JointState",
                    "selector": {"names": ["position.j1", "position.j2"]},
                }
            ],
            "actions": [],
        }
    )
    intf = contract_interface(contract)
    assert intf["state"]["observation.state"]["dim"] == 2
