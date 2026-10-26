# Packages

The [`rosetta`](https://github.com/iblnkn/rosetta) repository holds the first four
packages. Each LeRobot package has its own repository.

| Package | Contains |
|---|---|
| [`rosetta`](https://github.com/iblnkn/rosetta/tree/main/rosetta) | Contract loader, frame pipeline, the recorder, policy runner and keyboard nodes, the Python API for building apps, `rosetta_port`, example contracts. |
| [`rosetta_interfaces`](https://github.com/iblnkn/rosetta/tree/main/rosetta_interfaces) | `RecordEpisode`, `RunPolicy` actions. `StartRecording`, `StartPolicy` services. |
| [`rosetta_hil`](https://github.com/iblnkn/rosetta/tree/main/rosetta_hil) | The human-in-the-loop manager, an app built on the recorder, the policy runner and the Python API. |
| [`rosetta_hil_interfaces`](https://github.com/iblnkn/rosetta/tree/main/rosetta_hil_interfaces) | `ManageEpisode` action, `StartHILEpisode` service. |
| [`lerobot_rosetta`](https://github.com/iblnkn/lerobot-rosetta) | The LeRobot adapter. Registers `lerobot` under `rosetta.dataset_writers` and `rosetta.policy_runners`. Executables `rosetta_policy_server`, `rosetta_classifier_server`. |
| [`lerobot_robot_rosetta`](https://github.com/iblnkn/lerobot-robot-rosetta) | A LeRobot `Robot` plugin. `--robot.type=rosetta` in LeRobot's own CLIs. |
| [`lerobot_teleoperator_rosetta`](https://github.com/iblnkn/lerobot-teleoperator-rosetta) | A LeRobot `Teleoperator` plugin. Experimental. |

All seven are at version 0.2.0.

## Entry-point groups

| Group | Registers | Read by |
|---|---|---|
| `rosetta.codecs` | A module that calls `register_decoder` and `register_encoder`. | Contract load. |
| `rosetta.operators` | A module that calls `register_operator`. | Contract load. |
| `rosetta.dataset_writers` | A class implementing `DatasetWriter`. | `rosetta_port --framework`. |
| `rosetta.policy_runners` | A class implementing `PolicyRunner`. | `policy_runner_node` parameter `framework`. |

## Dependencies

`rosetta` depends on ROS 2 packages resolvable by `rosdep`, plus `numpy`,
`pyyaml` and `rosbag2_storage_mcap`. It doesn't import a learning framework.

`lerobot_rosetta` needs `lerobot`, `torch`, `grpcio` and
`lerobot_robot_rosetta`. `lerobot_robot_rosetta` needs `lerobot`. `rosdep`
has a key for `grpcio` but none for `lerobot` or `torch`, so install those
with `pip`. [`rosetta_ws`](https://github.com/iblnkn/rosetta_ws) installs
LeRobot 0.6.0 from a source checkout with the `dataset`, `async`, `training`,
`peft` and `smolvla` extras.
