# Packages

| Package | Contains |
|---|---|
| [`rosetta`](https://github.com/iblnkn/rosetta) | Contract loader, frame pipeline, the four nodes, `rosetta_port`, example contracts. |
| [`rosetta_interfaces`](https://github.com/iblnkn/rosetta_interfaces) | `RecordEpisode`, `RunPolicy`, `ManageEpisode` actions. `StartRecording`, `StartPolicy`, `StartHILEpisode` services. |
| [`lerobot_rosetta`](https://github.com/iblnkn/lerobot-rosetta) | The LeRobot adapter. Registers `lerobot` under `rosetta.dataset_writers` and `rosetta.policy_runners`. Executables `rosetta_policy_server`, `rosetta_classifier_server`. |
| [`lerobot_robot_rosetta`](https://github.com/iblnkn/lerobot-robot-rosetta) | A LeRobot `Robot` plugin. `--robot.type=rosetta` in LeRobot's own CLIs. |
| [`lerobot_teleoperator_rosetta`](https://github.com/iblnkn/lerobot-teleoperator-rosetta) | A LeRobot `Teleoperator` plugin. Experimental. |

All five are at version 0.2.0.

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
