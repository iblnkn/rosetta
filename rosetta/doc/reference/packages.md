# Packages

The [`rosetta`](https://github.com/iblnkn/rosetta) repository holds the first five
packages. Each LeRobot package has its own repository.

`rosetta` is the core. Two kinds of package plug into it: robot interfaces
(`rosetta_ros`; named `rosetta_<interface>`) turn a robot's middleware into
frames, and framework adapters (`lerobot_rosetta`; named `<framework>_rosetta`)
turn frames into datasets and policy actions.

| Package | Contains |
|---|---|
| [`rosetta`](https://github.com/iblnkn/rosetta/tree/main/rosetta) | Contract loader, frame pipeline, and the protocols and registries for robot interfaces and framework adapters. Imports no ROS. |
| [`rosetta_ros`](https://github.com/iblnkn/rosetta/tree/main/rosetta_ros) | The ROS 2 robot interface: message codecs, `TopicBridge`, the recorder, policy runner and keyboard nodes, `rosetta_port`, the node toolkit for building apps, example contracts and launch files. |
| [`rosetta_interfaces`](https://github.com/iblnkn/rosetta/tree/main/rosetta_interfaces) | `RecordEpisode`, `RunPolicy` actions. `StartRecording`, `StartPolicy` services. |
| [`rosetta_ros_hil`](https://github.com/iblnkn/rosetta/tree/main/rosetta_ros_hil) | The human-in-the-loop manager, an app built on the recorder, the policy runner and the node toolkit. |
| [`rosetta_hil_interfaces`](https://github.com/iblnkn/rosetta/tree/main/rosetta_hil_interfaces) | `ManageEpisode` action, `StartHILEpisode` service. |
| [`lerobot_rosetta`](https://github.com/iblnkn/lerobot-rosetta) | The LeRobot adapter. Registers `lerobot` under `rosetta.dataset_writers` and `rosetta.policy_runners`. Executables `rosetta_policy_server`, `rosetta_classifier_server`. |
| [`lerobot_robot_rosetta`](https://github.com/iblnkn/lerobot-robot-rosetta) | A LeRobot `Robot` plugin. `--robot.type=rosetta` in LeRobot's own CLIs. |
| [`lerobot_teleoperator_rosetta`](https://github.com/iblnkn/lerobot-teleoperator-rosetta) | A LeRobot `Teleoperator` plugin. Experimental. |

All eight are at version 0.2.0.

## Entry-point groups

| Group | Registers | Read by |
|---|---|---|
| `rosetta.robot_interfaces` | A class implementing `RobotInterface`, named after a contract's `robot_interface` value. `rosetta_ros` registers `ros2`. | Contract load. |
| `rosetta.codecs` | A module that calls `register_decoder` and `register_encoder`. | Contract load. |
| `rosetta.operators` | A module that calls `register_operator`. | Contract load. |
| `rosetta.dataset_writers` | A class implementing `DatasetWriter`. | `rosetta_port --framework`. |
| `rosetta.policy_runners` | A class implementing `PolicyRunner`. | `policy_runner_node` parameter `framework`. |

## Dependencies

`rosetta` needs only `numpy` and `pyyaml`. It doesn't import ROS or a
learning framework. `rosetta_ros` depends on ROS 2 packages resolvable by
`rosdep`, including `rosbag2_storage_mcap`.

Codecs share one registry keyed by type string, whichever interface registers
them. Two robot interfaces that register the same type string conflict at
load.

`lerobot_rosetta` needs `lerobot`, `torch`, `grpcio` and
`lerobot_robot_rosetta`. `lerobot_robot_rosetta` needs `lerobot`. `rosdep`
has a key for `grpcio` but none for `lerobot` or `torch`, so install those
with `pip`. [`rosetta_ws`](https://github.com/iblnkn/rosetta_ws) installs
LeRobot 0.6.0 from a source checkout with the `dataset`, `async`, `training`,
`peft` and `smolvla` extras.
