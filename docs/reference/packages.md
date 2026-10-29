# Packages

`rosetta` is the core. Two kinds of package build on it. A robot interface
turns a robot's middleware into frames, and a framework adapter turns frames
into datasets and policy actions. Robot interfaces are named
`rosetta_<middleware>`, as in `rosetta_ros`. Framework adapters are named
`<framework>_rosetta`, as in `lerobot_rosetta`.

The [`rosetta`](https://github.com/iblnkn/rosetta) repository holds the core
and every package maintained with it, organized by side:

```text
rosetta/                     core
robots/<middleware>/         robot interfaces and the apps built on them
policies/<framework>/        framework adapters and framework plugins
```

Each side is named after its endpoint: the robot on one side, the policy on
the other. Core's modules use the same names, `rosetta.robots` and
`rosetta.policies`. [About the contract](../explanation/design.md#robot-interfaces-and-framework-adapters)
explains the split.

`rosetta_interfaces` holds ROS 2 message definitions. It is not a robot
interface in this sense.

## Packages in this repository

All seven are at version 0.2.0.

| Package | Path | Build type | Contains |
|---|---|---|---|
| [`rosetta`](https://github.com/iblnkn/rosetta/tree/main/rosetta) | `rosetta/` | `ament_python` | The contract loader, the frame pipeline, and the protocols and registries for robot interfaces and framework adapters. Imports no ROS. |
| [`rosetta_ros`](https://github.com/iblnkn/rosetta/tree/main/robots/ros/rosetta_ros) | `robots/ros/rosetta_ros/` | `ament_python` | The ROS 2 robot interface, registered as `ros2`: message codecs, `TopicBridge`, the node toolkit, the executables `episode_recorder_node`, `episode_keyboard_node`, `policy_runner_node` and `rosetta_port`, and the `contracts/`, `params/` and `launch/` files. |
| [`rosetta_interfaces`](https://github.com/iblnkn/rosetta/tree/main/robots/ros/rosetta_interfaces) | `robots/ros/rosetta_interfaces/` | `ament_cmake` | The actions `RecordEpisode`, `RunPolicy` and `ManageEpisode`, the services `StartRecording`, `StartPolicy` and `StartHILEpisode`, and the six `~/status` and `~/result` messages. |
| [`rosetta_ros_hil`](https://github.com/iblnkn/rosetta/tree/main/robots/ros/rosetta_ros_hil) | `robots/ros/rosetta_ros_hil/` | `ament_python` | The human-in-the-loop manager, `hil_manager_node`, built on the recorder, the policy runner and the node toolkit. |
| [`lerobot_rosetta`](https://github.com/iblnkn/rosetta/tree/main/policies/lerobot/lerobot_rosetta) | `policies/lerobot/lerobot_rosetta/` | `ament_python` | The LeRobot framework adapter, registered as `lerobot`. Also installs `rosetta_policy_server` and `rosetta_classifier_server` as plain console scripts, not `ros2 run` executables. |
| [`lerobot_robot_rosetta`](https://github.com/iblnkn/rosetta/tree/main/policies/lerobot/lerobot_robot_rosetta) | `policies/lerobot/lerobot_robot_rosetta/` | `ament_python` | A LeRobot `Robot` plugin, `--robot.type=rosetta` in LeRobot's own command-line tools. |
| [`lerobot_teleoperator_rosetta`](https://github.com/iblnkn/rosetta/tree/main/policies/lerobot/lerobot_teleoperator_rosetta) | `policies/lerobot/lerobot_teleoperator_rosetta/` | `ament_python` | A LeRobot `Teleoperator` plugin, `--teleop.type=rosetta_teleop`. Experimental. |

The two LeRobot plugin packages are named for LeRobot's plugin discovery,
which imports installed packages whose names start with `lerobot_robot_` or
`lerobot_teleoperator_`.

On ROS 2, install `rosetta_ros`. It brings `rosetta` and `rosetta_interfaces`
with it.

## Packages outside this repository

An adapter can live in its own repository and register the same entry points.
Nothing in Rosetta changes. [Add a framework](../how-to/add-a-framework.md)
and [Add a robot interface](../how-to/add-a-robot-interface.md) cover both.

## Entry-point groups

| Group | Value registered | Read by | Registered in this repository |
|---|---|---|---|
| `rosetta.robot_interfaces` | A class implementing `RobotInterface`, named after a contract's `robot_interface` value | Contract load | `ros2` by `rosetta_ros` |
| `rosetta.codecs` | A module that calls `register_decoder` and `register_encoder` | Contract load | None |
| `rosetta.operators` | A module that calls `register_operator` | Contract load | None |
| `rosetta.dataset_writers` | A class implementing `DatasetWriter` | `rosetta_port --framework` | `lerobot` by `lerobot_rosetta` |
| `rosetta.policy_runners` | A class implementing `PolicyRunner` | `policy_runner_node` parameter `framework` | `lerobot` by `lerobot_rosetta` |

Each class is instantiated with no arguments. A codec plugin that fails to
import fails every contract load in that environment. An operator plugin that
fails to import fails every contract with an `apply` step. A robot interface
that fails to register its codecs fails only the contracts that name it.

## Dependencies

`rosetta` needs Python 3.12 or newer, `numpy` and `pyyaml`. It imports no ROS
and no learning framework. `rosetta_ros` and `rosetta_ros_hil` depend on ROS 2
packages that `rosdep` resolves, including `rosbag2_storage_mcap`.

All interfaces share one codec registry keyed by type string. Two robot
interfaces that register the same type string conflict, and the one later in
name order fails to load.

`lerobot_rosetta` needs `lerobot<0.6.2`, `torch`, `grpcio` and
`lerobot_robot_rosetta`. LeRobot 0.6.2 removes `async_inference`, which the
LeRobot policy runner uses. `lerobot_robot_rosetta` and
`lerobot_teleoperator_rosetta` need `lerobot<0.6.2`, `rosetta` and
`rosetta_ros`. `rosdep` has no key for LeRobot, so install it with `pip`. For
the same reason the LeRobot packages are not released to the ROS buildfarm.
The repository's pixi environment installs LeRobot 0.6.1 from PyPI with the
`dataset`, `async`, `training`, `peft` and `smolvla` extras.
