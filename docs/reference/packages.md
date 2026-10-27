# Packages

`rosetta` is the core. Two kinds of package plug into it: robot interfaces
(`rosetta_ros`; named `rosetta_<interface>`) turn a robot's middleware into
frames, and framework adapters (`lerobot_rosetta`; named `<framework>_rosetta`)
turn frames into datasets and policy actions.

The [`rosetta`](https://github.com/iblnkn/rosetta) repository holds the core and
every package maintained with it, organized by side:

```
rosetta/                     core
robots/<middleware>/         robot interfaces and the apps built on them
policies/<framework>/        framework adapters
```

Rosetta sits between a robot and a policy, and each side is named after its
endpoint. The packages on a side adapt to what that endpoint runs on: robot
interfaces adapt a middleware such as ROS 2, and framework adapters adapt a
learning framework such as LeRobot. Core's modules use the same names,
`rosetta.robots` and `rosetta.policies`. [Design](../explanation/design.md#two-kinds-of-plugin)
explains the split.

| Package | Contains |
|---|---|
| [`rosetta`](https://github.com/iblnkn/rosetta/tree/main/rosetta) | Contract loader, frame pipeline, and the protocols and registries for robot interfaces and framework adapters. Imports no ROS. |
| [`rosetta_ros`](https://github.com/iblnkn/rosetta/tree/main/robots/ros/rosetta_ros) | The ROS 2 robot interface: message codecs, `TopicBridge`, the recorder, policy runner and keyboard nodes, `rosetta_port`, the node toolkit for building apps, example contracts and launch files. |
| [`rosetta_interfaces`](https://github.com/iblnkn/rosetta/tree/main/robots/ros/rosetta_interfaces) | Every Rosetta ROS 2 interface: `RecordEpisode`, `RunPolicy`, `ManageEpisode` actions; `StartRecording`, `StartPolicy`, `StartHILEpisode` services; their `~/status` and `~/result` messages. |
| [`rosetta_ros_hil`](https://github.com/iblnkn/rosetta/tree/main/robots/ros/rosetta_ros_hil) | The human-in-the-loop manager, an app built on the recorder, the policy runner and the node toolkit. |
| [`lerobot_rosetta`](https://github.com/iblnkn/rosetta/tree/main/policies/lerobot/lerobot_rosetta) | The LeRobot adapter. Registers `lerobot` under `rosetta.dataset_writers` and `rosetta.policy_runners`. Executables `rosetta_policy_server`, `rosetta_classifier_server`. |
| [`lerobot_robot_rosetta`](https://github.com/iblnkn/rosetta/tree/main/policies/lerobot/lerobot_robot_rosetta) | A LeRobot `Robot` plugin. `--robot.type=rosetta` in LeRobot's own CLIs. |
| [`lerobot_teleoperator_rosetta`](https://github.com/iblnkn/rosetta/tree/main/policies/lerobot/lerobot_teleoperator_rosetta) | A LeRobot `Teleoperator` plugin. Experimental. |

All seven are at version 0.2.0. On ROS 2, install `rosetta_ros`; it brings
`rosetta` and `rosetta_interfaces` with it.

## Packages outside this repository

An adapter can live in its own repository and register the same entry points;
nothing in Rosetta changes. [Add a framework](../how-to/add-a-framework.md)
and [Add a robot interface](../how-to/add-a-robot-interface.md) cover both.

| Package | Contains |
|---|---|
| `starvla_rosetta` | The starVLA adapter: a dataset writer and a websocket policy runner. Not yet public. |
| `vla_foundry_rosetta` | The vla_foundry adapter: a dataset writer and an in-process policy runner. Not yet public. |
| `rerun_rosetta` | A Rerun dataset backend: trains directly from `.rrd` recordings on a Rerun catalog server. Not yet public. |

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
with `pip`; for the same reason the LeRobot packages aren't released to the ROS buildfarm. [`rosetta_ws`](https://github.com/iblnkn/rosetta_ws) installs
LeRobot 0.6.1 from a source checkout with the `dataset`, `async`, `training`,
`peft` and `smolvla` extras.
