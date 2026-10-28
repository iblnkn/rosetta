# rosetta_interfaces

The ROS 2 action, service and message definitions for
[Rosetta](https://github.com/iblnkn/rosetta). They are the control plane:
starting, stopping and reporting on recordings, policy runs and
human-in-the-loop episodes. The data the robot and the policy exchange travels
on ordinary ROS 2 topics that a contract declares. This package is unrelated
to Rosetta's robot interfaces (the `robot_interface:` a contract names), which
are Python packages such as `rosetta_ros`.

| Interface | Kind | Served by |
|---|---|---|
| `RecordEpisode` | action | `episode_recorder` in [`rosetta_ros`](../rosetta_ros) |
| `StartRecording` | service | `episode_recorder` |
| `RecordEpisodeStatus`, `RecordEpisodeResult` | message | `episode_recorder` `~/status` and `~/result` |
| `RunPolicy` | action | `policy_runner` in [`rosetta_ros`](../rosetta_ros) |
| `StartPolicy` | service | `policy_runner` |
| `RunPolicyStatus`, `RunPolicyResult` | message | `policy_runner` `~/status` and `~/result` |
| `ManageEpisode` | action | `hil_manager` in [`rosetta_ros_hil`](../rosetta_ros_hil) |
| `StartHILEpisode` | service | `hil_manager` |
| `ManageEpisodeStatus`, `ManageEpisodeResult` | message | `hil_manager` `~/status` and `~/result` |

Each start service starts the same work as its action, for clients that can
call services but not actions, such as Foxglove. It returns once the work is
accepted. The status and result topics carry the action's feedback and result
fields however the work started.

## Build

```bash
colcon build --packages-select rosetta_interfaces
source install/setup.bash
ros2 interface package rosetta_interfaces
```

The last command lists the twelve interfaces. The `.action`, `.srv` and
`.msg` files are commented.

## Documentation

Field-by-field reference:
[Interfaces](https://iblnkn.github.io/rosetta/reference/interfaces.html).
The nodes that serve them:
[Nodes](https://iblnkn.github.io/rosetta/reference/nodes.html).
Full Rosetta documentation: [iblnkn.github.io/rosetta](https://iblnkn.github.io/rosetta/)

## License

Apache-2.0
