# rosetta_ros

The ROS 2 robot interface for [Rosetta](https://github.com/iblnkn/rosetta).
It connects Rosetta's core to a ROS 2 graph and registers as the `ros2` robot
interface, the value a contract declares as `robot_interface: ros2`.

It provides:

- decoders and encoders for common sensor and command message types
- `TopicBridge`, which samples subscriptions into frames and publishes actions
- the nodes `episode_recorder_node`, `policy_runner_node` and
  `episode_keyboard_node`, with their launch files and params files
- `rosetta_port`, which turns recorded bags into a dataset
- the node toolkit (`RosettaLifecycleNode`, `node_utils`, `child_goals` and
  others) that apps such as [`rosetta_ros_hil`](../rosetta_ros_hil) build on
- example contracts in `contracts/`: `so_101.yaml`, `turtlebot3.yaml`, and
  `stone.yaml`, the annotated tour of the contract format

Learning frameworks connect through framework adapters such as
[`lerobot_rosetta`](../../../policies/lerobot/lerobot_rosetta), which
`policy_runner_node` and `rosetta_port` load by name.

## Build

```bash
colcon build --packages-up-to rosetta_ros
source install/setup.bash
```

## Run

Record episodes against a contract. Each episode is one bag directory under
`datasets/bags` in the current directory.

```bash
ros2 launch rosetta_ros episode_recorder_launch.py contract_path:=robot.yaml
```

Drive the recorder from a second terminal with `r` to start, `s` to stop and
`d` to discard:

```bash
ros2 run rosetta_ros episode_keyboard_node
```

Every launch file prints its arguments with `--show-args`.

## Documentation

- [Record episodes](https://iblnkn.github.io/rosetta/how-to/record-episodes.html)
  and [Deploy a policy](https://iblnkn.github.io/rosetta/how-to/deploy-a-policy.html)
- [Nodes](https://iblnkn.github.io/rosetta/reference/nodes.html): parameters,
  topics, services and actions
- [rosetta_port](https://iblnkn.github.io/rosetta/reference/rosetta-port.html)
- Full Rosetta documentation: [iblnkn.github.io/rosetta](https://iblnkn.github.io/rosetta/)

## License

Apache-2.0
