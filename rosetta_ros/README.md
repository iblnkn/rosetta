# rosetta_ros

The ROS 2 robot interface for [Rosetta](https://github.com/iblnkn/rosetta).
It connects Rosetta's core to a ROS 2 graph and registers itself as the `ros2`
robot interface, the value contracts declare as `robot_interface: ros2`.

It provides:

- message decoders and encoders for the common sensor and command types
- `TopicBridge`, which samples subscriptions into frames and publishes actions
- the `episode_recorder_node`, `policy_runner_node` and `episode_keyboard_node` nodes, with their launch files and params
- `rosetta_port`, which converts recorded bags into datasets
- the node toolkit (`RosettaLifecycleNode`, `node_utils`, `child_goals`, ...) that apps such as [`rosetta_ros_hil`](../rosetta_ros_hil) build on
- the example contracts in `contracts/`

## Build

```bash
colcon build --packages-up-to rosetta_ros
source install/setup.bash
```

## Run

Record episodes against a contract:

```bash
ros2 launch rosetta_ros episode_recorder_launch.py contract_path:=robot.yaml
```

The launch file prints every argument with `--show-args`.

## Documentation

Full Rosetta documentation: **https://iblnkn.github.io/rosetta/**

## License

Apache-2.0
