# rosetta

Rosetta's core: the contract loader, the frame pipeline, and the protocols
and registries that robot interfaces and framework adapters implement. A
robot interface such as [`rosetta_ros`](../robots/ros/rosetta_ros) turns a
robot's middleware into frames. A framework adapter such as
[`lerobot_rosetta`](../policies/lerobot/lerobot_rosetta) turns frames into
datasets and policy actions. Both register through entry points, so core
imports neither.

Core imports no ROS. It needs Python 3.12 or newer, `numpy` and `pyyaml`,
and installs and tests as a plain Python package. It has no nodes. Loading a
contract needs the robot interface it names installed: `robot_interface: ros2`
comes from `rosetta_ros`, which brings this package with it.

## Build

In a ROS 2 workspace:

```bash
colcon build --packages-up-to rosetta
source install/setup.bash
```

Without ROS, from the repository root:

```bash
pip install ./rosetta
```

## Documentation

- [Python API](https://iblnkn.github.io/rosetta/reference/python-api.html):
  the public modules
- [Contract](https://iblnkn.github.io/rosetta/reference/contract.html): the
  contract schema
- [Add a robot interface](https://iblnkn.github.io/rosetta/how-to/add-a-robot-interface.html)
  and [Add a framework](https://iblnkn.github.io/rosetta/how-to/add-a-framework.html)
- Full Rosetta documentation: [iblnkn.github.io/rosetta](https://iblnkn.github.io/rosetta/)

## License

Apache-2.0
