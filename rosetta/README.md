# rosetta

Rosetta's core: the contract loader, the frame pipeline, and the protocols and
registries the two kinds of plugin package implement. Robot interfaces such as
[`rosetta_ros`](../robots/ros/rosetta_ros) turn a robot's middleware into frames.
Framework adapters such as `lerobot_rosetta` turn frames into datasets and
policy actions. Core imports no ROS, so it installs and tests without it.

The public modules are listed on the
[Python API reference](https://iblnkn.github.io/rosetta/reference/python-api.html)
page.

## Build

```bash
colcon build --packages-up-to rosetta
source install/setup.bash
```

## Documentation

Full Rosetta documentation: **https://iblnkn.github.io/rosetta/**

## License

Apache-2.0
