# lerobot_teleoperator_rosetta

LeRobot [Teleoperator](https://huggingface.co/docs/lerobot/integrate_hardware#adding-a-teleoperator)
plugin for ROS 2, part of [Rosetta](https://github.com/iblnkn/rosetta), and
experimental. It reads a human operator's input, episode events, and operator
feedback from the topics declared in a contract's `teleop` section, so
LeRobot's teleoperation and human-in-the-loop tools can drive a ROS 2 robot.

The package name follows LeRobot's `lerobot_teleoperator_*`
[discovery convention](https://huggingface.co/docs/lerobot/integrate_hardware#the-4-core-conventions),
so installing it is enough for `--teleop.type=rosetta_teleop` to work. It is a
library with no executables. It pairs with
[lerobot_robot_rosetta](../lerobot_robot_rosetta),
usually reading both from the same contract.

## Usage

```python
from lerobot_teleoperator_rosetta import RosettaTeleop, RosettaTeleopConfig

teleop = RosettaTeleop(RosettaTeleopConfig(config_path="contract.yaml"))
teleop.connect()

action = teleop.get_action()
# {"position.j1": 0.1, "position.j2": 0.2}

events = teleop.get_teleop_events()
# {TeleopEvents.IS_INTERVENTION: True, TeleopEvents.SUCCESS: False, ...}

teleop.disconnect()
```

Or through the LeRobot CLI:

```bash
lerobot-teleoperate \
    --robot.type=rosetta --robot.config_path=contract.yaml \
    --teleop.type=rosetta_teleop --teleop.config_path=contract.yaml
```

If an input stream has not yet received data, `get_action()` omits its values
rather than filling zeros, because a fabricated zero would command motion.

## The teleop contract section

A contract's `teleop` section declares three roles. `input` entries each name
the action topic they drive (`target`) and are decoded like observations.
`events` maps event names onto message fields, typically
`sensor_msgs/msg/Joy` buttons. Event names are open: this plugin maps
`is_intervention`, `success`, `failure`, `end_success` and `end_failure`, and
logs any other name as ignored. `feedback` entries each name the observation topic they
forward back to the operator (`origin`).

On this LeRobot-native path, `end_success` and `end_failure` assert the
matching reward event together with `TERMINATE_EPISODE`, and `start_episode`
has no LeRobot counterpart (it is consumed by the HIL manager in
[`rosetta_ros_hil`](../../../robots/ros/rosetta_ros_hil) on the ROS 2-native path).

Schema and a worked example:
[contract reference](https://iblnkn.github.io/rosetta/reference/contract.html#teleop)
and the `so_101_hil.yaml` contract shipped with `rosetta_ros_hil`. For the ROS 2-native
loop, where the HIL manager muxes policy and teleop for you, see
[`rosetta_ros_hil`](../../../robots/ros/rosetta_ros_hil).

## Install

This package builds in a ROS 2 workspace with colcon, but it also needs
LeRobot, which has no rosdep key: `rosdep install` neither installs it nor
reports it missing. Install it with pip first. It isn't released to the ROS
buildfarm.

```bash
pip install 'lerobot[dataset]<0.6.2'
colcon build --packages-up-to lerobot_teleoperator_rosetta
```

## Documentation

Full Rosetta documentation: **https://iblnkn.github.io/rosetta/**

## License

Apache-2.0
