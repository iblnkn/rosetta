# lerobot_teleoperator_rosetta

A LeRobot [Teleoperator](https://huggingface.co/docs/lerobot/integrate_hardware#adding-a-teleoperator)
plugin for ROS 2, part of [Rosetta](https://github.com/iblnkn/rosetta). It is
experimental. It reads a human operator's input and episode events from the
topics in a contract's `teleop` section, and publishes operator feedback, so
LeRobot's teleoperation and human-in-the-loop tools can drive a ROS 2 robot.

The package name follows LeRobot's `lerobot_teleoperator_*`
[discovery convention](https://huggingface.co/docs/lerobot/integrate_hardware#the-4-core-conventions),
so installing it is enough for `--teleop.type=rosetta_teleop` to work. It is a
library with no executables. It pairs with
[`lerobot_robot_rosetta`](../lerobot_robot_rosetta), and both usually read the
same contract.

## Usage

```python
from lerobot_teleoperator_rosetta import RosettaTeleop, RosettaTeleopConfig

teleop = RosettaTeleop(RosettaTeleopConfig(config_path="contract.yaml"))
teleop.connect()

action = teleop.get_action()
# keys: "position.shoulder_pan", ... for each input stream that has data

events = teleop.get_teleop_events()
# {TeleopEvents.SUCCESS: False, TeleopEvents.FAILURE: False, ...}

teleop.disconnect()
```

From the LeRobot CLI:

```bash
lerobot-teleoperate \
    --robot.type=rosetta --robot.config_path=contract.yaml \
    --teleop.type=rosetta_teleop --teleop.config_path=contract.yaml
```

The config's only field beyond LeRobot's is `config_path`, and the contract
must have a `teleop` section. The id defaults to `<robot_type>_teleop`, and
the plugin's node is named `rosetta_teleop_<id>`.

`get_action()` leaves out the values of an input stream that has received no
message yet, because a fabricated zero would command motion. The robot
plugin's `send_action` needs every value, so `lerobot-teleoperate` stops with
a `KeyError` naming the first missing one. Start the teleop device before the
LeRobot tool.

## Teleop events

The contract's `teleop.events` maps event names onto message fields, usually
`sensor_msgs/msg/Joy` buttons. Event names are open. This plugin maps five of
them onto LeRobot's `TeleopEvents`:

| Contract event | LeRobot events |
|---|---|
| `is_intervention` | `IS_INTERVENTION` |
| `success` | `SUCCESS` |
| `failure` | `FAILURE` |
| `end_success` | `SUCCESS`, `TERMINATE_EPISODE` |
| `end_failure` | `FAILURE`, `TERMINATE_EPISODE` |

Other names, such as `start_episode`, are logged at configure and ignored.
`start_episode` is for the ROS 2-native human-in-the-loop manager in
[`rosetta_ros_hil`](../../../robots/ros/rosetta_ros_hil), which also muxes
policy and teleop commands itself. `get_teleop_events()` returns every
`TeleopEvents` member, each holding the value from the latest events message
and `False` before the first one. `RERECORD_EPISODE` has no contract event
and stays `False`.

## Install

LeRobot has no rosdep key, so `rosdep install` neither installs it nor
reports it missing. Install it with pip first. The package is not released to
the ROS buildfarm.

```bash
pip install 'lerobot[dataset]<0.6.2'
colcon build --packages-up-to lerobot_teleoperator_rosetta
```

## Documentation

- [Contract reference, teleop section](https://iblnkn.github.io/rosetta/reference/contract.html#teleop),
  and `contracts/so_101_hil.yaml` in `rosetta_ros_hil` for a complete example
- Full Rosetta documentation: [iblnkn.github.io/rosetta](https://iblnkn.github.io/rosetta/)

## License

Apache-2.0
