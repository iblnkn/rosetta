# lerobot_robot_rosetta

A LeRobot [Robot](https://huggingface.co/docs/lerobot/integrate_hardware)
plugin for ROS 2, part of [Rosetta](https://github.com/iblnkn/rosetta). It
makes a ROS 2 robot usable from LeRobot's tools by treating the topics a
contract declares as the hardware: observations come from subscriptions,
actions go out as publications, and the drivers run elsewhere in the ROS 2
graph.

The package name follows LeRobot's `lerobot_robot_*`
[discovery convention](https://huggingface.co/docs/lerobot/integrate_hardware#the-4-core-conventions),
so installing it is enough for `--robot.type=rosetta` to work. It is a library
with no executables. The opposite direction, Rosetta loading LeRobot as a
policy framework, is [`lerobot_rosetta`](../lerobot_rosetta).

## Usage

```python
from lerobot_robot_rosetta import Rosetta, RosettaConfig

robot = Rosetta(RosettaConfig(config_path="so_101.yaml"))
robot.connect()

obs = robot.get_observation()
# keys: "position.shoulder_pan", ..., "right", "top", "wrist.right"

# so_101.yaml names its actions like its state, so this holds position.
robot.send_action({name: obs[name] for name in robot.action_features})
robot.disconnect()
```

`so_101.yaml` ships in `rosetta_ros`'s `contracts/`. `send_action` needs a
value for every name in `action_features`.

From the LeRobot CLI, paired with the teleoperator plugin from
[`lerobot_teleoperator_rosetta`](../lerobot_teleoperator_rosetta):

```bash
lerobot-teleoperate \
    --robot.type=rosetta --robot.config_path=contract.yaml \
    --teleop.type=rosetta_teleop --teleop.config_path=contract.yaml
```

| Config field | Default | Meaning |
|---|---|---|
| `config_path` | `""` | Contract file. |
| `fps` | contract `fps` | Hz. Sets the watchdog period and the pacing under sim time. |
| `is_classifier` | `false` | Use the contract's `rewards` section as the actions. |
| `use_sim_time` | `false` | Run the plugin's node on `/clock` time. |
| `id` | contract `robot_type` | LeRobot's device id. The node is named `rosetta_<id>`. |

Feature names come from the contract. Numeric values are named by their
`select` paths (`position.shoulder_pan`), and images by the part of the key
after `observation.images.` (`right` for `observation.images.right`). A key
fed by several topics prefixes each source's names with a part of its topic
(`arm.position.shoulder_pan`, `gripper.data`). `observation_features` and
`action_features` are readable before `connect()`.

## How it maps to ROS 2

The plugin hosts a ROS 2 lifecycle node, so ROS 2 must be installed even when
only LeRobot's CLI calls it. `connect()` configures and activates the node,
then waits up to 5 s for every observation stream to deliver a first message,
and logs a warning if one has not. `disconnect()` publishes each action
channel's `safety` command (`zeros`, `hold`, or nothing for `none`), then
deactivates and cleans up the node. While connected, a watchdog publishes the
same command whenever no action arrives for two frame periods.

LeRobot's live path merges every numeric observation into one
`observation.state` and every action into one `action`, and names them from
`select`. `connect()` therefore refuses a contract with more than one numeric
observation key, more than one action key, or a numeric stream without
`select`, with an error naming them. Such a contract still records and ports
to a dataset.

## Install

LeRobot has no rosdep key, so `rosdep install` neither installs it nor
reports it missing. Install it with pip first. The package is not released to
the ROS buildfarm.

```bash
pip install 'lerobot[dataset]<0.6.2'
colcon build --packages-up-to lerobot_robot_rosetta
```

## Documentation

- [Contract](https://iblnkn.github.io/rosetta/reference/contract.html): the
  contract schema
- Full Rosetta documentation: [iblnkn.github.io/rosetta](https://iblnkn.github.io/rosetta/)

## License

Apache-2.0
