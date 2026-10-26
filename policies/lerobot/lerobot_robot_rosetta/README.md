# lerobot_robot_rosetta

LeRobot [Robot](https://huggingface.co/docs/lerobot/integrate_hardware) plugin
for ROS 2, part of [Rosetta](https://github.com/iblnkn/rosetta). It makes a
ROS 2 robot usable through LeRobot's standard tools by treating
contract-defined topics as the hardware interface: observations come from
subscriptions, actions go out as publications, and the drivers live elsewhere
in the ROS 2 graph.

The package name follows LeRobot's `lerobot_robot_*`
[discovery convention](https://huggingface.co/docs/lerobot/integrate_hardware#the-4-core-conventions),
so installing it is enough for `--robot.type=rosetta` to work. It is a library
with no executables. The opposite direction, Rosetta loading LeRobot as a
policy framework, lives in
[lerobot_rosetta](https://github.com/iblnkn/lerobot-rosetta).

## Usage

All configuration is the contract YAML; the plugin adds no options of its own.

```python
from lerobot_robot_rosetta import Rosetta, RosettaConfig

robot = Rosetta(RosettaConfig(config_path="contract.yaml"))
robot.connect()

obs = robot.get_observation()
# {"position.shoulder": 0.1, "position.elbow": 0.2, "cam": array(...)}

robot.send_action({"position.shoulder": 0.5, "position.elbow": 0.3})
robot.disconnect()
```

Or through the LeRobot CLI:

```bash
lerobot-record --robot.type=rosetta --robot.config_path=contract.yaml
lerobot-teleoperate --robot.type=rosetta --robot.config_path=contract.yaml
```

Feature names come from the contract. Vector features are named by their
`select` paths (`position.shoulder`), images by the part of the key after
`observation.images.` (`cam`), and a key fed by several topics gets a
distinguishing topic-derived prefix per source. Both `observation_features`
and `action_features` are readable before `connect()`.

## How it maps to ROS 2

The plugin hosts a ROS 2 lifecycle node, so ROS 2 must be installed even when
you only invoke it through LeRobot's CLI. `connect()` configures and activates
the node, then waits for every observation stream to deliver a first value.
`disconnect()` publishes each action channel's declared safety behavior, then
deactivates and cleans up. While connected, a watchdog applies the same
per-channel safety behavior (`zeros`, `hold`, or `none`) whenever no action
arrives for two frame periods.

One constraint applies to contracts served through LeRobot: the live path
expects a single numeric observation key and a single action key, because
LeRobot collapses numeric features into one `observation.state` and one
`action`. A richer contract still records and trains, and is rejected at
`connect()` with an error naming the offending keys.

## Documentation

Contract schema:
[contract reference](https://iblnkn.github.io/rosetta/reference/contract.html).
Full Rosetta documentation: **https://iblnkn.github.io/rosetta/**

## License

Apache-2.0
