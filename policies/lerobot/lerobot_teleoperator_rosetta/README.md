# lerobot_teleoperator_rosetta

LeRobot [Teleoperator](https://huggingface.co/docs/lerobot/integrate_hardware#adding-a-teleoperator) plugin for ROS2. Captures human input from contract-defined topics for teleoperation and human-in-the-loop training.

## Usage

```python
from lerobot_teleoperator_rosetta import RosettaTeleop, RosettaTeleopConfig

teleop = RosettaTeleop(RosettaTeleopConfig(config_path="contract.yaml"))
teleop.connect()

# Get action from human operator, keyed by the entry's own selector names
action = teleop.get_action()
# {"position.j1": 0.1, "position.j2": 0.2}
# (several input entries driving the same action get a topic-derived prefix
#  to keep their names distinct)

# Check events (buttons, intervention signals)
events = teleop.get_teleop_events()
# {TeleopEvents.IS_INTERVENTION: True, TeleopEvents.SUCCESS: False, ...}

# Send feedback to operator (optional)
teleop.send_feedback({"status": 1.0})

teleop.disconnect()
```

Or with LeRobot CLI:

```bash
lerobot-teleop \
    --robot.type=rosetta --robot.config_path=contract.yaml \
    --teleop.type=rosetta_teleop --teleop.config_path=contract.yaml
```

## Installation

```bash
colcon build --packages-select lerobot_teleoperator_rosetta
source install/setup.bash
```

The package follows LeRobot's `lerobot_teleoperator_*` [naming convention](https://huggingface.co/docs/lerobot/integrate_hardware#the-4-core-conventions) and is auto-discovered.

## Configuration

Configure via the `teleop` section of your contract. `input` and `feedback` are
lists of independently-targeted sources; each entry names an existing action
(`target`) or observation (`origin`) topic, validated at contract load:

```yaml
teleop:
  input:
    - target: /cmd                 # names an existing action channel's topic
      channel: {topic: /leader_arm/joint_states, type: sensor_msgs/msg/JointState}
      align: {strategy: hold, timeline: header}
      select: [position.j1, position.j2, position.j3]

  events:                          # edge-triggered; no align — events are not resampled
    channel: {topic: /joy, type: sensor_msgs/msg/Joy}
    select:                        # event_name -> button/axis path
      is_intervention: buttons.5   # Human taking over
      success: buttons.0           # Mark success
      end_success: buttons.6       # End episode, success
      end_failure: buttons.7       # End episode, failure
      failure: buttons.1           # Mark failure

  feedback: []                     # Optional publishers for operator feedback
```

The event vocabulary is closed — `is_intervention`, `start_episode`, `success`,
`failure`, `end_success`, `end_failure`. An unknown event name is a load error.
On this LeRobot-native path the first three map one-to-one onto LeRobot's
`TeleopEvents`, `end_success`/`end_failure` assert the reward event plus
`TERMINATE_EPISODE` together, and `start_episode` has no LeRobot counterpart
(ignored here, handled by `hil_manager_node`).

Full schema: [contract reference](https://iblnkn.github.io/rosetta/reference/contract.html#teleop).

## LeRobot Interface

Implements the [Teleoperator](https://github.com/huggingface/lerobot/blob/main/src/lerobot/teleoperators/teleoperator.py) base class:

| Property/Method | Description |
|-----------------|-------------|
| `action_features` | Dict of action names → types |
| `feedback_features` | Dict of feedback names → types |
| `is_connected` | True when lifecycle node is active |
| `connect()` | Configure and activate ROS2 subscriptions/publishers |
| `disconnect()` | Deactivate and cleanup |
| `get_action()` | Sample current action from input buffers |
| `get_teleop_events()` | Get current event states (intervention, success, etc.) |
| `send_feedback(feedback)` | Publish feedback to ROS2 topics |

## HIL-SERL Integration

```python
from lerobot_robot_rosetta import Rosetta, RosettaConfig
from lerobot_teleoperator_rosetta import RosettaTeleop, RosettaTeleopConfig
from lerobot.teleoperators.utils import TeleopEvents

robot = Rosetta(RosettaConfig(config_path="contract.yaml"))
teleop = RosettaTeleop(RosettaTeleopConfig(config_path="contract.yaml"))

robot.connect()
teleop.connect()

while True:
    events = teleop.get_teleop_events()

    if events[TeleopEvents.IS_INTERVENTION]:
        action = teleop.get_action()  # Human controls
    else:
        obs = robot.get_observation()
        action = policy(obs)          # Policy controls

    robot.send_action(action)

    if events[TeleopEvents.TERMINATE_EPISODE]:
        break
```

See [Set up teleop and HIL](https://iblnkn.github.io/rosetta/how-to/record-train-deploy.html) for the ROS 2-native path, where `hil_manager_node` runs this loop for you.

## Documentation

Full Rosetta documentation: **https://iblnkn.github.io/rosetta/**

## License

Apache-2.0
