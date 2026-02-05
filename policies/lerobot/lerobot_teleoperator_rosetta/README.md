# lerobot_teleoperator_rosetta

LeRobot [Teleoperator](https://huggingface.co/docs/lerobot/integrate_hardware#adding-a-teleoperator) plugin for ROS2. Captures human input from contract-defined topics for teleoperation and human-in-the-loop training.

## Usage

```python
from lerobot_teleoperator_rosetta import RosettaTeleop, RosettaTeleopConfig

teleop = RosettaTeleop(RosettaTeleopConfig(config_path="contract.yaml"))
teleop.connect()

# Get action from human operator
action = teleop.get_action()
# {"leader.position.j1": 0.1, "leader.position.j2": 0.2}

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

Configure via the `teleop` section of your contract:

```yaml
teleop:
  inputs:
    - key: teleop_input
      topic: /leader_arm/joint_states
      type: sensor_msgs/msg/JointState
      selector:
        names: [position.j1, position.j2, position.j3]

  events:
    topic: /joy
    type: sensor_msgs/msg/Joy
    mappings:
      is_intervention: buttons.5   # Human taking over
      success: buttons.0           # Mark success
      terminate_episode: buttons.6 # End episode
      rerecord_episode: buttons.7  # Discard and restart
      failure: buttons.1           # Mark failure

  feedback: []  # Optional publishers for operator feedback
```

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

## License

Apache-2.0
