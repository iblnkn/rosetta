# lerobot_robot_rosetta

LeRobot [Robot](https://huggingface.co/docs/lerobot/integrate_hardware) plugin for ROS2. Translates contract-defined topics into LeRobot's `get_observation()` / `send_action()` interface.

## Usage

```python
from lerobot_robot_rosetta import Rosetta, RosettaConfig

robot = Rosetta(RosettaConfig(config_path="contract.yaml"))
robot.connect()

# Get observations as dict
obs = robot.get_observation()
# {"shoulder.position": 0.1, "elbow.position": 0.2, "camera": array(...)}

# Send actions
robot.send_action({"shoulder.position": 0.5, "elbow.position": 0.3})

robot.disconnect()
```

Or with LeRobot CLI:

```bash
lerobot-record --robot.type=rosetta --robot.config_path=contract.yaml
lerobot-teleoperate --robot.type=rosetta --robot.config_path=contract.yaml
```

## Installation

```bash
colcon build --packages-select lerobot_robot_rosetta
source install/setup.bash
```

The package follows LeRobot's `lerobot_robot_*` [naming convention](https://huggingface.co/docs/lerobot/integrate_hardware#the-4-core-conventions) and is auto-discovered.

## Configuration

All configuration comes from the contract YAML. See [rosetta/README.md](../rosetta/README.md#contract-reference) for the full schema.

```yaml
robot_type: my_robot
fps: 30

observations:
  - key: observation.state
    topic: /joint_states
    type: sensor_msgs/msg/JointState
    selector: {names: [position.shoulder, position.elbow]}

actions:
  - key: action
    publish: {topic: /cmd, type: sensor_msgs/msg/JointState}
    selector: {names: [position.shoulder, position.elbow]}
    safety_behavior: hold  # what to publish if actions stop
```

## LeRobot Interface

Implements the [Robot](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/robot.py) base class:

| Property/Method | Description |
|-----------------|-------------|
| `observation_features` | Dict of feature names → types (callable before `connect()`) |
| `action_features` | Dict of action names → types (callable before `connect()`) |
| `is_connected` | True when lifecycle node is active |
| `connect()` | Configure and activate ROS2 subscriptions/publishers |
| `disconnect()` | Deactivate, send safety action, cleanup |
| `get_observation()` | Sample current observations from topic buffers |
| `send_action(action)` | Publish action to ROS2 topics |

## Behavior

**Lifecycle**: Uses ROS2 lifecycle nodes. `connect()` activates subscriptions and publishers. `disconnect()` triggers safety behavior then cleans up.

**Missing data**: If a topic has no data, zeros are returned and a warning is logged once.

**Safety watchdog**: If no action is sent within `2/fps` seconds:
- `none`: stop publishing
- `hold`: repeat last action
- `zeros`: publish zeros

**Timestamp alignment**: Observations from multiple topics are aligned using StreamBuffers with configurable strategies (`hold`, `asof`, `drop`).

## License

Apache-2.0
