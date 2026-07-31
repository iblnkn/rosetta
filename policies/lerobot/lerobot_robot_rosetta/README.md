# lerobot_robot_rosetta

LeRobot [Robot](https://huggingface.co/docs/lerobot/integrate_hardware) plugin for ROS2. Translates contract-defined topics into LeRobot's `get_observation()` / `send_action()` interface.

## Usage

```python
from lerobot_robot_rosetta import Rosetta, RosettaConfig

robot = Rosetta(RosettaConfig(config_path="contract.yaml"))
robot.connect()

# Get observations as dict, keyed by the contract's own selector names
obs = robot.get_observation()
# {"position.shoulder": 0.1, "position.elbow": 0.2, "cam": array(...)}

# Send actions
robot.send_action({"position.shoulder": 0.5, "position.elbow": 0.3})

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

All configuration comes from the contract YAML:

```yaml
robot_type: my_robot
robot_interface: ros2
fps: 30

observations:
  observation.state:
    channel: {topic: /joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: header}
    select: [position.shoulder, position.elbow]

  observation.images.cam:
    channel: {topic: /camera/image_raw/compressed,
              type: sensor_msgs/msg/CompressedImage}
    align: {strategy: hold, timeline: header}
    apply: [resize: [480, 640]]

actions:
  action:
    channel:
      topic: /cmd
      type: sensor_msgs/msg/JointState
      safety: hold          # what to publish if actions stop
    align: {strategy: hold, timeline: header}
    select: [position.shoulder, position.elbow]
```

Full schema — every section, operator, and alignment strategy:
[contract reference](https://iblnkn.github.io/rosetta/reference/contract.html).

## LeRobot Interface

Implements the [Robot](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/robot.py) base class:

| Property/Method | Description |
|-----------------|-------------|
| `observation_features` | Dict of feature names → types (callable before `connect()`). Vector features are named by their contract `select` path (`position.shoulder`), images by the part of the key after `observation.images.` (`cam`). A key fed by several topics prefixes each source with a distinguishing topic segment |
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

## Inference Servers

The gRPC inference servers, dataset writer, and policy runner moved to [`lerobot_rosetta`](https://github.com/iblnkn/lerobot-rosetta) — rosetta's LeRobot backend adapter. This package keeps only the LeRobot-discovered Robot plugin.

## Documentation

Full Rosetta documentation: **https://iblnkn.github.io/rosetta/**

## License

Apache-2.0
