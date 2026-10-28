<p align="center">
  <img alt="Rosetta" src="media/rosetta_logo.png" width="100%">
</p>

Rosetta connects a ROS 2 robot to a robot-learning framework such as
[LeRobot](https://github.com/huggingface/lerobot).

A robot publishes topics at whatever rate each sensor runs. A policy wants
one frame per tick, with a fixed set of keys. You write the mapping between
the two in a YAML file, the contract. Rosetta applies that one file when it
builds a dataset from bags and again when it runs the policy on the robot.

**Documentation: [iblnkn.github.io/rosetta](https://iblnkn.github.io/rosetta/)**

```yaml
robot_type: so_arm101
robot_interface: ros2
fps: 50
observations:
  observation.images.wrist:                          # key: what the model calls it
    channel: {topic: /wrist_camera/image_raw,        # channel: topic, type, QoS
              type: sensor_msgs/msg/Image}
    align: {strategy: hold, timeline: header}        # align: when
    apply: [resize: [480, 480]]                      # apply: how values change
  observation.state:
    channel: {topic: /joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: header}
    select: [position.shoulder_pan_joint, position.elbow_flex_joint]   # select: which fields
    apply: [rad2deg]
actions:
  action:
    channel: {topic: /forward_position_controller/commands,
              type: std_msgs/msg/Float64MultiArray, safety: hold}
    align: {strategy: hold, timeline: receive}
    select: [shoulder_pan.pos, elbow_flex.pos]
    apply: [clamp: {min: -3.14159, max: 3.14159}, rad2deg]
```

Five steps. When you change the contract you redo steps 2 to 5. The bags
from step 1 stay.

```bash
ros2 launch rosetta_ros episode_recorder_launch.py contract_path:=robot.yaml                 # 1 record
$EDITOR robot.yaml                                                                           # 2 define
ros2 run rosetta_ros rosetta_port --raw-dir datasets/bags --contract robot.yaml --repo-id my # 3 prepare
lerobot-train --dataset.repo_id=my --policy.type=act --policy.push_to_hub=false \
    --output_dir=outputs/train/my                                                            # 4 train
ros2 launch rosetta_ros policy_runner_launch.py contract_path:=robot.yaml \
    pretrained_name_or_path:=outputs/train/my/checkpoints/last/pretrained_model              # 5 deploy
```

The recorder writes one bag per episode under `datasets/bags`. Start and stop
episodes from a second terminal with `ros2 run rosetta_ros episode_keyboard_node`.

New here? [Install](https://iblnkn.github.io/rosetta/installation.html), then
follow
[From bags to a moving arm](https://iblnkn.github.io/rosetta/tutorials/first-policy.html).

## Packages

This repository holds Rosetta's core and every package maintained with it,
organized by side. Rosetta sits between a robot and a policy, and each side
is named after its endpoint: `robots/` holds robot interfaces, grouped by the
middleware they adapt (ROS 2 today), and `policies/` holds framework adapters,
grouped by the learning framework they adapt. Core's modules use the same
names, `rosetta.robots` and `rosetta.policies`.
[About the contract](https://iblnkn.github.io/rosetta/explanation/design.html)
explains the split.

| Package | Purpose |
|---------|---------|
| [`rosetta`](rosetta) | Core: contract, frame pipeline, and the protocols for robot interfaces and framework adapters. No ROS. |
| [`rosetta_ros`](robots/ros/rosetta_ros) | The ROS 2 robot interface: codecs, topic bridge, recorder, keyboard and policy runner nodes, `rosetta_port`, node toolkit |
| [`rosetta_interfaces`](robots/ros/rosetta_interfaces) | Action, service and message definitions for every Rosetta ROS 2 node, HIL included |
| [`rosetta_ros_hil`](robots/ros/rosetta_ros_hil) | Human-in-the-loop episode manager, built on `rosetta_ros`'s node toolkit |
| [`lerobot_rosetta`](policies/lerobot/lerobot_rosetta) | LeRobot adapter: dataset writer, policy runner, inference servers |
| [`lerobot_robot_rosetta`](policies/lerobot/lerobot_robot_rosetta) | LeRobot `Robot` plugin, for LeRobot's own CLIs |
| [`lerobot_teleoperator_rosetta`](policies/lerobot/lerobot_teleoperator_rosetta) | LeRobot `Teleoperator` plugin (experimental) |

On ROS 2, install `rosetta_ros`. It has the nodes, launch files and
`rosetta_port`, and brings `rosetta` and `rosetta_interfaces` with it.
`rosetta` alone is the core library, with no nodes.

Adapters can also live in their own repositories. They register the same
entry points, and nothing in this repository changes.
[CONTRIBUTING.md](CONTRIBUTING.md) says which way to go.

## Status

- `main` is developed and tested on ROS 2 Jazzy (Python 3.12). CI runs no
  other distro.
- The LeRobot packages need `lerobot<0.6.2`. LeRobot 0.6.2 removes the async
  inference that the LeRobot policy runner is built on.
- LeRobot's live path takes one numeric observation key and one action key.
  The policy runner and the `Robot` plugin refuse a contract with more.

## License

[Apache-2.0](LICENSE)
