# Rosetta

Rosetta connects a ROS 2 robot to a robot-learning framework such as
[LeRobot](https://github.com/huggingface/lerobot).

A robot publishes topics at whatever rate each sensor runs. A policy wants
one frame per tick, with a fixed set of keys. You write the mapping between
the two in a YAML file, the contract. Rosetta applies that one file when it
builds a dataset from bags and again when it runs the policy on the robot.

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
ros2 launch rosetta episode_recorder_launch.py contract_path:=robot.yaml         # 1 record
$EDITOR robot.yaml                                                                # 2 define
ros2 run rosetta rosetta_port --raw-dir bags --contract robot.yaml --repo-id my  # 3 prepare
lerobot-train --dataset.repo_id=my --policy.type=act                              # 4 train
ros2 launch rosetta policy_runner_launch.py contract_path:=robot.yaml pretrained_name_or_path:=ckpt  # 5 deploy
```

New here? [Install](installation.md), then follow
[From bags to a moving arm](tutorials/first-policy.md).

```{toctree}
:caption: Getting started
:maxdepth: 1

installation
tutorials/first-policy
```

```{toctree}
:caption: How-to guides
:maxdepth: 1

how-to/record-episodes
how-to/write-a-contract
how-to/prepare-a-dataset
how-to/train
how-to/deploy-a-policy
how-to/human-in-the-loop
how-to/add-a-message-type
how-to/add-a-framework
```

```{toctree}
:caption: Reference
:maxdepth: 1

reference/glossary
reference/contract
reference/message-types
reference/rosetta-port
reference/nodes
reference/interfaces
reference/packages
```

```{toctree}
:caption: Explanation
:maxdepth: 1

explanation/design
```
