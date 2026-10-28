# Rosetta

Rosetta connects your ROS 2 robot to robot-learning frameworks such as
[LeRobot](https://github.com/huggingface/lerobot).

Your robot publishes data streams. A policy wants frames.
The contract is a YAML file that defines this transformation from streams to frames.
Rosetta applies the transforms defined in the contract identically for both data preparation and when you go to run your policy live.

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

Five steps take you from data collection to running a policy on your robot.

```bash
ros2 launch rosetta_ros episode_recorder_launch.py contract_path:=robot.yaml                 # 1 record
$EDITOR robot.yaml                                                                           # 2 define
ros2 run rosetta_ros rosetta_port --raw-dir datasets/bags --contract robot.yaml --repo-id my # 3 prepare
lerobot-train --dataset.repo_id=my --policy.type=act --policy.push_to_hub=false \
    --output_dir=outputs/train/my                                                            # 4 train
ros2 launch rosetta_ros policy_runner_launch.py contract_path:=robot.yaml \
    pretrained_name_or_path:=outputs/train/my/checkpoints/last/pretrained_model              # 5 deploy
```

When you change the contract you redo steps 2 to 5. The bags from step 1
stay.

New here? [Install](installation.md), then follow
[From bags to a moving arm](tutorials/first-policy.md).

Getting started installs Rosetta and walks the whole workflow once, on a
simulated arm.

```{toctree}
:caption: Getting started
:maxdepth: 1

installation
tutorials/first-policy
```

How-to guides cover one task each: recording, writing a contract, preparing
a dataset, training, deploying, routing commands through a mux,
human-in-the-loop episodes, and extending
Rosetta with a message type, a framework or a robot interface.

```{toctree}
:caption: How-to guides
:maxdepth: 1

how-to/record-episodes
how-to/write-a-contract
how-to/prepare-a-dataset
how-to/train
how-to/deploy-a-policy
how-to/route-commands
how-to/human-in-the-loop
how-to/add-a-message-type
how-to/add-a-framework
how-to/add-a-robot-interface
```

Reference lists the exact contract keys, message types, CLI flags, node
interfaces and Python API.

```{toctree}
:caption: Reference
:maxdepth: 1

reference/glossary
reference/contract
reference/message-types
reference/rosetta-port
reference/nodes
reference/interfaces
reference/python-api
reference/packages
```

Explanation covers why Rosetta records raw and keeps the translation in one
file.

```{toctree}
:caption: Explanation
:maxdepth: 1

explanation/design
```
