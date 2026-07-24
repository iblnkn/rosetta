<p align="center">
  <img alt="Rosetta" src="media/rosetta_logo.png" width="100%">
</p>
<!-- <p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/ROS2-Jazzy-blue" alt="ROS2">
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python 3.10+">
</p> -->

**Rosetta** connects your ROS 2 robot to robot-learning frameworks like [LeRobot](https://github.com/huggingface/lerobot).

## Quick Start

```
  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
  │  DEFINE  │     │  RECORD  │     │ CONVERT  │     │  TRAIN   │     │  DEPLOY  │
  │ Contract │────▶│  Demos   │────▶│ Dataset  │────▶│  Policy  │────▶│ on Robot │
  └──────────┘     └──────────┘     └──────────┘     └──────────┘     └──────────┘
```

> **Getting started?** See [Installation](https://iblnkn.github.io/rosetta/installation.html). The [rosetta_ws](https://github.com/iblnkn/rosetta_ws) pixi workspace installs ROS2, Rosetta, and LeRobot together in one command; installing into an existing ROS 2 Jazzy workspace is also supported.

**1. Define** a contract for your robot:

```yaml
# my_contract.yaml
robot_type: my_robot
robot_interface: ros2
fps: 30

observations:
  observation.state:
    channel: {topic: /joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: header}
    select: [position.j1, position.j2]

  observation.images.cam:
    channel: {topic: /camera/image_raw/compressed,
              type: sensor_msgs/msg/CompressedImage}
    align: {strategy: hold, timeline: header}
    apply: [resize: [480, 640]]

actions:
  action:
    channel: {topic: /cmd, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: header}
    select: [position.j1, position.j2]
```

**2. Record** demonstrations to rosbag:

```bash
# Terminal 1: Start the recorder
ros2 launch rosetta episode_recorder_launch.py contract_path:=my_contract.yaml
```

```bash
# Terminal 2: Start an episode. Ctrl-C stops and saves it.
ros2 action send_goal /record_episode \
    rosetta_interfaces/action/RecordEpisode "{prompt: 'pick up the red block'}"
```

**3. Convert** bags to a LeRobot dataset:

```bash
rosetta_port \
    --raw-dir ./datasets/bags \
    --contract my_contract.yaml \
    --repo-id my-org/my-dataset \
    --root ./datasets/lerobot
```

**4. Train** with LeRobot:

```bash
lerobot-train \
    --dataset.repo_id=my-org/my-dataset \
    --policy.type=act \
    --output_dir=outputs/train/my_policy
```

**5. Deploy** the trained policy:

```bash
# Terminal 1: Start the policy runner
ros2 launch rosetta policy_runner_launch.py \
    contract_path:=my_contract.yaml \
    pretrained_name_or_path:=my-org/my-policy
```

```bash
# Terminal 2: Run the policy
ros2 action send_goal /run_policy \
    rosetta_interfaces/action/RunPolicy "{prompt: 'pick up red block'}"
```

Full walkthrough: [Train and deploy your first policy](https://iblnkn.github.io/rosetta/tutorials/first-policy.html).

## Documentation

The docs aspire to follow [Diátaxis](https://diataxis.fr/) framework. Start at the [documentation index](https://iblnkn.github.io/rosetta/).

| Section | Purpose | Start with |
|---------|---------|-----------|
| Getting started | Install, then learn by doing | [Installation](https://iblnkn.github.io/rosetta/installation.html), [Your first policy](https://iblnkn.github.io/rosetta/tutorials/first-policy.html) |
| How-to guides | Get a task done | [Write a contract](https://iblnkn.github.io/rosetta/how-to/write-a-contract.html), [Port existing bags](https://iblnkn.github.io/rosetta/how-to/port-existing-bags.html) |
| Reference | Look things up | [Contract](https://iblnkn.github.io/rosetta/reference/contract.html), [Nodes](https://iblnkn.github.io/rosetta/reference/nodes.html), [Actions and services](https://iblnkn.github.io/rosetta/reference/interfaces.html) |
| Explanation | Understand the design | [Design](https://iblnkn.github.io/rosetta/explanation/design.html), [Record raw, decide late](https://iblnkn.github.io/rosetta/explanation/record-raw-decide-late.html) |


## License

Apache-2.0
