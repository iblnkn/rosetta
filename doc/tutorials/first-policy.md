# Train and deploy your first policy

```
  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
  │  DEFINE  │     │  RECORD  │     │ CONVERT  │     │  TRAIN   │     │  DEPLOY  │
  │ Contract │────▶│  Demos   │────▶│ Dataset  │────▶│  Policy  │────▶│ on Robot │
  └──────────┘     └──────────┘     └──────────┘     └──────────┘     └──────────┘
```

We run the full pipeline once on a two-joint arm with one camera. Substitute your
own topics and joint names as you type. You need a ROS 2 robot you can
teleoperate, a camera stream, and a GPU for training.

## 1. Define a contract

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

If your driver does not stamp its messages, write `timeline: receive` instead of
`timeline: header` everywhere. Check with
`ros2 topic echo /joint_states --field header.stamp --once`: a `sec` of `0` means
unstamped.

## 2. Record demonstrations

```bash
# Terminal 1: Start the recorder
ros2 launch rosetta episode_recorder_launch.py contract_path:=my_contract.yaml
```

```bash
# Terminal 2: Keyboard controller (r=start, s=save, d=discard, t=set prompt, q=quit)
ros2 run rosetta episode_keyboard_node
```

Teleoperate the robot through a short task, save, and repeat. Ten episodes is
enough to close the loop, and the policy will be clumsy. Each bag is one episode.

## 3. Convert bags to a dataset

```bash
rosetta_port \
    --raw-dir ./datasets/bags \
    --contract my_contract.yaml \
    --repo-id my-org/my-dataset \
    --root ./datasets/lerobot
```

`ls datasets/lerobot/my-org/my-dataset/meta/` shows `info.json`, `tasks.parquet`,
and `rosetta_contract.yaml`. That last one is the contract from step 1, now
travelling inside the dataset.

## 4. Train

```bash
lerobot-train \
    --dataset.repo_id=my-org/my-dataset \
    --policy.type=act \
    --output_dir=outputs/train/my_policy
```

Training is stock LeRobot, and this is the long step.

## 5. Deploy

```bash
# Terminal 1: Start the policy runner
ros2 launch rosetta policy_runner_launch.py \
    contract_path:=my_contract.yaml \
    pretrained_name_or_path:=outputs/train/my_policy/checkpoints/last/pretrained_model
```

```bash
# Terminal 2: Run the policy
ros2 action send_goal /run_policy \
    rosetta_interfaces/action/RunPolicy "{prompt: 'pick up the red block'}"
```

Use the same prompt you recorded with. Ctrl-C stops execution. The robot now
moves by itself, through the contract we wrote in step 1.
