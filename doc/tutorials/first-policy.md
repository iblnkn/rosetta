# Train and Deploy Your First Policy

In this tutorial we will run the full Rosetta pipeline once: define a contract, record demonstrations, convert them to a dataset, train an ACT policy, and deploy the policy on a robot. Along the way we will meet every Rosetta workflow. At the end, your robot will move with nobody at the controls.

```
  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
  │  DEFINE  │     │  RECORD  │     │ CONVERT  │     │  TRAIN   │     │  DEPLOY  │
  │ Contract │────▶│  Demos   │────▶│ Dataset  │────▶│  Policy  │────▶│ on Robot │
  └──────────┘     └──────────┘     └──────────┘     └──────────┘     └──────────┘
```

One thing before we begin: the policy we train will be clumsy (ten demonstrations is really not enough). But completing the loop on training and deploying this policy, however clumsy, will be a success today.

## What you will need

- A ROS2 robot (real or simulated) with a topic controlling movement (the topic you publish to to tele-operate your robot), and a camera stream.
- ROS2, Rosetta, and LeRobot installed. The [rosetta_ws](https://github.com/iblnkn/rosetta_ws) devcontainer installs all three together.
- A GPU for training.
- An afternoon.

## Before we start: point the tutorial at your robot

 Our example robot is a two-joint arm with one camera. Fill in your robot's values next to the example's:

| The tutorial says | The example value | Your value |
|-------------------|-------------------|------------|
| state topic | `/joint_states` | |
| command topic | `/cmd` | |
| camera topic | `/camera/image_raw/compressed` | |
| two joint names | `j1`, `j2` | |

Now let's confirm your robot holds up its end. With the robot running:

```bash
ros2 topic hz /joint_states
```

You should see a steady rate line, e.g. `average rate: 50.012`. Repeat for the camera topic. If a topic is silent, start your robot's drivers before going further.

```bash
ros2 topic echo /joint_states --field header.stamp --once
```

You should see a nonzero `sec` value. If `sec` is `0`, your driver does not stamp messages. Note this down: wherever the tutorial says `timeline: header`, you will write `timeline: receive` instead.

Write your four values down. From here on, every command uses the example values. Substitute yours as you type.

## Step 1: Define a contract

Create `my_contract.yaml`:

```yaml
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

Let's check our work:

```bash
python3 -c "from rosetta import load_contract; c = load_contract('my_contract.yaml'); print(c.robot_type, c.fps)"
```

You should see:

```
my_robot 30
```

If you see a `ContractValidationError` instead, you have mistyped a key or a message type. The error names the exact problem. Fix and re-run until you see the line above.

**Waypoint.** You have a validated contract: a complete description of your robot's learning interface in one file.

## Step 2: Record demonstrations

Pick a short task your robot can do in under a minute: reach an object, push a block, a repeatable, teleoperatable motion.

```bash
# Terminal 1: Start the recorder
ros2 launch rosetta episode_recorder_launch.py contract_path:=my_contract.yaml
```

The recorder logs its configuration, reaches the `active` lifecycle state, and waits.

```bash
# Terminal 2: Start an episode
ros2 action send_goal /record_episode \
    rosetta_interfaces/action/RecordEpisode "{prompt: 'pick up the red block'}"
```

Use a prompt describing your task. The goal is accepted and feedback begins streaming. Notice the message count ticking upward: every contract topic, and by default everything else on the graph, is flowing into the bag. Teleoperate the robot through the task, then press Ctrl-C in Terminal 2. The episode stops and saves.

Let's check:

```bash
ls datasets/bags/
```

You should see one timestamped bag directory. Run the episode again and a second directory appears beside the first. Each bag is one episode, and nothing about the first is disturbed. Record ten episodes this way, one per run through the task.

## Step 3: Convert bags to a dataset

```bash
rosetta_port \
    --raw-dir ./datasets/bags \
    --contract my_contract.yaml \
    --repo-id my-org/my-dataset \
    --root ./datasets/lerobot
```

The porter logs each bag as it converts, one episode per bag. If the porter reports a warmup failure or zero frames for every bag, revisit the stamp check from the setup section: your observation topics are on a `header` timeline without stamped messages.

Let's check:

```bash
ls datasets/lerobot/my-org/my-dataset/meta/
```

You should see `info.json`, `tasks.parquet`, and `rosetta_contract.yaml`. Notice the last one: the exact contract we wrote in Step 1 now travels inside the dataset.

## Step 4: Train

```bash
lerobot-train \
    --dataset.repo_id=my-org/my-dataset \
    --policy.type=smolvla \
    --output_dir=outputs/train/my_policy
```

Training is stock LeRobot. Loss values stream to the console. This step takes the longest. Grabbing a coffee is appropriate. When training finishes, let's check:

```bash
ls outputs/train/my_policy/checkpoints/last/pretrained_model/
```

You should see the saved policy files, including `train_config.json`.

**Waypoint.** The checkpoint keeps. If the session ends here, deployment is a five-minute step for another day.

## Step 5: Deploy

```bash
# Terminal 1: Start the policy runner
ros2 launch rosetta policy_runner_launch.py \
    contract_path:=my_contract.yaml \
    pretrained_name_or_path:=outputs/train/my_policy/checkpoints/last/pretrained_model
```

The node starts a local policy server and loads the model during configure. Wait for the log line reporting the `active` state. The first start takes longer while the model loads.

```bash
# Terminal 2: Run the policy
ros2 action send_goal /run_policy \
    rosetta_interfaces/action/RunPolicy "{prompt: 'pick up the red block'}"
```

Use the same prompt you recorded with. Now watch the robot.

The motion will be imperfect, and the robot is moving by itself, doing a version of the task you showed it, through the same contract you wrote in Step 1. Ctrl-C stops execution.

You have taken a robot from bare topics to a learned policy: contract, demonstrations, dataset, training, deployment. One YAML defined your robot for every stage, and you have now run each stage yourself.

## Do it again

The experiment with trying to build a better policy by repeating what you just learned: record twenty more episodes, re-run the porter, re-train.

## Where to go next

- Bags from before Rosetta existed port fine: [Port existing bags](../how-to/port-existing-bags.md)
- Richer contracts (multi-source keys, safety, operators): [Write a contract](../how-to/write-a-contract.md)
- Remote GPU inference and contract-free deployment: [Deploy a policy](../how-to/deploy-a-policy.md)
- Why the pipeline is shaped this way: [Design](../explanation/design.md)
