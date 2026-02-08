<p align="center">
  <img alt="Rosetta" src="media/rosetta_logo.png" width="100%">
</p>
<!-- <p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/ROS2-Humble%20%7C%20Jazzy-blue" alt="ROS2">
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python 3.10+">
</p> -->

**Rosetta** brings [LeRobot](https://github.com/huggingface/lerobot) to ROS2 robots. 

## Table of Contents

- [Recent Changes](#recent-changes)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
  - [What is LeRobot?](#what-is-lerobot)
  - [What is Rosetta?](#what-is-rosetta)
- [Architecture](#architecture)
- [The Contract](#the-contract)
- [Recording Episodes](#recording-episodes)
  - [Why Record to Bag Files?](#why-record-to-bag-files)
- [Converting Bags to Datasets](#converting-bags-to-datasets)
- [Training a Policy](#training-a-policy)
  - [Supported Policies](#supported-policies)
- [Deploying Policies](#deploying-policies)
- [Contract Reference](#contract-reference)
  - [Minimal Example](#minimal-example)
  - [Observations](#observations)
  - [Actions](#actions)
  - [Teleop](#teleop)
  - [Tasks, Rewards, and Signals](#tasks-rewards-and-signals)
  - [Selector Syntax](#selector-syntax)
  - [Alignment Strategies](#alignment-strategies)
  - [Supported Message Types](#supported-message-types)
  - [Custom Encoders/Decoders (Experimental)](#custom-encodersdecoders-experimental)
- [LeRobot Data Model Reference](#lerobot-data-model-reference)
  - [Key System](#key-system)
  - [EnvTransition](#envtransition)
  - [Data Types](#data-types)
  - [Policy Feature Compatibility](#policy-feature-compatibility)
- [License](#license)

<a id="recent-changes"></a>
<details>
<summary><strong>Recent Changes</strong></summary>

- **Contract:** `name` → `robot_type`, `rate_hz` → `fps`
- **Nodes:** `PolicyBridge` → `rosetta_client_node`, `EpisodeRecorderServer` → `episode_recorder_node`
- **Actions:** `/run_policy` → `/rosetta_client/run_policy`, `/record_episode` → `/episode_recorder/record_episode`
- **Launch:** `turtlebot_policy_bridge.launch.py` → `rosetta_client_launch.py`, `turtlebot_recorder_server.launch.py` → `episode_recorder_launch.py`
- **Conversion:** `bag_to_lerobot.py` → `port_bags.py` (now processes directories, supports sharding)
- **Inference:** Policy loading moved to LeRobot's async gRPC server
- **New:** `lerobot_teleoperator_rosetta` (experimental), `rosetta_rl` (coming soon)

</details>

---

## Quick Start

```
  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
  │  DEFINE  │     │  RECORD  │     │ CONVERT  │     │  TRAIN   │     │  DEPLOY  │
  │ Contract │────▶│  Demos   │────▶│ Dataset  │────▶│  Policy  │────▶│ on Robot │
  └──────────┘     └──────────┘     └──────────┘     └──────────┘     └──────────┘
```

[**Define**](#the-contract) a contract that maps your ROS2 topics to [LeRobot](https://github.com/huggingface/lerobot) features, [**record**](#recording-episodes) demos to bag files, [**convert**](#converting-bags-to-datasets) them to a LeRobot dataset, [**train**](#training-a-policy) a policy, and [**deploy**](#deploying-policies) it back to your robot.

> **Getting started?** The [rosetta_ws](https://github.com/iblnkn/rosetta_ws) devcontainer handles the non-trivial setup of getting ROS2, Rosetta, and LeRobot installed together.

**1. Define** a [contract](#the-contract) for your robot:

```yaml
# my_contract.yaml
robot_type: my_robot
fps: 30

observations:
  - key: observation.state
    topic: /joint_states
    type: sensor_msgs/msg/JointState
    selector:
      names: [position.j1, position.j2]

  - key: observation.images.cam
    topic: /camera/image_raw/compressed
    type: sensor_msgs/msg/CompressedImage
    image:
      resize: [480, 640]

actions:
  - key: action
    publish:
      topic: /cmd
      type: sensor_msgs/msg/JointState
    selector:
      names: [position.j1, position.j2]
```

**2. Record** demonstrations to rosbag:

```bash
# Terminal 1: Start the recorder
ros2 launch rosetta episode_recorder_launch.py contract_path:=contract.yaml
```

```bash
# Terminal 2: Trigger recording
ros2 action send_goal /episode_recorder/record_episode \
    rosetta_interfaces/action/RecordEpisode "{prompt: 'pick up red block'}"
```

> **How many episodes?** Plan on recording **50–200+ demonstrations** depending on task complexity. More diverse, high-quality demonstrations tend to produce better policies. For practical data collection tips, see [Collecting Your Dataset](https://abenstirling.com/lerobot/) and [Improving Your Robotics AI Model](https://docs.phospho.ai/learn/improve-robotics-ai-model).

**3. Convert** bags to LeRobot dataset:

```bash
python -m rosetta.port_bags \
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
# Terminal 1: Start the client
ros2 launch rosetta rosetta_client_launch.py \
    contract_path:=contract.yaml \
    pretrained_name_or_path:=my-org/my-policy
```

```bash
# Terminal 2: Run the policy
ros2 action send_goal /rosetta_client/run_policy \
    rosetta_interfaces/action/RunPolicy "{prompt: 'pick up red block'}"
```

---

## Core Concepts

### What is LeRobot?

[LeRobot](https://github.com/huggingface/lerobot) is Hugging Face's open-source framework for [robot learning](https://huggingface.co/spaces/lerobot/robot-learning-tutorial). It provides a unified interface for recording demonstrations, training policies (ACT, Diffusion Policy, VLAs like SmolVLA and Pi0), and deploying them on hardware. LeRobot defines a standard dataset format (v3) built on Parquet files and MP4 videos, with a growing ecosystem of community-contributed datasets and models on the [Hugging Face Hub](https://huggingface.co/datasets?other=LeRobot).

### What is Rosetta?

Rosetta is a set of ROS2 packages and tools to bring state-of-the-art robot learning capabilites to the ROS 2 community. Specifically leveraging LeRobot for training and inference.

## Architecture

Rosetta consists of five packages that implement LeRobot's official interfaces:

| Package | Purpose |
|---------|---------|
| `rosetta` | Core library, nodes, bag conversion |
| [`rosetta_interfaces`](https://github.com/iblnkn/rosetta_interfaces) | ROS2 action/service definitions |
| [`lerobot_robot_rosetta`](https://github.com/iblnkn/lerobot-robot-rosetta) | LeRobot Robot plugin |
| [`lerobot_teleoperator_rosetta`](https://github.com/iblnkn/lerobot-teleoperator-rosetta) | LeRobot Teleoperator plugin (experimental) |
| `rosetta_rl` | HIL-SERL reinforcement learning (coming soon) |

```
rosetta/
├── launch/
│   ├── episode_recorder_launch.py
│   └── rosetta_client_launch.py
└── params/
    ├── episode_recorder.yaml    # Default config for Episode Recorder
    └── rosetta_client.yaml      # Default config for Rosetta Client
```

### LeRobot Plugin Architecture

The `lerobot_robot_rosetta` and `lerobot_teleoperator_rosetta` packages implement LeRobot's [Robot](https://huggingface.co/docs/lerobot/integrate_hardware) and [Teleoperator](https://huggingface.co/docs/lerobot/integrate_hardware#adding-a-teleoperator) interfaces. They follow LeRobot's [plugin conventions](https://huggingface.co/docs/lerobot/integrate_hardware#using-your-own-lerobot-devices-) (`lerobot_robot_*` and `lerobot_teleoperator_*` prefixes) for auto-discovery when installed.

**Typical LeRobot robots** (like `so101_follower`) communicate directly with hardware:
- Motors via serial/CAN (`FeetechMotorsBus`, `DynamixelMotorsBus`)
- Cameras via USB/OpenCV
- The `Robot` class IS the hardware interface

**Rosetta robots** are ROS2 lifecycle nodes:
- Subscribe to ROS2 topics for observations
- Publish to ROS2 topics for actions
- Hardware drivers exist elsewhere in the ROS2 graph
- The contract YAML defines topic-to-feature mapping

**Important:** Because `lerobot_robot_rosetta` creates a ROS2 lifecycle node internally, **your system needs ROS2 installed** to use it, even when invoking it through LeRobot's standard CLI tools. When `rosetta_client_node` launches inference, the chain is: `rosetta_client_node` (ROS2 node) → LeRobot `RobotClient` → `lerobot_robot_rosetta` (also a ROS2 node) → your robot's ROS2 topics. Both the convenience node and the robot plugin are ROS2 nodes running in the same ROS2 graph.

This means any ROS2 robot can use LeRobot's tools. Define a contract and use `--robot.type=rosetta`.

### ROS2 Lifecycle Integration

LeRobot's `connect()` / `disconnect()` map to ROS2 lifecycle transitions:

| LeRobot Method | Lifecycle Transition | Effect |
|----------------|---------------------|--------|
| - | `configure` | Create subscriptions (start buffering), create publishers (disabled) |
| `connect()` | `activate` | Enable publishers, start watchdog |
| `disconnect()` | `deactivate` → `cleanup` | Safety action, disable publishers, destroy resources |

### Policy Inference

The `rosetta_client_node` delegates inference to LeRobot's async gRPC policy server (`lerobot.async_inference.policy_server`). This server is a standard LeRobot component with no ROS2 dependency and can run on any machine with LeRobot and a GPU. Benefits:

- Better GPU memory management
- Support for all LeRobot policy types without code changes
- Consistent behavior between training and deployment
- Can run on a remote machine, letting a resource-constrained robot offload inference over the network

### rosetta_ws Workspace

We provide [rosetta_ws](https://github.com/iblnkn/rosetta_ws), a devcontainer workspace for getting started quickly. Getting ROS2 and LeRobot installed together is not trivial; the workspace handles this setup.

---

## The Contract

The contract defines the translation between ROS 2 topics and the keys LeRobot expects.

| ROS2 Side | | LeRobot Side |
|-----------|---|-------------|
| `/front_camera/image_raw/compressed` | &rarr; | `observation.images.front` |
| `/follower_arm/joint_states` (position fields) | &rarr; | `observation.state` |
| `/imu/data` (orientation, angular_velocity) | &rarr; | `observation.state.imu` |
| `/leader_arm/joint_states` (position fields) | &larr; | `action` |
| `/base_controller/cmd_vel` (linear, angular) | &larr; | `action.base` |
| `/task_prompt` (String) | &rarr; | `task` |
| `/reward_signal` (Float64) | &rarr; | `next.reward` |

On the ROS2 side, data lives in typed messages on named topics with rich structure (headers, arrays, nested fields). On the LeRobot side, data lives in flat dictionaries with dot-separated string keys and numpy/tensor values. The contract maps one to the other, handling type conversion, field extraction, timestamp alignment, and resampling.

Here's how a concrete contract entry translates a ROS2 topic to a LeRobot feature:

```yaml
- key: observation.state
  topic: /follower_arm/joint_states
  type: sensor_msgs/msg/JointState
  selector:
    names: [position.shoulder_pan, position.shoulder_lift, position.elbow,
            position.wrist_pitch, position.wrist_roll, position.wrist_yaw]
```

At each timestep, this:
1. **Subscribes** to `/follower_arm/joint_states` (a `JointState` message)
2. **Extracts** the named fields using dot notation (`position.shoulder_pan` → `msg.position[msg.name.index("shoulder_pan")]`)
3. **Assembles** a numpy array: `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]` (dtype `float64`)
4. **Stores** it under the key `observation.state` in the LeRobot dataset

**Multi-topic concatenation**: Multiple contract entries can map to the **same key**. Their values are concatenated in declaration order. This lets you combine data from separate ROS2 topics into a single feature vector:

```yaml
observations:
  # These two entries share the same key; values are concatenated
  - key: observation.state
    topic: /arm/joint_states
    type: sensor_msgs/msg/JointState
    selector:
      names: [position.j1, position.j2, position.j3]

  - key: observation.state
    topic: /gripper/state
    type: std_msgs/msg/Float32
    # Result: observation.state = [j1, j2, j3, gripper] (4D vector)
```
This is important because, as shown in [Policy Feature Compatibility](#policy-feature-compatibility), all of the available core policies depend on explicit names for most keys. If you have multiple ros2 topics you would like to use for observation, the most straightforward way to achieve this is to include both topics with the same key name.

For images, each image key must be unique; image features cannot be concatenated.

A minimal contract typically only needs `observations` and `actions`. See the full [Contract Reference](#contract-reference) for all options, and the [LeRobot Data Model Reference](#lerobot-data-model-reference) for how keys, features, and policies interact.

---

## Recording Episodes

The `episode_recorder_node` is a convenience node that records contract-specified topics to [rosbag2](https://github.com/ros2/rosbag2) files. It reads the contract to determine which topics to subscribe to, then lets you start and stop recording via ROS2 actions, with feedback on duration and message count.

**This node is not the only way to record compatible bags.** Any method that produces a valid rosbag2 file containing the contract's topics will work, including `ros2 bag record`, custom scripts using `rosbag2_py`, or third-party recording tools. The `episode_recorder_node` makes this convenient within the ROS2 ecosystem: you define your topics once in the contract, and it handles subscription setup, bag lifecycle, and action-based control. It may also be useful standalone for any workflow where you need to define a set of topics and start/stop recording programmatically via ROS2 actions.

> Both Rosetta nodes use parameter files (`params/`) as defaults. All parameters are also exposed as launch arguments, which override the defaults. Run `ros2 launch rosetta <launch_file> --show-args` to see all options.

```bash
ros2 launch rosetta episode_recorder_launch.py contract_path:=/path/to/contract.yaml
```

Trigger recording:

```bash
ros2 action send_goal /episode_recorder/record_episode \
    rosetta_interfaces/action/RecordEpisode "{prompt: 'task description'}"
```

**Parameters** (all available as launch arguments):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `contract_path` | `contracts/so_101.yaml` | Path to contract YAML |
| `bag_base_dir` | `/workspaces/rosetta_ws/datasets/bags` | Directory for rosbag output |
| `storage_id` | `mcap` | Rosbag format: `mcap` (recommended) or `sqlite3` |
| `default_max_duration` | `300.0` | Max episode duration in seconds |
| `feedback_rate_hz` | `2.0` | Recording feedback publish rate |
| `default_qos_depth` | `10` | QoS queue depth for subscriptions |
| `log_level` | `info` | Logging level: `debug`, `info`, `warn`, `error` |
| `configure` | `true` | Auto-configure on startup |
| `activate` | `true` | Auto-activate on startup |

**Examples:**

```bash
# Override output directory
ros2 launch rosetta episode_recorder_launch.py \
    contract_path:=/path/to/contract.yaml \
    bag_base_dir:=/data/recordings

# Change max duration and storage format
ros2 launch rosetta episode_recorder_launch.py \
    contract_path:=/path/to/contract.yaml \
    default_max_duration:=600.0 \
    storage_id:=sqlite3
```

### Why Record to Bag Files?

Rosetta records demonstrations to [rosbag2](https://github.com/ros2/rosbag2) files first, then converts them to LeRobot datasets in a separate step. This is a deliberate design choice with several benefits:

- **Preserves raw data.** Bag files store every message at its original rate and timestamp, with no alignment, downsampling, or lossy transformation. This means you can reprocess the same recordings later with a different contract (changing feature keys, adjusting resampling rates, adding new topics) without re-recording.
- **Familiar to ROS2 users.** Bag files are the standard data format in the ROS2 ecosystem, with mature tooling for [recording, playback, inspection](https://docs.ros.org/en/jazzy/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html), and analysis. Any tool that works with bag files works with your recorded data.
- **Stores data beyond what LeRobot needs.** Bags can include topics that don't map to any LeRobot feature: diagnostics, TF trees, debug streams, extra sensors. This data is preserved for analysis, debugging, or future use even though it isn't part of the training dataset.
- **Leverages MCAP.** Rosetta defaults to [MCAP](https://mcap.dev/) storage, which provides [high-performance](https://mcap.dev/guides/benchmarks/rosbag2-storage-plugins) random-access reads, efficient compression, and broad ecosystem support beyond ROS2.
- **Write-optimized for live recording.** Bag files (especially MCAP) are designed for high-throughput sequential writes with minimal overhead, well-suited for capturing live sensor data. LeRobot datasets (Parquet + MP4) are read-optimized for training but involve more overhead when writing live, including in-memory buffering and post-episode video encoding.


## Converting Bags to Datasets

`port_bags.py` converts rosbag2 files to LeRobot datasets using the contract for key mapping, timestamp alignment, resampling, and dtype conversion. It applies the same `StreamBuffer` resampling logic used during live inference, ensuring your offline dataset matches what the robot would see at runtime.

While you could write your own conversion script using the primitives in `rosetta.common` (contract loader, decoders, stream buffers), `port_bags.py` handles the full pipeline: reading bags, applying the contract, encoding video, building the LeRobot dataset structure, and optionally pushing to the Hub. Because the raw bag preserves all data without transformation, you can re-run `port_bags.py` with an updated contract (changing keys, adjusting `fps`, adding or removing features) without re-recording.


### Relationship to LeRobot

`port_bags.py` mirrors the interface of LeRobot's example porters (like `port_droid.py`):

```bash
# LeRobot's port_droid.py
python examples/port_datasets/port_droid.py \
    --raw-dir /data/droid/1.0.1 \
    --repo-id my_org/droid \
    --push-to-hub

# Rosetta's port_bags.py (same pattern + contract)
python -m rosetta.port_bags \
    --raw-dir ./datasets/bags \
    --contract contract.yaml \
    --repo-id my_org/my_dataset \
    --root ./datasets/lerobot
```

**Rosetta-specific additions:**

| Argument | Description |
|----------|-------------|
| `--contract` | **(Required)** Rosetta contract YAML that defines ROS2 topic → LeRobot feature mapping |
| `--root` | Override output directory (LeRobot defaults to `~/.cache/huggingface/lerobot`) |
| `--vcodec` | Video codec selection (not in base LeRobot porters) |

### Basic Usage

```bash
python -m rosetta.port_bags \
    --raw-dir ./datasets/bags \
    --contract ./contract.yaml \
    --repo-id my_dataset \
    --root ./datasets/lerobot
```

 For additional information on large-scale conversions, parallel processing, and SLURM cluster workflows, see the **[LeRobot Porting Datasets Guide](https://huggingface.co/docs/lerobot/en/porting_datasets_v3)** and substitute `port_bags.py` for `port_droid.py` in the examples. 



## Training a Policy

Once you've converted your ROS2 bags to a LeRobot dataset, [train a policy](https://huggingface.co/docs/lerobot/il_robots#train-a-policy) with `lerobot-train`.


### Quick Start: ACT

```bash
lerobot-train \
    --dataset.repo_id=my-org/my-dataset \
    --policy.type=act \
    --output_dir=outputs/train/act_my_robot \
    --policy.device=cuda \
    --wandb.enable=true
```

### Fine-tuning VLA Models

VLA models are large pre-trained vision-language-action models. Use [PEFT](https://huggingface.co/docs/peft/index)/[LoRA](https://huggingface.co/docs/peft/task_guides/lora_based_methods) for [efficient fine-tuning](https://huggingface.co/docs/lerobot/peft_training):

```bash
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=my-org/my-dataset \
    --policy.output_features=null \
    --policy.input_features=null \
    --steps=100000 \
    --batch_size=32 \
    --peft.method_type=LORA \
    --peft.r=64
```

### Multi-GPU Training

LeRobot supports [training on multiple GPUs](https://huggingface.co/docs/lerobot/multi_gpu_training) using [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index):

```bash
accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --mixed_precision=fp16 \
    $(which lerobot-train) \
    --dataset.repo_id=my-org/my-dataset \
    --policy.type=act \
    --batch_size=32
```

### Resume Training

```bash
lerobot-train \
    --config_path=outputs/train/my_run/checkpoints/last/pretrained_model/train_config.json \
    --resume=true
```

### Upload to HuggingFace Hub

```bash
huggingface-cli upload my-org/my-policy \
    outputs/train/my_run/checkpoints/last/pretrained_model
```


### Supported Policies

| Policy | Type | Best For |
|--------|------|----------|
| [**ACT**](https://huggingface.co/docs/lerobot/act) | Behavior Cloning | General manipulation, fast training (recommended for beginners) |
| [**SmolVLA**](https://huggingface.co/docs/lerobot/smolvla) | VLA | Efficient VLA, good for resource-constrained setups |
| [**Pi0**](https://huggingface.co/docs/lerobot/pi0) / [**Pi0Fast**](https://huggingface.co/docs/lerobot/pi0fast) | VLA | Physical Intelligence foundation models |
| [**Pi0.5**](https://huggingface.co/docs/lerobot/pi05) | VLA | Open-world generalization |
| [**NVIDIA GR00T N1.5**](https://huggingface.co/docs/lerobot/groot) | VLA | Humanoid and general robotics |
| [**Wall-X**](https://huggingface.co/docs/lerobot/walloss) | VLA | Qwen 2.5-VL backbone, multi-embodiment |
| [**X-VLA**](https://huggingface.co/docs/lerobot/xvla) | VLA | Cross-embodiment with soft prompts |

## Deploying Policies

The `rosetta_client_node` is a convenience node that wraps LeRobot's inference pipeline in ROS2 actions. It lets you start and stop policy execution via `ros2 action send_goal`, with feedback on inference progress. It can optionally launch a local LeRobot gRPC policy server as a subprocess, or connect to a remote one.

Launch Client:

```bash
ros2 launch rosetta rosetta_client_launch.py contract_path:=/path/to/contract.yaml
```

Run policy:

```bash
ros2 action send_goal /rosetta_client/run_policy \
    rosetta_interfaces/action/RunPolicy "{prompt: 'task description'}"
```

**Remote inference:** When `launch_local_server` is `false`, the node connects to a LeRobot gRPC policy server at `server_address`. This server is a standard LeRobot component with no ROS2 dependency. It can run on any machine with a GPU, completely independent of your robot's ROS2 environment. This lets a resource-constrained robot offload inference to a remote GPU server.

**Parameters** (all available as launch arguments):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `contract_path` | `contracts/so_101.yaml` | Path to contract YAML |
| `pretrained_name_or_path` | *(see params file)* | HuggingFace model ID or local path |
| `server_address` | `127.0.0.1:8080` | Policy server address |
| `policy_type` | `act` | Policy type: `act`, `smolvla`, `diffusion`, `pi0`, `pi05`, etc. |
| `policy_device` | `cuda` | Inference device: `cuda`, `cpu`, `mps`, or `cuda:0` |
| `actions_per_chunk` | `30` | Actions per inference chunk |
| `chunk_size_threshold` | `0.95` | When to request new chunk (0.0-1.0) |
| `aggregate_fn_name` | `weighted_average` | Chunk aggregation: `weighted_average`, `latest_only`, `average`, `conservative` |
| `feedback_rate_hz` | `2.0` | Execution feedback publish rate |
| `launch_local_server` | `true` | Auto-start policy server subprocess |
| `obs_similarity_atol` | `-1.0` | Observation filtering tolerance (-1.0 to disable)* |
| `log_level` | `info` | Logging level: `debug`, `info`, `warn`, `error` |
| `configure` | `true` | Auto-configure on startup |
| `activate` | `true` | Auto-activate on startup |

*\*`obs_similarity_atol`: The policy server filters observations that are "too similar" (L2 norm of state difference < threshold). The default threshold (1.0) assumes joint states change significantly between frames. Many robots have smaller movements, causing most observations to be skipped. Set to `-1.0` to disable filtering.*

**Example:**

```bash
# Run with a pretrained model
ros2 launch rosetta rosetta_client_launch.py \
    contract_path:=/path/to/contract.yaml \
    pretrained_name_or_path:=my-org/my-policy
```

**This node is not the only way to deploy.** You can run inference using LeRobot's standard CLI tools directly with the Rosetta robot plugin:

```bash
# Standard LeRobot deployment, no rosetta_client_node needed
lerobot-record --robot.type=rosetta --robot.config_path=contract.yaml
```

See [Imitation Learning on Real Robots](https://huggingface.co/docs/lerobot/il_robots) for LeRobot's native deployment workflow. The `rosetta_client_node` adds ROS2 action-based lifecycle management on top of this, which is convenient if your workflow is already ROS2-centric.

---

## Contract Reference

A contract is a YAML file that maps ROS2 topics to LeRobot's observation/action interface. The contract currently maps to the full LeRobot `EnvTransition` interface:

| Contract Section | EnvTransition Slot | Status |
|-----------------|-------------------|--------|
| `observations` | `observation.*` | Supported |
| `actions` | `action*` | Supported |
| `tasks` | `complementary_data.task` | Supported |
| `rewards` | `next.reward` | Supported |
| `signals` | `next.done`, `next.truncated` | Supported |
| `complementary_data` | `complementary_data.*` | Supported |

Not every section needs to be filled for every robot. A minimal contract only needs `observations` and `actions`. To see which keys are required or accepted by different policies, see [Policy Feature Compatibility](#policy-feature-compatibility).

### Minimal Example

```yaml
robot_type: my_robot
fps: 30

observations:
  - key: observation.state
    topic: /joint_states
    type: sensor_msgs/msg/JointState
    selector:
      names: [position.j1, position.j2]

actions:
  - key: action
    publish:
      topic: /joint_commands
      type: sensor_msgs/msg/JointState
    selector:
      names: [position.j1, position.j2]
```

### Observations

```yaml
observations:
  # State vector (with all optional fields shown)
  - key: observation.state
    topic: /joint_states
    type: sensor_msgs/msg/JointState
    selector:
      names: [position.j1, velocity.j1]
    align:
      strategy: hold    # hold (default), asof, drop
      stamp: header     # header (default), receive
    qos:
      reliability: best_effort
      depth: 10

  # Camera
  - key: observation.images.camera
    topic: /camera/image_raw/compressed
    type: sensor_msgs/msg/CompressedImage
    image:
      resize: [224, 224]  # [height, width]
```

Multiple topics can share the same `key`. Values are concatenated (see [The Contract](#the-contract)).

### Actions

```yaml
actions:
  - key: action
    publish:
      topic: /joint_commands
      type: sensor_msgs/msg/JointState
      qos: {reliability: reliable, depth: 10}
    selector:
      names: [position.j1, position.j2]
    safety_behavior: hold  # none, hold, zeros
```

### Teleop

For human-in-the-loop recording with a leader arm or other input device:

```yaml
teleop:
  inputs:
    - key: teleop_input
      topic: /leader_arm/joint_states
      type: sensor_msgs/msg/JointState
      selector:
        names: [position.j1, position.j2]

  events:
    topic: /joy
    type: sensor_msgs/msg/Joy
    mappings:
      is_intervention: buttons.5
      success: buttons.0
      terminate_episode: buttons.6
      rerecord_episode: buttons.7
      failure: buttons.1
```

### Tasks, Rewards, and Signals

These sections are optional. Use them when your workflow requires task prompts from ROS2 topics, RL reward signals, or episode termination signals.

```yaml
tasks:
  - key: task
    topic: /task_prompt
    type: std_msgs/msg/String

rewards:
  - key: next.reward
    topic: /reward
    type: std_msgs/msg/Float64
    dtype: float64

signals:
  - key: next.done
    topic: /episode_done
    type: std_msgs/msg/Bool

  - key: next.truncated
    topic: /episode_truncated
    type: std_msgs/msg/Bool
```

For VLA policies, the `task` string can also be provided via the `prompt` argument when recording or running a policy, so you don't need a ROS2 topic for it.

### Selector Syntax

Dot notation extracts nested fields from ROS2 messages:

```yaml
# JointState: {field}.{joint_name}
names: [position.shoulder, velocity.shoulder]

# Odometry: nested path
names: [twist.twist.linear.x, pose.pose.position.z]
```

### Alignment Strategies

| Strategy | Behavior |
|----------|----------|
| `hold` | Use most recent message, no matter how old (default) |
| `asof` | Use most recent message only if within `tol_ms` tolerance window, otherwise return nothing (zero-filled). Useful for rejecting stale data |
| `drop` | Use most recent message only if it arrived within the current step/frame window |

### Supported Message Types

| Type | Extracted Fields |
|------|------------------|
| `sensor_msgs/msg/JointState` | position, velocity, effort by joint name |
| `sensor_msgs/msg/Image` | RGB uint8 array |
| `sensor_msgs/msg/CompressedImage` | Decoded to RGB uint8 |
| `geometry_msgs/msg/Twist` | linear.xyz, angular.xyz |
| `nav_msgs/msg/Odometry` | pose, twist fields |
| `sensor_msgs/msg/Joy` | axes, buttons arrays |
| `sensor_msgs/msg/Imu` | orientation, angular_velocity, linear_acceleration |
| `std_msgs/msg/Float32` | Scalar float32 |
| `std_msgs/msg/Float64` | Scalar float64 |
| `std_msgs/msg/String` | Text string |
| `std_msgs/msg/Bool` | Boolean |
| `std_msgs/msg/Float64MultiArray` | Vector float64 |

The dtype is auto-detected from the message type. You can override it with the `dtype` field in the contract, or use a custom decoder for non-standard types.

### Custom Encoders/Decoders (Experimental)

> **Note:** Custom encoder/decoder support is **experimental**.

Add support for unsupported ROS message types by writing custom decoders (ROS → numpy) and encoders (numpy → ROS).

#### Method 1: Specify in Contract (Recommended)

Point directly to your converter functions in the contract YAML:

```yaml
observations:
  - key: observation.state
    topic: /my_sensor
    type: my_msgs/msg/MyCustomSensor
    decoder: my_package.converters:decode_my_sensor  # module:function

actions:
  - key: action
    publish:
      topic: /my_command
      type: my_msgs/msg/MyCustomCommand
    decoder: my_package.converters:decode_my_command  # for reading bags
    encoder: my_package.converters:encode_my_command  # for publishing
```

The module must be importable in your Python environment. Paths are validated at contract load time.

#### Method 2: Global Registration

Register converters globally so they're used for all instances of a message type:

```python
# my_converters.py
import numpy as np
from rosetta.common.converters import register_decoder, register_encoder

@register_decoder("my_msgs/msg/MyCustomSensor", dtype="float64")
def decode_my_sensor(msg, spec):
    return np.array([msg.field1, msg.field2], dtype=np.float64)

@register_encoder("my_msgs/msg/MyCustomCommand")
def encode_my_command(values, spec, stamp_ns=None):
    from my_msgs.msg import MyCustomCommand
    msg = MyCustomCommand()
    msg.field1, msg.field2 = float(values[0]), float(values[1])
    return msg
```

Import before using Rosetta:

```python
import my_converters  # Registers on import
from lerobot_robot_rosetta import Rosetta, RosettaConfig
robot = Rosetta(RosettaConfig(config_path="contract.yaml"))
```

#### Function Signatures

**Decoder:** Converts ROS message → numpy array

```python
def my_decoder(msg, spec) -> np.ndarray:
    # msg: ROS message instance
    # spec.names: list of selector names from contract
    # spec.msg_type: ROS message type string
    return np.array([...], dtype=np.float64)
```

**Encoder:** Converts numpy array → ROS message

```python
def my_encoder(values, spec, stamp_ns=None):
    # values: numpy array of action values
    # spec.names: list of selector names from contract
    # spec.clamp: optional (min, max) tuple
    # stamp_ns: optional timestamp in nanoseconds
    msg = MyMessage()
    # ... populate msg from values ...
    return msg
```

#### When Each Is Used

| Field | Used By | Purpose |
|-------|---------|---------|
| `decoder` on observations | Runtime, `port_bags.py` | Decode incoming sensor data |
| `decoder` on actions | `port_bags.py` | Read recorded actions from bags |
| `encoder` on actions | Runtime | Publish actions to ROS topics |

---


## LeRobot Data Model Reference

This section covers LeRobot's internal data model in detail. You don't need this to get started. Refer back here when you need to understand key conventions, feature types, or policy compatibility.

### Key System


**LeRobot keys are flat dictionary strings that use dots as a naming convention.** `observation.state.joint_position` is a single string key, not a nested lookup. The only hard rule is **no forward slashes** (`/`) in key names.

This means you can create keys at any depth:

```python
# These are all valid, independent LeRobot feature keys:
"observation.state"                              # (14,) float64
"observation.state.joint_position"               # (7,)  float32
"observation.state.gripper_position"             # (1,)  float32
"observation.state.imu.orientation"              # (4,)  float64
"observation.environment_state"                  # (25,) float64
"observation.environment_state.object_positions" # (12,) float32
"observation.images.front"                       # (480, 640, 3) video
"observation.images.wrist.left"                  # (480, 640, 3) video
"action"                                         # (8,) float32
"action.arm"                                     # (6,) float32
"action.gripper"                                 # (1,) float32
"action.base"                                    # (2,) float32
```

There is no parent-child relationship between these keys. `observation.state` and `observation.state.joint_position` can coexist as completely independent features with different shapes. They just happen to share a prefix.

#### How LeRobot classifies keys

While keys are free-form strings, LeRobot policies use **prefix matching** to classify them into feature types. This classification determines how policies process each feature:

| Prefix | FeatureType | How policies use it |
|--------|-------------|---------------------|
| `observation.images.*` or `observation.image` | `VISUAL` | Fed through vision encoder |
| `observation.environment_state` (exact) | `ENV` | Separate encoder projection (privileged sim state) |
| `observation.*` (everything else under observation) | `STATE` | Robot state encoder |
| `observation.language.*` | `LANGUAGE` | Tokenized text for VLA forward pass |
| `action*` | `ACTION` | Policy output / training target |
| `next.reward` | `REWARD` | RL reward signal |

This means `observation.state.imu`, `observation.state.joint_position`, and `observation.state` are all classified as `STATE`. Similarly, `action.arm` and `action.gripper` are both `ACTION`.

#### Convention vs. compatibility

LeRobot's key system has two layers:

1. **The dataset format** accepts any key string. You can store `observation.state.fake_sensor.special_data` or `my_custom_thing` and it works.
2. **Built-in policies** look for specific keys by exact match. ACT, SmolVLA, and Pi0 all expect `observation.state` and `action` as single combined vectors.

The [DROID dataset](https://huggingface.co/datasets/lerobot/droid) demonstrates the recommended approach when you need both richness and compatibility: **store split sub-keys alongside combined keys**:

```python
# Split sub-keys (rich, self-documenting):
"observation.state.joint_position":     {"shape": (7,)}
"observation.state.cartesian_position": {"shape": (6,)}
"observation.state.gripper_position":   {"shape": (1,)}

# Combined key (policy-compatible):
"observation.state":                    {"shape": (8,)}  # joints + gripper

# Same pattern for actions:
"action.joint_position":    {"shape": (7,)}
"action.gripper_position":  {"shape": (1,)}
"action":                   {"shape": (8,)}  # joints + gripper
```

The sub-keys preserve semantic meaning and enable richer downstream analysis. The combined keys keep existing policies working without modification.

### EnvTransition

LeRobot defines a [Universal Data Container](https://huggingface.co/docs/lerobot/introduction_processors#envtransition-the-universal-data-container) that descends from the classic Gymnasium `step()` return (`observation, reward, terminated, truncated, info`), called `EnvTransition`.

The `EnvTransition` TypedDict defines six top-level slots. The contract aims to make explicit the mapping between ROS2 and the semantic categories defined by the EnvTransition. No core policy currently leverages all components.

#### Observation (`observation.*`)

Everything the robot senses. Sub-divided by modality:

```
observation.
├── state                           # Robot proprioception (joints, EEF pose)
│   ├── joint_position              #   Optional: split out joint positions
│   ├── cartesian_position          #   Optional: split out EEF pose
│   └── gripper_position            #   Optional: split out gripper
│
├── environment_state               # External/privileged state (sim only)
│   ├── object_positions            #   Optional: sub-key for object poses
│   └── contact_forces              #   Optional: sub-key for forces
│
├── images.                         # Camera feeds (stored as MP4 video)
│   ├── top                         #   Overhead / third-person view
│   ├── front                       #   Front-facing view
│   ├── left / right                #   Side views
│   ├── wrist.left / wrist.right    #   Wrist-mounted cameras
│   └── wrist.top / wrist.bottom    #   Wrist camera orientations
│
└── language                        # Tokenized text (generated by processor)
    ├── tokens                      #   Token IDs (int tensor)
    └── attention_mask              #   Attention mask (bool tensor)
```

**`observation.state`** vs **`observation.environment_state`**: These are semantically distinct. `state` is the robot's proprioception, i.e. what the robot knows about its own body (joint angles, gripper width, EEF pose). `environment_state` is privileged information about the external world (object positions, contact forces), typically only available in simulation. They have different `FeatureType`s (`STATE` vs `ENV`) and policies encode them with separate projections.

#### Action (`action*`)

Motor commands the robot executes:

```
action                              # Combined action vector (policy-compatible)
├── joint_position                  # Optional: split out joint commands
├── cartesian_position              # Optional: split out EEF commands
├── gripper_position                # Optional: split out gripper
├── base                            # Optional: mobile base velocity
└── arm1.fingers                    # Optional: arbitrary depth is allowed
```

Most built-in policies expect a single `action` key. If you split into sub-keys, also provide the combined `action` for compatibility (see the DROID pattern above).

#### Task and Language

These serve different purposes and can coexist:

| Concept | Key(s) | Type | Purpose |
|---------|--------|------|---------|
| **Task string** | `task` | `str` | Human-readable label: `"pick up the red block"` |
| **Language tokens** | `observation.language.tokens` | `Tensor (int)` | Tokenized text for VLA forward pass |
| **Language mask** | `observation.language.attention_mask` | `Tensor (bool)` | Attention mask for tokenized text |

The **flow** between them: the dataset stores a `task_index` (int) per frame, which resolves to a `task` string via `meta/tasks.parquet`. How that string reaches the policy depends on the policy:

- **Pre-tokenized** (SmolVLA, Pi0, Pi0Fast, Pi0.5, X-VLA): LeRobot's `TokenizerProcessorStep` reads the `task` string and produces `observation.language.tokens` and `observation.language.attention_mask` tensors. The policy consumes these tensors.
- **Internally tokenized** (GR00T, Wall-X): The raw `task` string is passed directly to the policy, which tokenizes it through its own VLM backbone (Eagle 2.5 for GR00T, Qwen 2.5-VL for Wall-X).

`task` is always a single string per frame. `subtask` is a recognized complementary data key.

#### Reward and Episode Signals

RL signals and episode boundaries:

```
next.reward                         # Scalar float: RL reward signal
next.done                           # Bool: episode terminated naturally (goal reached, failure)
next.truncated                      # Bool: episode ended artificially (time limit)
```

These use the `next.` prefix because they describe the outcome *after* taking the action.

#### Complementary Data

Per-frame metadata that flows through training but isn't a model input:

```
task                                # Task description string (resolved from task_index)
task_index                          # int64: index into meta/tasks.parquet
episode_index                       # int64: which episode this frame belongs to
frame_index                         # int64: position within the episode
index                               # int64: global frame index
timestamp                           # float32: time in seconds
observation.state_is_pad            # bool tensor: padding flag for state
observation.images.front_is_pad     # bool tensor: padding flag per image key
action_is_pad                       # bool tensor: padding flag for action
```

The `*_is_pad` flags mark which frames in a temporal window are real vs. padded (used when a policy looks at multiple past frames and some haven't occurred yet).

The five default features (`timestamp`, `frame_index`, `episode_index`, `index`, `task_index`) are automatically added to every dataset. You don't need to declare them.

#### Info

The `info` slot in `EnvTransition` is **runtime-only** and is not persisted to datasets. It carries transient signals like teleop events (`is_intervention`, `rerecord_episode`, `terminate_episode`) used during live recording and policy execution. If you need persistent metadata, use `complementary_data` instead.

Note: `meta/info.json` in the dataset directory is unrelated; it stores the dataset schema (features, fps, robot_type), not per-frame data.

### Data Types

Each feature key maps to a specific data type. LeRobot datasets support:

| Data Type | LeRobot dtype | Shape | Description | Example Keys |
|-----------|--------------|-------|-------------|-------------|
| **Float vector** | `float32` / `float64` | `(N,)` | Continuous values: joints, poses, velocities | `observation.state`, `action` |
| **Image** | `video` | `(H, W, 3)` | RGB uint8 frames, stored as MP4 | `observation.images.*` |
| **String** | `string` | `(1,)` | Text labels, prompts | `task`, `language_instruction` |
| **Boolean** | `bool` | `(1,)` or `(N,)` | Binary flags | `next.done`, `action_is_pad` |
| **Integer** | `int32` / `int64` | `(1,)` or `(N,)` | Discrete values, indices | `task_index`, `episode_index` |

In the Rosetta contract, dtype is usually **auto-detected** from the ROS2 message type:

| ROS2 Message Type | Auto dtype | Output |
|-------------------|-----------|--------|
| `sensor_msgs/msg/JointState` | `float64` | Selected position/velocity/effort values |
| `sensor_msgs/msg/CompressedImage` | `video` | RGB uint8 `(H, W, 3)` |
| `sensor_msgs/msg/Image` | `video` | RGB uint8 `(H, W, 3)` |
| `geometry_msgs/msg/Twist` | `float64` | Selected linear/angular components |
| `nav_msgs/msg/Odometry` | `float64` | Selected pose/twist fields |
| `sensor_msgs/msg/Imu` | `float64` | Orientation, angular vel, linear accel |
| `std_msgs/msg/Float32` | `float32` | Scalar `(1,)` |
| `std_msgs/msg/Float64` | `float64` | Scalar `(1,)` |
| `std_msgs/msg/String` | `string` | Text `(1,)` |
| `std_msgs/msg/Bool` | `bool` | Boolean `(1,)` |
| `std_msgs/msg/Float64MultiArray` | `float64` | Vector `(N,)` |

You can override the auto-detected dtype with the `dtype` field in the contract, or use a [custom decoder](#custom-encodersdecoders-experimental) for non-standard message types.

### Policy Feature Compatibility

Each LeRobot policy implements its own `validate_features()` and accesses batch keys differently. There is no single enforced schema; what keys a policy accepts depends on the policy. This table summarizes the actual requirements based on the modeling code in `lerobot/src/lerobot/policies/`:

| Feature | ACT | SmolVLA | Pi0 | Pi0-Fast | Pi0.5 | GR00T N1.5 | Wall-X | X-VLA |
|---------|:---:|:-------:|:---:|:--------:|:-----:|:----------:|:------:|:-----:|
| **Type** | BC | VLA | VLA | VLA | VLA | VLA | VLA | VLA |
| **`observation.state`** | optional | **required** | optional | - | - | optional | **required** | optional |
| **`observation.environment_state`** | optional | - | - | - | - | - | - | - |
| **`observation.images.*`** | multi | multi | multi | multi | multi | multi | multi | multi |
| **`task` string** | - | **required** | **required** | **required** | **required** | **required** | **required** | **required** |
| **`action`** | **required** | **required** | **required** | **required** | **required** | **required** | **required** | **required** |
| **VLM backbone (params)** | - | SmolVLM2 (0.5B) | PaliGemma (3B / 0.7B) | PaliGemma (3B) | PaliGemma (3B / 0.7B) | Eagle 2.5 (3B) | Qwen 2.5-VL (7B) | Florence2 (0.7B / 0.2B) |
| **RTC support** | - | yes | yes | yes | yes | - | - | - |
| **Max state dim** | any | 32 | 32 | 32 | - | 64 | 20 | 32 |
| **Max action dim** | any | 32 | 32 | 32 | 32 | 32 | 20 | 20 |
| **Image size** | any | 512×512 | 224×224 | 224×224 | 224×224 | 224×224 | any | any |
| **Max language tokens** | - | 48 | 48 | 200 | 48 | 4096 | 768 | 64 |
| **Chunk size (default) [max]** | (100) | (50) | (50) | (50) | (50) | (50) [1024] | (32) | (32) [512] |
| **Async inference** | yes | yes | yes | - | yes | yes | - | - |



**Key dimensions:**

- **Max images**: All "multi" policies dynamically handle N cameras, configured at init time. However, no policy has truly unlimited image capacity. ACT concatenates image features, so the practical limit depends on the model's hidden dimension. VLA policies (Pi0 family, SmolVLA, Wall-X) feed images through a VLM, so the number of images is constrained by the VLM context window. For most robotics setups (2-3 cameras), this is probably not a bottleneck.
- **Max language tokens**: Maximum number of tokens the policy's tokenizer will keep from your task string. Longer prompts get truncated.
- **Chunk size**: Number of future action steps the policy predicts per inference call. Larger chunks mean fewer inference calls but less reactivity. Most policies build architecture (positional embeddings, pre-allocated tensors) to match the configured `chunk_size` at init time.
- **RTC (Real-Time Chunking)**: An [inference wrapper](https://huggingface.co/docs/lerobot/rtc) that improves real-time performance by overlapping action chunks with continuous re-planning. Only works with flow-matching policies (Pi0 family + SmolVLA).
- **Async inference**: Whether the policy is in LeRobot's gRPC-based asynchronous inference server allowlist (`SUPPORTED_POLICIES` in `async_inference/constants.py`). [Async](https://huggingface.co/docs/lerobot/rtc) decouples observation collection from action computation, which is useful for high-frequency control loops. Pi0-Fast, Wall-X, and X-VLA all implement `predict_action_chunk()` and are technically compatible, but haven't been added to the allowlist yet.

**VLA language pipeline**: All VLA policies require a `task` string (e.g., `"pick up the red block"`). In Rosetta, this comes from the `prompt` argument when recording or running a policy. The string gets tokenized into tensors automatically, either by LeRobot's `TokenizerProcessorStep` (a pipeline step that runs before the policy sees the data) or by the policy itself internally. From a Rosetta/ROS2 perspective, **you just provide the task prompt**.

**Subtask support**: LeRobot provides a `lerobot-annotate` [tool](https://huggingface.co/spaces/lerobot/annotate) for adding subtask annotations to recorded episodes (e.g., marking "reach for object", "grasp", "lift" within a longer task). These annotations are stored as `language_instruction` columns in the dataset. However, **no current action policy consumes subtask annotations**. They are used by [SARM](https://huggingface.co/docs/lerobot/sarm) (a reward model) to compute progress scores for [RA-BC](https://huggingface.co/docs/lerobot/sarm) weighted training of Pi0, Pi0.5, and SmolVLA.

#### What this means for your contract

The keys you define in your Rosetta contract determine which policies you can train with. Some practical guidance:

**Maximum compatibility**: if you want your dataset to work with the widest range of policies:

```yaml
observations:
  - key: observation.state          # Required by: SmolVLA, Wall-X
    topic: /joint_states
    type: sensor_msgs/msg/JointState
    selector: { names: [...] }

  - key: observation.images.top     # At least 1 image required by most policies
    topic: /camera/image_raw/compressed
    type: sensor_msgs/msg/CompressedImage
    image: { resize: [480, 640] }

actions:
  - key: action                     # Required by all action policies
    publish:
      topic: /joint_commands
      type: sensor_msgs/msg/JointState
    selector: { names: [...] }

# For VLA policies, also provide a task prompt when recording:
# ros2 action send_goal ... "{prompt: 'pick up the red block'}"
```

**For VLA fine-tuning**: add a second camera and ensure your recording prompts are descriptive:

```yaml
observations:
  # ... state and first camera as above ...

  - key: observation.images.wrist.right
    topic: /wrist_camera/image_raw/compressed
    type: sensor_msgs/msg/CompressedImage
    image: { resize: [512, 512] }
```


## License

Apache-2.0
