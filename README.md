<p align="center">
  <img alt="Rosetta" src="media/rosetta_logo.png" width="100%">
</p>
<!-- <p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/ROS2-Jazzy-blue" alt="ROS2">
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python 3.10+">
</p> -->

**Rosetta** connects your pub/sub robot to robot-learning frameworks like [LeRobot](https://github.com/huggingface/lerobot). A YAML contract defines the mapping between topics and training frames. The same contract produces training data and drives live inference, eliminating structural train-serve skew.

## Table of Contents

- [Updates and Breaking Changes](#updates-and-breaking-changes)
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
  - [Operators](#operators)
  - [Teleop](#teleop)
  - [Tasks, Rewards, and Signals](#tasks-rewards-and-signals)
  - [Adjunct Topics](#adjunct-topics)
  - [Select Syntax](#select-syntax)
  - [Alignment Strategies](#alignment-strategies)
  - [Supported Message Types](#supported-message-types)
  - [Custom Encoders/Decoders](#custom-encodersdecoders)
- [LeRobot Data Model Reference](#lerobot-data-model-reference)
  - [Key System](#key-system)
  - [EnvTransition](#envtransition)
  - [Data Types](#data-types)
  - [Policy Feature Compatibility](#policy-feature-compatibility)
- [License](#license)



## Quick Start

```
  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
  │  DEFINE  │     │  RECORD  │     │ CONVERT  │     │  TRAIN   │     │  DEPLOY  │
  │ Contract │────▶│  Demos   │────▶│ Dataset  │────▶│  Policy  │────▶│ on Robot │
  └──────────┘     └──────────┘     └──────────┘     └──────────┘     └──────────┘
```

[**Define**](#the-contract) a contract mapping your ROS2 topics to [LeRobot](https://github.com/huggingface/lerobot) features, [**record**](#recording-episodes) demos to bag files, [**convert**](#converting-bags-to-datasets) the bags to a LeRobot dataset, [**train**](#training-a-policy) a policy, and [**deploy**](#deploying-policies) the policy back to your robot.

> **Getting started?** The [rosetta_ws](https://github.com/iblnkn/rosetta_ws) devcontainer installs ROS2, Rosetta, and LeRobot together.

**1. Define** a [contract](#the-contract) for your robot:

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
ros2 launch rosetta episode_recorder_launch.py contract_path:=contract.yaml
```

```bash
# Terminal 2: Start an episode. Ctrl-C stops and saves it.
ros2 action send_goal /record_episode \
    rosetta_interfaces/action/RecordEpisode "{prompt: 'pick up the red block'}"
```

Repeat per episode, adjusting the prompt as the task changes. A keyboard controller also exists (see [Recording Episodes](#recording-episodes)).

> **How many episodes?** Plan on **50 to 200+ demonstrations** depending on task complexity. Diverse, high-quality demonstrations produce better policies. For data collection tips, see [Collecting Your Dataset](https://abenstirling.com/lerobot/) and [Improving Your Robotics AI Model](https://docs.phospho.ai/learn/improve-robotics-ai-model).

**3. Convert** bags to LeRobot dataset:

```bash
python -m rosetta.robots.ros2.offline.port \
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
ros2 launch rosetta policy_runner_launch.py \
    contract_path:=contract.yaml \
    pretrained_name_or_path:=my-org/my-policy
```

```bash
# Terminal 2: Run the policy
ros2 action send_goal /run_policy \
    rosetta_interfaces/action/RunPolicy "{prompt: 'pick up red block'}"
```

(The action's relative name is `run_policy`. A launch-file namespace prefixes the full name, e.g. `/robot_policy/run_policy` in the HIL launch.)

---

## Architecture

```
rosetta/
├── contract/    # schema, specs, operators
├── frames/      # layout, resampling, codecs, stream protocols
├── robots/      # robot side: pub/sub ecosystems (ros2/)
└── policies/    # policy side: DatasetWriter + PolicyRunner seams, entry-point loading
```


- A **channel** is a declared endpoint in the robot interface's dialect (`schema.Channel`).
- A **stream** is the decoded, timelined sample sequence of one channel *before* alignment.
- A **frame** is one synchronized sample of every contract key per clock tick, *after* alignment.


The workspace splits into several packages. Framework adapters register into `rosetta.policies` entry points:

| Package | Purpose |
|---------|---------|
| `rosetta` | Core library, nodes, bag conversion |
| [`rosetta_interfaces`](https://github.com/iblnkn/rosetta_interfaces) | ROS2 action/service definitions |
| [`lerobot_rosetta`](https://github.com/iblnkn/lerobot-rosetta) | LeRobot backend adapter: dataset writer, policy runner, inference servers |
| [`lerobot_robot_rosetta`](https://github.com/iblnkn/lerobot-robot-rosetta) | LeRobot Robot plugin (discovered by LeRobot) |
| [`lerobot_teleoperator_rosetta`](https://github.com/iblnkn/lerobot-teleoperator-rosetta) | LeRobot Teleoperator plugin (experimental) |

```
rosetta/
├── launch/
│   ├── episode_keyboard_launch.py
│   ├── episode_recorder_launch.py
│   ├── hil_launch.py
│   └── policy_runner_launch.py
└── params/
    ├── episode_recorder.yaml    # Default config for Episode Recorder
    ├── hil_manager.yaml         # HIL launch "super YAML" (all four nodes)
    └── policy_runner.yaml       # Default config for the policy runner
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

**Important:** `lerobot_robot_rosetta` creates a ROS2 lifecycle node internally, so **your system needs ROS2 installed**, even when you invoke the plugin through LeRobot's standard CLI tools. When `policy_runner_node` launches inference, the chain is: `policy_runner_node` (ROS2 node) → LeRobot `RobotClient` → `lerobot_robot_rosetta` (also a ROS2 node) → your robot's ROS2 topics. Both the convenience node and the robot plugin are ROS2 nodes in the same ROS2 graph.

Any ROS2 robot works with LeRobot's tools this way. Define a contract and use `--robot.type=rosetta`.

### ROS2 Lifecycle Integration

LeRobot's `connect()` / `disconnect()` map to ROS2 lifecycle transitions:

| LeRobot Method | Lifecycle Transition | Effect |
|----------------|---------------------|--------|
| - | `configure` | Create subscriptions (start buffering), create publishers (disabled) |
| `connect()` | `activate` | Enable publishers, start watchdog |
| `disconnect()` | `deactivate` → `cleanup` | Safety action, disable publishers, destroy resources |

### Policy Inference

The `policy_runner_node` delegates inference to a gRPC policy server (`lerobot_rosetta.policy_server`, a thin preload/cache wrapper over LeRobot's `lerobot.async_inference.policy_server`). The server has no ROS2 dependency and runs on any machine with LeRobot and a GPU. Benefits:

- Better GPU memory management
- Support for all LeRobot policy types without code changes
- Consistent behavior between training and deployment
- Runs on a remote machine, so a resource-constrained robot offloads inference over the network

When `launch_local_server` is `true`, the node starts the server and fully loads the model at **configure** time, so the first `run_policy` goal costs the same as any other. The configure transition blocks until the model is up (bounded by `server_startup_timeout_sec`), GPU memory is held from startup, and later goals reuse the loaded model instead of re-reading the checkpoint on every handshake.

### rosetta_ws Workspace

[rosetta_ws](https://github.com/iblnkn/rosetta_ws) is a devcontainer workspace for getting started. Installing ROS2 and LeRobot together is not trivial. The workspace handles this setup.

---

## The Contract

The contract defines the translation between ROS 2 topics and the keys LeRobot expects.

On the ROS2 side, data lives in typed messages on named topics with rich structure (headers, arrays, nested fields). On the LeRobot side, data lives in flat dictionaries with dot-separated string keys and numpy/tensor values. The contract maps one to the other: type conversion, field extraction, timestamp alignment, resampling.

Every frame-clock entry reads as one pipeline. `channel` provides, `align` chooses a timeline (mandatory), then `select`, then `apply`, then the mapping key:

```yaml
observation.state:
  channel: {topic: /follower_arm/joint_states, type: sensor_msgs/msg/JointState}
  align: {strategy: hold, timeline: header}
  select: [position.shoulder_pan, position.shoulder_lift, position.elbow,
           position.wrist_pitch, position.wrist_roll, position.wrist_yaw]
```

At each timestep, this entry:
1. **Subscribes** to `/follower_arm/joint_states` (a `JointState` message). The channel block is exactly what a different pub/sub ecosystem would replace.
2. **Aligns** the stream onto the frame clock using the message's `header` timeline, holding the latest value.
3. **Extracts** the named fields using dot notation (`position.shoulder_pan` → `msg.position[msg.name.index("shoulder_pan")]`).
4. **Assembles** a numpy array: `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]` (dtype `float64`).
5. **Stores** the array under the key `observation.state` in the LeRobot dataset.

**Multi-source concatenation**: a **list** value under one key declares ordered sources whose values concatenate into a single feature vector:

```yaml
observations:
  observation.state:
    - channel: {topic: /arm/joint_states, type: sensor_msgs/msg/JointState}
      align: {strategy: hold, timeline: header}
      select: [position.j1, position.j2, position.j3]
    - channel: {topic: /gripper/state, type: std_msgs/msg/Float32}
      align: {strategy: hold, timeline: receive}
      # Result: observation.state = [j1, j2, j3, gripper] (4D vector)
```
This matters because core policies expect specific key names (see [Policy Feature Compatibility](#policy-feature-compatibility)). To feed several ROS2 topics into one observation, declare them as ordered sources of one key.

The same topic also works in several sources (e.g. position and orientation slices of one pose topic). Each source keeps its own selector and buffer.

Multi-source keys are validated at load (`ContractValidationError` otherwise):

- Every source of a multi-source key needs a `select`. Concatenation needs static dims to lay out the combined vector.
- All sources of a key must resolve to the **same** dtype. Set `dtype:` in the channel explicitly to align them.
- Images and strings never share a key. Give each an own key. Image features never concatenate.

Layout follows declaration order. There is no separate ordering key (e.g. `position: 2`). A YAML list already expresses order: any position for any source, reordered with a one-line diff. An explicit ordering field would duplicate the same information in a second place and let the two drift apart.

A minimal contract typically only needs `observations` and `actions`. See the full [Contract Reference](#contract-reference) for all options, and the [LeRobot Data Model Reference](#lerobot-data-model-reference) for how keys, features, and policies interact.

---

## Recording Episodes

The `episode_recorder_node` is a convenience node recording contract-specified topics to [rosbag2](https://github.com/ros2/rosbag2) files. The node reads the contract to determine which topics to subscribe to, then starts and stops recording via ROS2 actions, with feedback on duration and message count.

**This node is not the only way to record compatible bags.** Any valid rosbag2 file containing the contract's topics works: `ros2 bag record`, custom `rosbag2_py` scripts, third-party recording tools. Bags you already have port fine. See [Bring Your Own Bags](#bring-your-own-bags). The `episode_recorder_node` adds convenience: define your topics once in the contract, and the node handles subscription setup, bag lifecycle, and action-based control.

> Both Rosetta nodes use parameter files (`params/`) as defaults. All parameters are also exposed as launch arguments, which override the defaults. Run `ros2 launch rosetta <launch_file> --show-args` to see all options.

```bash
ros2 launch rosetta episode_recorder_launch.py contract_path:=/path/to/contract.yaml
```

### Controlling Recording

#### Option A: Keyboard controller (recommended)

The `episode_keyboard_node` starts, stops, and discards episodes with single key presses. Run the node in a **second terminal** while the recorder runs:

```bash
ros2 run rosetta episode_keyboard_node
```

Or via launch (supports optional arguments):

```bash
ros2 launch rosetta episode_keyboard_launch.py \
    default_prompt:="pick up the cube"
```

| Key | Action |
|-----|--------|
| `r` / `→` | Start recording |
| `s` / `←` | Stop and save |
| `d` / `⌫` | Discard episode (stop + delete bag) |
| `t` | Edit task prompt for the next episode |
| `h` / `?` | Help |
| `q` | Quit |

Launch arguments for `episode_keyboard_launch.py`:

| Argument | Default | Description |
|----------|---------|-------------|
| `recorder_ns` | `/episode_recorder` | Namespace of the recorder node |
| `default_prompt` | `` | Initial task prompt used when starting recordings |

#### Option B: ROS2 action

For scripted or automated workflows, trigger recording directly via the action interface:

```bash
ros2 action send_goal /record_episode \
    rosetta_interfaces/action/RecordEpisode "{prompt: 'task description'}"
```

Stop by sending `Ctrl-C` to the `send_goal` command, or via the cancel service:

```bash
ros2 service call /episode_recorder/cancel_recording std_srvs/srv/Trigger
```

The recorder also serves `~/start_recording`
(`rosetta_interfaces/srv/StartRecording`) for callers without action support,
and `~/delete_last_bag` (`std_srvs/srv/Trigger`) to remove the most recently
saved bag.

**Parameters** (all available as launch arguments):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `contract_path` | `contracts/so_101.yaml` | Path to contract YAML (launch-arg default, required at the node level) |
| `bag_base_dir` | `datasets/bags` | Rosbag output dir. Relative paths resolve against the launch cwd (like `ros2 bag record`) |
| `storage_id` | `mcap` | Rosbag format: `mcap` (recommended) or `sqlite3` |
| `default_max_duration` | `0.0` | Max episode duration in seconds. `0.0` records until stopped |
| `feedback_rate_hz` | `2.0` | Recording feedback publish rate |
| `record_all` | `true` | Record every topic on the graph, not only contract topics |
| `exclude_topics` | `[]` | Regex list of topics to skip when `record_all` is on |
| `embed_contract` | `true` | Embed the exact contract text into bag metadata |
| `log_level` | `info` | Logging level: `debug`, `info`, `warn`, `error` (launch argument) |
| `configure` | `true` | Auto-configure on startup (launch argument) |
| `activate` | `true` | Auto-activate on startup (launch argument) |

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

Recording is the expensive step. Robot time, operator time, hardware wear. Every decision baked into a recording is permanent, so Rosetta bakes in nothing: bags store raw messages, and the contract assigns meaning at port time. Recorders bound to a training format fix fps, keys, and image size the moment you press record. Deferring buys you:

- **Preserves raw data.** Bags store every message at original rate and timestamp. No alignment, no downsampling, no loss. Change the contract and reprocess without re-recording. Format churn stops mattering: LeRobot's dataset format has moved from v1 to v2 to v3, and bags re-port to each. The dataset is disposable. The bags are the asset.
- **Familiar to ROS2 users.** Bag files are the standard data format in the ROS2 ecosystem, with mature tooling for [recording, playback, inspection](https://docs.ros.org/en/jazzy/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html), and analysis. Any bag-file tool works with your recorded data.
- **Stores data beyond what LeRobot needs.** Bags include topics with no LeRobot feature mapping: diagnostics, TF trees, debug streams, extra sensors. This data stays available for analysis, debugging, or future use even outside the training dataset.
- **Leverages MCAP.** Rosetta defaults to [MCAP](https://mcap.dev/) storage, which provides [high-performance](https://mcap.dev/guides/benchmarks/rosbag2-storage-plugins) random-access reads, efficient compression, and broad ecosystem support beyond ROS2.
- **Write-optimized for live recording.** Bag files (especially MCAP) are designed for high-throughput sequential writes with minimal overhead, well-suited for capturing live sensor data. LeRobot datasets (Parquet + MP4) are read-optimized for training but involve more overhead when writing live, including in-memory buffering and post-episode video encoding.


## Converting Bags to Datasets

The porter (`rosetta_port`, i.e. `python -m rosetta.robots.ros2.offline.port`) converts rosbag2 files to LeRobot datasets using the contract for key mapping, timestamp alignment, resampling, and dtype conversion. The porter applies the same `StreamBuffer` resampling logic as live inference, so your offline dataset matches what the robot sees at runtime.

The primitives in `rosetta.contract` / `rosetta.frames` (contract loader, stream buffers) and `rosetta.robots.ros2` (decoders) support custom conversion scripts. The porter handles the full pipeline: reading bags, applying the contract, encoding video, building the LeRobot dataset structure, and optionally pushing to the Hub. The raw bag preserves all data without transformation, so a re-run with an updated contract (changed keys, adjusted `fps`, added or removed features) needs no re-recording.

### Bring Your Own Bags

The porter consumes standard rosbag2. Bags from `ros2 bag record`, `rosbag2_py` scripts, third-party tools, or an old archive all work. The recorder, keyboard, and HIL nodes are conveniences, not requirements. Write a contract for the topics your bags contain and port them.

What the porter expects:

- **One bag directory = one episode.** The porter finds bags by `metadata.yaml` and searches `--raw-dir` recursively. Split long multi-episode recordings into per-episode bags first.
- **Observation topics must be present.** A bag missing one fails the warmup gate. Missing action, reward, and signal topics zero-fill, same as the live bridge.
- **`align.timeline: header` needs stamped messages.** Unstamped messages on a header timeline drop at ingest, same as live.
- **Task labels need a source.** The porter reads per-frame tasks from a `tasks:` topic, falling back to the `lerobot.operator_prompt` field in bag metadata (the episode recorder writes this). Foreign bags have neither and produce an empty task string. Declare a task topic or write the metadata field before training VLA policies.
- **No embedded contract, no problem.** The mismatch warning only runs when a bag carries one. `--contract` defines the translation, and the output dataset still embeds the contract for deployment-time resolution.


### Relationship to LeRobot

The porter mirrors the interface of LeRobot's example porters (like `port_droid.py`):

```bash
# LeRobot's port_droid.py
python examples/port_datasets/port_droid.py \
    --raw-dir /data/droid/1.0.1 \
    --repo-id my_org/droid \
    --push-to-hub

# Rosetta's rosetta.robots.ros2.offline.port (same pattern + contract)
python -m rosetta.robots.ros2.offline.port \
    --raw-dir ./datasets/bags \
    --contract contract.yaml \
    --repo-id my_org/my_dataset \
    --root ./datasets/lerobot
```

**Rosetta-specific additions:**

| Argument | Description |
|----------|-------------|
| `--contract` | **(Required)** Rosetta contract YAML that defines ROS2 topic → LeRobot feature mapping |
| `--framework` | Dataset writer to use: `lerobot` (default), `vla_foundry`, `starvla` |
| `--root` | Override output directory (LeRobot defaults to `~/.cache/huggingface/lerobot`) |
| `--repo-id` | Dataset repo ID. Defaults to the `--raw-dir` directory name |
| `--vcodec` | Video codec selection (default `libsvtav1`, not in base LeRobot porters) |
| `--num-shards` / `--shard-index` | Split a directory of bags across parallel porter invocations |
| `--no-embed-contract` | Skip writing the contract into the dataset as `meta/rosetta_contract.yaml` |
| `--hub-public` / `--hub-tags` | With `--push-to-hub`: make the repo public (private by default) / set tags (default `rosetta,rosbag`) |
| `--past-steps`, `--future-steps`, `--image-indices`, `--samples-per-shard` | `vla_foundry` windowing and sharding options |

The porter embeds the contract into the dataset (`meta/rosetta_contract.yaml`)
by default, so the dataset carries the exact translation behind the data.
Inference resolves the contract from there when no `contract_path` is given
(see [Deploying Policies](#deploying-policies)). When a bag carries an
embedded contract (recorder default), the porter compares the bag's contract
against `--contract` and warns on a semantic mismatch. `--contract` stays
authoritative.

### Basic Usage

```bash
python -m rosetta.robots.ros2.offline.port \
    --raw-dir ./datasets/bags \
    --contract ./contract.yaml \
    --repo-id my_dataset \
    --root ./datasets/lerobot
```

For large-scale conversions, parallel processing, and SLURM cluster workflows, see the **[LeRobot Porting Datasets Guide](https://huggingface.co/docs/lerobot/en/porting_datasets_v3)** and substitute `rosetta_port` for `port_droid.py` in the examples.



## Training a Policy

After converting your bags to a LeRobot dataset, [train a policy](https://huggingface.co/docs/lerobot/il_robots#train-a-policy) with `lerobot-train`.


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

The `policy_runner_node` is a convenience node wrapping LeRobot's inference pipeline in ROS2 actions. Start and stop policy execution via `ros2 action send_goal`, with feedback on inference progress. The node launches a local LeRobot gRPC policy server as a subprocess, or connects to a remote one.

Launch Client:

```bash
ros2 launch rosetta policy_runner_launch.py contract_path:=/path/to/contract.yaml
```

Run policy:

```bash
ros2 action send_goal /run_policy \
    rosetta_interfaces/action/RunPolicy "{prompt: 'task description'}"
```

**Remote inference:** When `launch_local_server` is `false`, the node connects to a gRPC policy server at `server_address`. The server has no ROS2 dependency and runs on any machine with a GPU, independent of your robot's ROS2 environment. A resource-constrained robot offloads inference to a remote GPU server this way. To pre-warm the remote server so even the first goal skips the model load:

```bash
python -m lerobot_rosetta.policy_server --host=0.0.0.0 --port=8080 \
    --policy-type=act --pretrained-name-or-path=my-org/my-policy --policy-device=cuda
```

(Stock `lerobot.async_inference.policy_server` also works, but reloads the checkpoint on every goal.)

**Parameters** (all available as launch arguments):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `contract_path` | `contracts/so_101.yaml` | Path to contract YAML. Optional when the checkpoint's training dataset embeds one (see below) |
| `pretrained_name_or_path` | *(see params file)* | HuggingFace model ID or local path |
| `framework` | `lerobot` | Policy framework adapter: `lerobot`, `vla_foundry`, `starvla` |
| `is_classifier` | `false` | Publish the reward section as the action output (HIL reward classifier) |
| `server_address` | `127.0.0.1:8080` | Policy server address |
| `policy_type` | `act` | Policy type: `act`, `smolvla`, `diffusion`, `pi0`, `pi05`, etc. |
| `policy_device` | `cuda` | Inference device: `cuda`, `cpu`, `mps`, or `cuda:0` |
| `actions_per_chunk` | `30` | Actions per inference chunk |
| `chunk_size_threshold` | `0.95` | When to request new chunk (0.0-1.0) |
| `aggregate_fn_name` | `weighted_average` | Chunk aggregation: `weighted_average`, `latest_only`, `average`, `conservative` |
| `feedback_rate_hz` | `2.0` | Execution feedback publish rate |
| `launch_local_server` | `true` | Auto-start policy server subprocess (at configure, with model preload) |
| `server_startup_timeout_sec` | `120.0` | Max wait for the server to come up (covers model preload, raise for cold HF downloads) |
| `obs_similarity_atol` | `-1.0` | Observation filtering tolerance (-1.0 to disable)* |
| `log_level` | `info` | Logging level: `debug`, `info`, `warn`, `error` |
| `configure` | `true` | Auto-configure on startup |
| `activate` | `true` | Auto-activate on startup |

*\*`obs_similarity_atol`: The policy server filters observations as "too similar" when the L2 norm of the state difference falls below the threshold. The default threshold (1.0) assumes joint states change substantially between frames. Robots with smaller movements skip most observations. Set to `-1.0` to disable filtering.*

The params file also documents `sim_time_multiplier` (default `1.0`), which
scales the contract `fps` before it reaches LeRobot's wall-clock control loop
when your simulator runs slower or faster than real time.

**Contract resolution.** When `contract_path` is empty, the node resolves the
contract from the checkpoint: `pretrained_name_or_path` → its
`train_config.json` → the training dataset (local path first, then the Hub) →
the dataset's embedded `meta/rosetta_contract.yaml`. Datasets ported with the
default `embed_contract` settings deploy with no separate contract file. If no
link in the chain resolves, the node errors and asks for `contract_path`.

**Example:**

```bash
# Run with a pretrained model
ros2 launch rosetta policy_runner_launch.py \
    contract_path:=/path/to/contract.yaml \
    pretrained_name_or_path:=my-org/my-policy
```

### Human-in-the-Loop (HIL)

`hil_launch.py` wires four nodes for HIL workflows: `hil_manager_node`, a
policy runner (namespace `robot_policy`), an optional reward classifier (a
second policy runner with `is_classifier: true`, namespace
`reward_classifier`), and the episode recorder. The manager runs an episode
end to end: start recording, run the policy, mux teleop intervention onto the
action topics (see [Teleop](#teleop)), apply reward overrides. Control comes
through the `manage_episode` action (`rosetta_interfaces/action/ManageEpisode`)
and services (`~/start_episode`, `~/stop_episode`, `~/set_intervention`,
`~/set_reward_override`, `~/clear_reward_override`). Configuration lives in
`params/hil_manager.yaml`, the "super YAML" covering all four nodes.

**This node is not the only way to deploy.** LeRobot's standard CLI tools run inference directly with the Rosetta robot plugin:

```bash
# Standard LeRobot deployment, no policy_runner_node needed
lerobot-record --robot.type=rosetta --robot.config_path=contract.yaml
```

The `lerobot_robot_rosetta` / `lerobot_teleoperator_rosetta` distributions
follow LeRobot's third-party plugin naming convention (`lerobot_robot_*`,
`lerobot_teleoperator_*`). LeRobot CLIs and the async robot client
auto-discover them when installed. No manual import or registration step.

See [Imitation Learning on Real Robots](https://huggingface.co/docs/lerobot/il_robots) for LeRobot's native deployment workflow. The `policy_runner_node` adds ROS2 action-based lifecycle management on top, useful when your workflow is already ROS2-centric.

---

## Contract Reference

A contract is a YAML file mapping ROS2 topics to LeRobot's observation/action interface. The contract covers the full LeRobot `EnvTransition` interface:

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
robot_interface: ros2
fps: 30

observations:
  observation.state:
    channel: {topic: /joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: header}
    select: [position.j1, position.j2]

actions:
  action:
    channel: {topic: /joint_commands, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: header}
    select: [position.j1, position.j2]
```

`robot_type`, `robot_interface` (only `ros2` today), and `fps` are required.
Top-level keys starting with `x-` are ignored. Use them to hold shared YAML
anchors, e.g. an `x-qos:` block of reusable QoS profiles (see
`contracts/stone.yaml`).

### Observations

```yaml
observations:
  # State vector (with all optional fields shown)
  observation.state:
    channel:
      topic: /joint_states
      type: sensor_msgs/msg/JointState
      qos: {reliability: best_effort, depth: 10}
      dtype: float64            # optional; defaults to the codec's native dtype
    align:
      strategy: hold            # hold | asof | drop — mandatory, no default
      timeline: header          # a timeline the channel provides — mandatory
    select: [position.j1, velocity.j1]
    apply: [rad2deg]            # optional operator pipeline (see Operators)
```

Data on a channel carries several timestamps at once. The robot interface
produces them as named **timelines**, and `align.timeline` selects one by
name. Every ros2 channel provides `receive` (arrival time at the node). A
message type carrying a std_msgs `Header` also provides `header` (sensor
time, more accurate, but publishers must stamp correctly and hosts must be
time-synced). Naming a timeline the channel does not provide is a load-time
error. A header-timeline message arriving unstamped is dropped at ingest,
never silently re-timed.

`align.strategy` picks how samples land on the frame clock. `hold` carries
the last value forward. `asof` holds only within `tolerance_ms` (required,
and only valid, with `asof`). `drop` gaps out anything older than one frame.

```yaml

  # Camera (resize is an operator in the apply pipeline; encoding hints live in the channel)
  observation.images.camera:
    channel: {topic: /camera/image_raw/compressed,
              type: sensor_msgs/msg/CompressedImage}
    align: {strategy: hold, timeline: header}
    apply: [resize: [224, 224]]  # [height, width]
```

A list value under one key declares ordered sources whose values are
concatenated (see [The Contract](#the-contract)).

### Actions

```yaml
actions:
  action:
    channel:
      topic: /joint_commands
      type: sensor_msgs/msg/JointState
      qos: {reliability: reliable, depth: 10}
      safety: hold              # none (default) | hold | zeros
    align: {strategy: hold, timeline: header}
    select: [position.j1, position.j2]
    apply: [rad2deg]            # only serveable operators allowed on actions
```

`channel.safety` is the stop behavior published by the watchdog and on
deactivate. The default is `none` because a fabricated zero command is only a
safe stop under velocity control. Under position control (the common case),
zero commands a slam to the zero pose. Opt in explicitly per channel: `zeros`
for velocity-controlled robots (e.g. a Twist base), `hold` where re-sending
the last command is safe. `zeros` means zeros in *action space*: the zero
vector runs through the inverse `apply` pipeline before encoding, so a
`clamp` still bounds the published command.

Actions read the same pipeline right-to-left: recording decodes from the
channel, serving encodes to it. A list value splits one action vector across
channels in order (see `contracts/stone.yaml`), each with its own safety
behavior and align.

### Field kinds (`kind`)

`kind` is an optional tag on an observation or action source naming the
value's representation. LeRobot ignores the tag. The vla_foundry / starVLA
adapters use the tag to pick per-group normalization and rotation handling.
The default leaves every existing contract unchanged.

`kind` is a single token (default `continuous`):

| value | dims | meaning |
|---|---|---|
| `continuous` (default) | any | plain scalar/vector (joints, positions, velocities) |
| `quaternion` | 4 | rotation `[x, y, z, w]` |
| `euler_rpy` | 3 | roll / pitch / yaw |
| `axis_angle` | 3 | rotation vector |
| `rotation_6d` | 6 | 6-D continuous rotation |
| `binary` | any | discrete on/off (e.g. gripper) |

**Keys stay canonical.** Don't encode the type in the key. `action.binary`
becomes a separate LeRobot feature and breaks policies reading `action`.
Keep the canonical key and split a mixed vector into one spec per kind. The
specs share the key and concatenate into one flat feature, each with its own
`kind`:

```yaml
observations:
  observation.state:                # canonical key; one flat vector to LeRobot
    - channel: {topic: /ee_pose, type: geometry_msgs/msg/PoseStamped}
      align: {strategy: hold, timeline: header}
      select: [pose.position.x, pose.position.y, pose.position.z]
      # kind: continuous (default)
    - channel: {topic: /ee_pose, type: geometry_msgs/msg/PoseStamped}
      align: {strategy: hold, timeline: header}
      select: [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
      kind: quaternion
    - channel: {topic: /gripper, type: sensor_msgs/msg/JointState}
      align: {strategy: hold, timeline: header}
      select: [position.gripper]
      kind: binary
```

`kind` is per-source, so a mixed vector carries one tag per slice.
Validation runs at contract load: the dim count must match the `kind`
(`quaternion` is 4, etc.), and an untagged `x/y/z/w` run warns ("looks like a
quaternion, set `kind: quaternion`") so a rotation is never silently
min-max normalized.

### Backend notes (LeRobot / vla_foundry / starVLA)

One contract drives all three frameworks. Only `kind` is VLA-specific
(LeRobot ignores the tag). A few differences matter:

- **Training is framework-native for all three.** Rosetta produces the dataset
  and hosts deploy. You train with each framework's own tools: LeRobot with
  `lerobot-train` on the dataset, vla_foundry with its trainer on the tar shards,
  starVLA from the starVLA repo after pasting the generated `rosetta_dataconfig.py`.
- **Deploy topology.** LeRobot runs a gRPC policy-server subprocess. vla_foundry
  loads the model in-process in the ROS node (heavy on the robot). starVLA is a
  websocket client to a separate server process (GPU box, possibly remote).
- **State/action layout.** LeRobot keeps one feature per contract key, so
  multi-key state like `observation.state` + `observation.environment_state` stays
  separate. The VLA writers concatenate all state into one `observation.state`
  column (starVLA preserves the per-`kind` slices via `meta/modality.json`). A
  model needing separate state inputs loses the split on the VLA path.
- **Sections consumed.** The VLA writers use observations and actions only. The
  `rewards` / `signals` / `info` / `complementary_data` / `adjunct` sections
  (LeRobot RL and record-only) are not part of the VLA dataset.
- **`fps` consistency.** `fps` sets the control rate and action time-horizon. Keep
  it identical across record, train, and deploy or the policy degrades.

### Train/deploy skew

Rosetta matches training input to live input. Offline conversion
(`bag_frames`) and online inference (`topic_bridge`) run the same
`StreamBuffer` resampling, `aggregate_frame`, and operator pipeline from the
same contract, so decode, `select`, `apply` operators, alignment, and key
aggregation match across all three frameworks.

Downstream, per framework:

- **LeRobot.** Normalization is policy-internal, with the same dataset stats at
  train and infer time. The only residual is the video codec (mp4 dataset vs raw
  online), inherent to any video dataset.
- **vla_foundry.** Inference reuses the training `RoboticsProcessor` and
  normalizer, so input and output normalization match training. Make sure the
  online image-history matches the writer's `image_indices` windowing.
- **starVLA.** The server un-normalizes the output. The stock server leaves
  input-state normalization to the client. The Rosetta runner stays starVLA-free
  and sends raw state, so a state-conditioned model would see skew. Use
  `scripts/rosetta_starvla_server.py` (the default `Dockerfile.starvla` entrypoint),
  which normalizes input state server-side with the training transform.
  Vision+language-only models send no state and are unaffected. Also keep the
  runner's `image_width/height` equal to the model's training `obs_image_size`.

### Operators

`apply` is an ordered operator pipeline run after `select` (field projection).
On the **record/decode** path operators run front-to-back via their forward
direction. On the **serve/encode** path (policy command → ROS) operators run
back-to-front via their inverse. Select and apply are pure per-message
transforms, so they commute with alignment. The runtime applies them once per
message at ingest, observationally identical to the contract's
channel → align → select → apply reading.

Each operator declares an **invertibility tier** governing where the operator runs:

| Tier | Meaning | On actions? | Round-trip gate |
|------|---------|:-----------:|:---------------:|
| `FORWARD_ONLY` | decode/build only, lossy and one-way | rejected at load | — |
| `BIDIRECTIONAL` | runs both ways but lossy (applies a bound) | allowed | — |
| `BIJECTIVE` | inverse exactly undoes forward | allowed | verified at load |

An action's `apply` accepts only serveable operators (`BIDIRECTIONAL` or
`BIJECTIVE`). A `FORWARD_ONLY` operator on an action is rejected at contract
load. A `BIJECTIVE` operator is round-trip verified at load: the contract
refuses to load unless `inverse(forward(x)) == x`, so a wrong inverse fails
at load instead of corrupting actions silently.

| Operator | Form | Tier | Notes |
|----|------|------|-------|
| `rad2deg` | `rad2deg` | `BIJECTIVE` | radians (ROS) ↔ degrees (dataset) |
| `clamp` | `clamp: {min: lo, max: hi}` | `BIDIRECTIONAL` | clip element-wise, preserving the input dtype. On actions, bounds the outgoing command (lossy, so not bijective) |
| `resize` | `resize: [h, w]` | `FORWARD_ONLY` | nearest-neighbor image resize (integers, ≤ 8192). Image observations only. Declares the stream's output geometry |

```yaml
apply: [rad2deg, clamp: {min: -180, max: 180}]   # convert then bound
apply: [resize: [224, 224]]                      # image resize (observations only)
```

Operators bound values. They do not repair them. `clamp` passes NaN through
(`np.clip` semantics), so the encode path refuses non-finite values: a
NaN/Inf command from a diverged policy is never published. The frame drops
whole (no partial frame across a multi-channel action) with a throttled
error, and if the condition persists the watchdog applies each channel's
declared `safety` behavior. A recovered policy resumes seamlessly.

Add a built-in capability by registering a new operator in
`rosetta/contract/builtin_operators.py`. The framework
(`rosetta/contract/operators.py`) and the contract schema do not change:

```python
from rosetta.contract.operators import register_operator, Operator, Invertibility

@register_operator("my_op", kind=Invertibility.BIJECTIVE)
class MyOperator(Operator):
    def forward(self, arr): ...
    def inverse(self, arr): ...   # round-trip verified at contract load
```

Registration enforces the tier's promises up front. A serveable tier
(`BIDIRECTIONAL`/`BIJECTIVE`) without an `inverse` is rejected at import. A
name already registered raises unless you pass `override=True` (same rule as
the codec registries), so a plugin never silently shadows a built-in like
`clamp`. An operator fixing image geometry declares the output by setting
`self.output_hw = (h, w)`. Image observations require some operator in their
pipeline to declare geometry (built-in `resize` does).

**Bring your own operator as a plugin**, no fork needed. Package your operator
and advertise the module under the `rosetta.operators` entry-point group.
Rosetta discovers and imports the module at contract load, so the contract
references the operator by name only (no module paths in the YAML):

```toml
# in your plugin's pyproject.toml
[project.entry-points."rosetta.operators"]
my_operators = "my_pkg.my_operators"   # imported once; its @register_operator calls run
```
```yaml
apply: [my_op]             # resolves to the plugin's registered operator
```

### Teleop

For human-in-the-loop recording with a leader arm or other input device.
Teleop uses fixed **role** sections (`input`, `events`, `feedback`) instead
of frame keys. `input`/`feedback` are each a *list* of independently-targeted
sources, not a single block. Every entry names a topic from the
`actions`/`observations` sections (`target`/`origin`), validated at load, so
a contract teleops one action, several, or none, without touching the
`actions`/`observations` sections themselves:

```yaml
teleop:
  input:
    - target: /arm/joint_commands   # names an existing action channel's topic
      channel: {topic: /leader_arm/joint_states, type: sensor_msgs/msg/JointState}
      align: {strategy: hold, timeline: header}
      select: [position.j1, position.j2]

  events:                 # edge-triggered; no align — events are not resampled
    channel: {topic: /joy, type: sensor_msgs/msg/Joy}
    select:               # event_name -> button/axis path
      is_intervention: buttons.5
      success: buttons.0
      end_success: buttons.6
      end_failure: buttons.7
      failure: buttons.1

  # Event vocabulary is closed: is_intervention, start_episode, success,
  # failure, end_success, end_failure. An unknown event name is a load error.

  feedback:
    - origin: /arm/joint_states      # names an existing observation channel's topic
      channel: {topic: /leader_arm/effort_feedback, type: sensor_msgs/msg/JointState}
      align: {strategy: hold, timeline: receive}
      select: [effort.j1, effort.j2]
```

`hil_manager_node` drives each `input` entry live: the node decodes the
teleop message and encodes+publishes the value onto `target` with the same
decode/encode machinery as everything else in the contract, not a raw
byte-for-byte republish. A leader device with different fields, units, or
timeline than the action still works. The action's own topic already carries
the human's command during teleop. The porter additionally exposes
`input`/`feedback` as their own `teleop.input.<action key>` /
`teleop.feedback.<observation key>` dataset columns, for diagnostics.
Feedback runs the mirror direction (observation → encode → publish to the
human device) regardless of mux state. Feedback channels never declare
`safety`: a teleop device gets no fabricated commands, and declaring
`safety` on feedback is a load error. See `contracts/stone.yaml` for a full
worked example.

### Tasks, Rewards, and Signals

These sections are optional. Use them when your workflow requires task prompts from ROS2 topics, RL reward signals, or episode termination signals.

```yaml
tasks:
  task:                       # not on the frame clock — no align
    channel: {topic: /task_prompt, type: std_msgs/msg/String}

rewards:
  next.reward:                # extended sections: dtype is mandatory
    channel: {topic: /reward, type: std_msgs/msg/Float64, dtype: float64}
    align: {strategy: hold, timeline: receive}

signals:
  next.done:
    channel: {topic: /episode_done, type: std_msgs/msg/Bool, dtype: bool}
    align: {strategy: hold, timeline: receive}

  next.truncated:
    channel: {topic: /episode_truncated, type: std_msgs/msg/Bool, dtype: bool}
    align: {strategy: hold, timeline: receive}
```

The extended sections (`rewards`, `signals`, `info`, `complementary_data`)
are ordinary frame entries with three extra rules: `dtype` is mandatory, they
are never images, and they are record-only (written to bags and datasets,
never fed to a policy at inference).

Task labels are **per-frame**. During conversion, each frame's `task` is the latest string received on a task topic at or before the frame (hold semantics), so the task changes mid-episode and LeRobot stores a per-frame `task_index` accordingly. Frames before the first task message, or recordings with no task topic at all, fall back to the `prompt` argument passed when recording. Single-task episodes need no ROS2 topic. At inference, the task comes from the `RunPolicy` goal.

### Topic Recording

By default, the episode recorder records **every topic** on the ROS2 graph, not only those declared in the contract. Contract topics (observations, actions, etc.) must be present. Everything else is captured automatically so you never lose data you need later. This behaves like `ros2 bag record -a`.

To exclude topics, pass the `exclude_topics` parameter to the recorder node.

```python
# In a launch file
Node(
    package='rosetta',
    executable='episode_recorder_node',
    parameters=[{
        'contract_path': '/path/to/contract.yaml',
        'exclude_topics': ['/camera/.*/debug', '/diagnostics'],
    }],
)
```

Only `/rosout` and `/parameter_events` are excluded automatically.

To disable auto-recording and only record contract-declared topics, set the recorder's `record_all` parameter to `false` (in `params/episode_recorder.yaml` or per-launch).

### Adjunct Topics

Adjunct topics are recorded to the bag file with no LeRobot feature mapping. Unlike auto-discovered topics, adjunct topics are **required** at record time.

```yaml
adjunct:
  - channel: {topic: /tf, type: tf2_msgs/msg/TFMessage}
  - channel: {topic: /diagnostics, type: diagnostic_msgs/msg/DiagnosticArray}
  - channel: {topic: /imu/raw, type: sensor_msgs/msg/Imu}
```

The bag preserves this data, so add a contract mapping for these topics later and re-run the porter without re-recording.

### Select Syntax

`select` is a flat list of dot-notation paths extracting nested fields from
ROS2 messages:

```yaml
# JointState: {field}.{joint_name}
select: [position.shoulder, velocity.shoulder]

# Odometry: nested path
select: [twist.twist.linear.x, pose.pose.position.z]
```

### Alignment Strategies

| Strategy | Behavior |
|----------|----------|
| `hold` | Use most recent message, no matter how old |
| `asof` | Use most recent message only if within the `tolerance_ms` window, otherwise a gap. Useful for rejecting stale data |
| `drop` | Use most recent message only if it arrived within the current step/frame window, otherwise a gap |

Every frame-clock entry declares one explicitly. There is no default.

**What a gap looks like.** Before **warmup**, no frames are emitted at all.
Recording and inference start only once every *observation* stream has
produced at least one sample. Actions and extended sections legitimately
start late or publish sparsely. After warmup, a stream with no sample at a
tick **zero-fills** at its static dim (a zero vector for numerics, a zero
image, `""` for strings), so every frame always has the declared shape. This
is deliberate and not configurable. Live inference never skips a tick: the
policy needs an observation every step, and staleness is handled by
missing-stream logging and the action safety watchdog, not by the frame
shape. Offline, dropped frames would silently break the `fps` grid the
dataset declares. Bag conversion and the live bridge share this behavior, so
a gap looks identical in training data and at inference. With `hold`,
post-warmup gaps rarely occur. Zero-fill shows up with `asof`/`drop` and on
sparse extended streams (e.g. a reward published once per episode holds
after the first message, zero-filled before).

### Supported Message Types

| Type | Extracted Fields |
|------|------------------|
| `sensor_msgs/msg/JointState` | position, velocity, effort by joint name |
| `sensor_msgs/msg/Image` | RGB uint8 array |
| `sensor_msgs/msg/CompressedImage` | Decoded to RGB uint8 |
| `geometry_msgs/msg/Twist` | linear.xyz, angular.xyz |
| `geometry_msgs/msg/TwistStamped` | twist.linear.xyz, twist.angular.xyz |
| `nav_msgs/msg/Odometry` | pose, twist fields |
| `sensor_msgs/msg/Joy` | axes, buttons arrays |
| `sensor_msgs/msg/Imu` | orientation, angular_velocity, linear_acceleration |
| `control_msgs/msg/MultiDOFCommand` | values, values_dot by DOF name |
| `trajectory_msgs/msg/JointTrajectory` | first-point position/velocity/acceleration/effort by joint name |
| `std_msgs/msg/Float32` | Scalar float32 |
| `std_msgs/msg/Float64` | Scalar float64 |
| `std_msgs/msg/Int32` | Scalar int32 |
| `std_msgs/msg/Int64` | Scalar int64 |
| `std_msgs/msg/String` | Text string |
| `std_msgs/msg/Bool` | Boolean |
| `std_msgs/msg/Float32MultiArray` | Vector float32 |
| `std_msgs/msg/Float64MultiArray` | Vector float64 |
| `std_msgs/msg/Int32MultiArray` | Vector int32 |

**When to write `dtype`.** A stream's dtype resolves by precedence (explicit
`channel.dtype` > `video` for image keys > `float64` for custom decoders >
the codec registry's native dtype), so most entries never declare one. Write
`dtype` in exactly three situations:

1. **Custom decoder.** The registry has no knowledge of your function's
   return type. Without a declared dtype, custom-decoded streams are assumed
   `float64`.
2. **Multi-source keys.** All sources of one key must resolve to a single
   dtype, and different codecs have different natives (e.g. turtlebot3 pins
   `dtype: float64` on `/imu` and `/odom` to match the JointState codec).
3. **Extended sections** (`rewards`/`signals`/`info`/`complementary_data`).
   Mandatory: record-only columns have no other type source.

### Custom Encoders/Decoders

Support ROS message types beyond the built-ins by writing custom decoders
(ROS → numpy) and encoders (numpy → ROS). Codecs are keyed by message type in
a registry. There are two ways to register yours.

#### Method 1: Plugin via entry points (recommended)

Package your codecs and advertise the module under the `rosetta.codecs`
entry-point group. Rosetta discovers and imports the module at contract load,
running the `@register_*` decorators, so the contract names the type only,
with no module paths in the YAML:

```python
# my_pkg/my_codecs.py
import numpy as np
from rosetta.frames.codecs import register_decoder, register_encoder

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
```toml
# in your plugin's pyproject.toml
[project.entry-points."rosetta.codecs"]
my_codecs = "my_pkg.my_codecs"
```
```yaml
observations:
  observation.state:
    channel: {topic: /my_sensor,
              type: my_msgs/msg/MyCustomSensor}  # codec self-registered; no path needed
    align: {strategy: hold, timeline: receive}
```

Registering a second codec for an already-covered type is an error, so two
plugins never silently conflict over a type. A plugin whose import fails
latches the failure: every later discovery call re-raises the original error
until the process restarts. To replace a built-in (e.g. you wrote a better
`sensor_msgs/msg/Image` decoder), pass `override=True`:

```python
@register_decoder("sensor_msgs/msg/Image", dtype="video", override=True)
def my_better_image_decoder(msg, spec): ...
```

#### Method 2: Inline path in the contract (per-spec override)

Point a single spec directly at a codec function. The override applies to
one spec only and needs no packaging. Use this for a one-off, or to run a
different decoder on one topic while the registry default applies elsewhere:

```yaml
actions:
  action:
    channel:
      topic: /my_command
      type: my_msgs/msg/MyCustomCommand
      decoder: my_package.codecs:decode_my_command  # module:function (reading bags)
      encoder: my_package.codecs:encode_my_command  # for publishing
    align: {strategy: hold, timeline: receive}
```

The module must be importable. Paths are validated at contract load time.

> **Trust model: a contract is code-equivalent.** Loading a contract *imports*
> every named `decoder:`/`encoder:` module (the import is the path validation)
> and, at runtime, invokes those functions on robot message data. Only load
> contracts you trust. This matters most for the policy runner's sidecar
> fallback, which fetches `rosetta_contract.yaml` from a Hugging Face Hub
> model or dataset repo. A contract downloaded from a third-party repo is
> treated as trusted input, exactly like a launch file. When a hub-resolved
> contract declares inline `decoder:`/`encoder:` paths, the runner logs a
> warning naming each one.

> **Round-trip safety:** every built-in encoder/decoder pair is round-trip
> tested (`decode(encode(v)) == v`) in the test suite. When you contribute a
> new built-in pair, add a sample message to those tests.

Registration accepts one more flag. `requires_select=True` marks a codec
unable to produce a value without a `select` list (e.g. `JointState`, which
needs joint names to know which fields to extract). A contract using such a
channel without `select` fails at load instead of at runtime.

#### Function Signatures

**Decoder:** Converts ROS message → numpy array

```python
def my_decoder(msg, spec) -> np.ndarray:
    # msg: ROS message instance
    # spec.names: list of selected field paths from the contract
    # spec.source.channel.type: ROS message type string
    return np.array([...], dtype=np.float64)
```

**Encoder:** Converts numpy array → ROS message

```python
def my_encoder(values, spec, stamp_ns=None):
    # values: numpy array of action values (decode_value/encode_value already ran
    #         spec.operators in the serve/inverse direction, e.g. clamp/deg2rad,
    #         before your encoder sees them)
    # spec.names: list of selected field paths from the contract
    # stamp_ns: optional timestamp in nanoseconds
    msg = MyMessage()
    # ... populate msg from values ...
    return msg
```

#### When Each Is Used

| Field | Used By | Purpose |
|-------|---------|---------|
| `decoder` on observations | Runtime, porter | Decode incoming sensor data |
| `decoder` on actions | porter | Read recorded actions from bags |
| `encoder` on actions | Runtime | Publish actions to ROS topics |

---


## LeRobot Data Model Reference

This section covers LeRobot's internal data model in detail. You do not need this to get started. Refer back here for key conventions, feature types, or policy compatibility.

### Key System


**LeRobot keys are flat dictionary strings using dots as a naming convention.** `observation.state.joint_position` is a single string key, not a nested lookup. The only hard rule is **no forward slashes** (`/`) in key names.

Keys work at any depth:

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

#### How LeRobot classifies keys

Keys are free-form strings, but LeRobot policies use **prefix matching** to classify them into feature types. The classification determines how policies process each feature:

| Prefix | FeatureType | How policies use it |
|--------|-------------|---------------------|
| `observation.images.*` or `observation.image` | `VISUAL` | Fed through vision encoder |
| `observation.environment_state` (exact) | `ENV` | Separate encoder projection (privileged sim state) |
| `observation.*` (everything else under observation) | `STATE` | Robot state encoder |
| `observation.language.*` | `LANGUAGE` | Tokenized text for VLA forward pass |
| `action*` | `ACTION` | Policy output / training target |
| `next.reward` | `REWARD` | RL reward signal |

So `observation.state.imu`, `observation.state.joint_position`, and `observation.state` all classify as `STATE`, and `action.arm` and `action.gripper` both classify as `ACTION`.

#### Convention vs. compatibility

LeRobot's key system has two layers:

1. **The dataset format** accepts any key string. `observation.state.fake_sensor.special_data` or `my_custom_thing` both store fine.
2. **Built-in policies** look for specific keys by exact match. ACT, SmolVLA, and Pi0 all expect `observation.state` and `action` as single combined vectors.


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

LeRobot defines a [Universal Data Container](https://huggingface.co/docs/lerobot/introduction_processors#envtransition-the-universal-data-container) called `EnvTransition`, descended from the classic Gymnasium `step()` return (`observation, reward, terminated, truncated, info`).

The `EnvTransition` TypedDict defines six top-level slots. The contract makes the mapping between ROS2 and the EnvTransition's semantic categories explicit. No core policy uses all components.

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

**`observation.state`** vs **`observation.environment_state`**: semantically distinct. `state` is the robot's proprioception, what the robot knows about its own body (joint angles, gripper width, EEF pose). `environment_state` is privileged information about the external world (object positions, contact forces), usually available only in simulation. They carry different `FeatureType`s (`STATE` vs `ENV`) and policies encode them with separate projections.

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

These serve different purposes and coexist:

| Concept | Key(s) | Type | Purpose |
|---------|--------|------|---------|
| **Task string** | `task` | `str` | Human-readable label: `"pick up the red block"` |
| **Language tokens** | `observation.language.tokens` | `Tensor (int)` | Tokenized text for VLA forward pass |
| **Language mask** | `observation.language.attention_mask` | `Tensor (bool)` | Attention mask for tokenized text |

The **flow** between them: the dataset stores a `task_index` (int) per frame, resolved to a `task` string via `meta/tasks.parquet`. How the string reaches the policy depends on the policy:

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

Per-frame metadata flowing through training without being a model input:

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

The `*_is_pad` flags mark which frames in a temporal window are real vs. padded (used when a policy looks at past frames before enough exist).

Every dataset gets the five default features (`timestamp`, `frame_index`, `episode_index`, `index`, `task_index`) automatically. Do not declare them.

#### Info

The `info` slot in `EnvTransition` is **runtime-only**, never persisted to datasets. The slot carries transient signals like teleop events (`is_intervention`, `end_success`, `end_failure`) used during live recording and policy execution. For persistent metadata, use `complementary_data` instead.

Note: `meta/info.json` in the dataset directory is unrelated. That file stores the dataset schema (features, fps, robot_type), not per-frame data.

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
| `geometry_msgs/msg/TwistStamped` | `float64` | Selected twist components |
| `nav_msgs/msg/Odometry` | `float64` | Selected pose/twist fields |
| `sensor_msgs/msg/Imu` | `float64` | Orientation, angular vel, linear accel |
| `sensor_msgs/msg/Joy` | `float32` | Selected axes/buttons |
| `control_msgs/msg/MultiDOFCommand` | `float64` | Selected DOF values |
| `trajectory_msgs/msg/JointTrajectory` | `float64` | Selected first-point fields |
| `std_msgs/msg/Float32` | `float32` | Scalar `(1,)` |
| `std_msgs/msg/Float64` | `float64` | Scalar `(1,)` |
| `std_msgs/msg/Int32` | `int32` | Scalar `(1,)` |
| `std_msgs/msg/Int64` | `int64` | Scalar `(1,)` |
| `std_msgs/msg/String` | `string` | Text `(1,)` |
| `std_msgs/msg/Bool` | `bool` | Boolean `(1,)` |
| `std_msgs/msg/Float32MultiArray` | `float32` | Vector `(N,)` |
| `std_msgs/msg/Float64MultiArray` | `float64` | Vector `(N,)` |
| `std_msgs/msg/Int32MultiArray` | `int32` | Vector `(N,)` |

Override the auto-detected dtype with the `dtype` field in the contract, or use a [custom decoder](#custom-encodersdecoders) for non-standard message types.

### Policy Feature Compatibility

Each LeRobot policy implements its own `validate_features()` and accesses batch keys differently. There is no single enforced schema. Accepted keys depend on the policy. This table summarizes the requirements from the modeling code in `lerobot/src/lerobot/policies/`:

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

- **Max images**: All "multi" policies handle N cameras, configured at init time. No policy has unlimited image capacity. ACT concatenates image features, so the practical limit depends on the model's hidden dimension. VLA policies (Pi0 family, SmolVLA, Wall-X) feed images through a VLM, so the VLM context window constrains the image count. Setups with 2 to 3 cameras rarely hit these limits.
- **Max language tokens**: Maximum token count the policy's tokenizer keeps from your task string. Longer prompts truncate.
- **Chunk size**: Future action steps the policy predicts per inference call. Larger chunks mean fewer inference calls and less reactivity. Most policies build architecture (positional embeddings, pre-allocated tensors) to match the configured `chunk_size` at init time.
- **RTC (Real-Time Chunking)**: An [inference wrapper](https://huggingface.co/docs/lerobot/rtc) improving real-time performance by overlapping action chunks with continuous re-planning. Flow-matching policies only (Pi0 family plus SmolVLA).
- **Async inference**: Whether the policy is in LeRobot's gRPC-based asynchronous inference server allowlist (`SUPPORTED_POLICIES` in `async_inference/constants.py`). [Async](https://huggingface.co/docs/lerobot/rtc) decouples observation collection from action computation, useful for high-frequency control loops. Pi0-Fast, Wall-X, and X-VLA implement `predict_action_chunk()` and are technically compatible, but are not on the allowlist yet.

**VLA language pipeline**: All VLA policies require a `task` string (e.g. `"pick up the red block"`). In Rosetta, the string comes from the `prompt` argument when recording or running a policy. Tokenization into tensors happens automatically, either by LeRobot's `TokenizerProcessorStep` (a pipeline step running before the policy sees the data) or inside the policy. From a Rosetta/ROS2 perspective, **you provide the task prompt and nothing else**.

**Subtask support**: LeRobot's `lerobot-annotate` [tool](https://huggingface.co/spaces/lerobot/annotate) adds subtask annotations to recorded episodes (e.g. marking "reach for object", "grasp", "lift" within a longer task). Annotations store as `language_instruction` columns in the dataset. **No current action policy consumes subtask annotations.** [SARM](https://huggingface.co/docs/lerobot/sarm) (a reward model) uses them to compute progress scores for [RA-BC](https://huggingface.co/docs/lerobot/sarm) weighted training of Pi0, Pi0.5, and SmolVLA.

#### What this means for your contract

The keys in your contract determine which policies train on your dataset. Practical guidance:

**Maximum compatibility**: for a dataset working with the widest range of policies:

```yaml
observations:
  observation.state:                # Required by: SmolVLA, Wall-X
    channel: {topic: /joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: header}
    select: [...]

  observation.images.top:           # At least 1 image required by most policies
    channel: {topic: /camera/image_raw/compressed,
              type: sensor_msgs/msg/CompressedImage}
    align: {strategy: hold, timeline: header}
    apply: [resize: [480, 640]]

actions:
  action:                           # Required by all action policies
    channel: {topic: /joint_commands, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: header}
    select: [...]

# For VLA policies, also provide a task prompt when recording:
# ros2 action send_goal ... "{prompt: 'pick up the red block'}"
```

**For VLA fine-tuning**: add a second camera and ensure your recording prompts are descriptive:

```yaml
observations:
  # ... state and first camera as above ...

  observation.images.wrist.right:
    channel: {topic: /wrist_camera/image_raw/compressed,
              type: sensor_msgs/msg/CompressedImage}
    align: {strategy: hold, timeline: header}
    apply: [resize: [512, 512]]
```


## License

Apache-2.0
