<p align="center">
  <img alt="Rosetta" src="media/rosetta_logo.png" width="100%">
</p>
<!-- <p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/ROS2-Humble%20%7C%20Jazzy-blue" alt="ROS2">
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python 3.10+">
</p> -->

**Rosetta** bridges ROS2 robots to [LeRobot](https://github.com/huggingface/lerobot). Write a contract YAML that maps your ROS2 topics to LeRobot features, then use that contract for recording, dataset conversion, and deployment. Works with any ROS2 robot using [supported message types](#supported-message-types), or bring your own [encoders/decoders](#extending-experimental).

Rosetta implements LeRobot's [Robot](https://huggingface.co/docs/lerobot/integrate_hardware) and [Teleoperator](https://huggingface.co/docs/lerobot/integrate_hardware#adding-a-teleoperator) interfaces, so you can use LeRobot's CLI tools directly.

> **Recent Changes:**
> - **Contract:** `name` → `robot_type`, `rate_hz` → `fps`
> - **Nodes:** `PolicyBridge` → `rosetta_client_node`, `EpisodeRecorderServer` → `episode_recorder_node`
> - **Actions:** `/run_policy` → `/rosetta_client/run_policy`, `/record_episode` → `/episode_recorder/record_episode`
> - **Launch:** `turtlebot_policy_bridge.launch.py` → `rosetta_client_launch.py`, `turtlebot_recorder_server.launch.py` → `episode_recorder_launch.py`
> - **Conversion:** `bag_to_lerobot.py` → `port_bags.py` (now processes directories, supports sharding)
> - **Inference:** Policy loading moved to LeRobot's async gRPC server
> - **New:** `lerobot_teleoperator_rosetta` (experimental), `rosetta_rl` (coming soon)

## Workflow

**1. Define** a contract for your robot:

```yaml
# contract.yaml
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

**3. Convert** bags to LeRobot dataset:

```bash
python -m rosetta.port_bags \
    --raw-dir ./recordings \
    --contract contract.yaml \
    --repo-id my-org/my-dataset
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

## Architecture

Rosetta consists of five packages that implement LeRobot's official interfaces:

| Package | Purpose |
|---------|---------|
| `rosetta` | Core library, nodes, bag conversion |
| `rosetta_interfaces` | ROS2 action/service definitions |
| `lerobot_robot_rosetta` | LeRobot Robot plugin |
| `lerobot_teleoperator_rosetta` | LeRobot Teleoperator plugin (experimental) |
| `rosetta_rl` | HIL-SERL reinforcement learning (coming soon) |

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

This means any ROS2 robot can use LeRobot's tools. Define a contract and use `--robot.type=rosetta`.

### ROS2 Lifecycle Integration

LeRobot's `connect()` / `disconnect()` map to ROS2 lifecycle transitions:

| LeRobot Method | Lifecycle Transition | Effect |
|----------------|---------------------|--------|
| - | `configure` | Create subscriptions (start buffering), create publishers (disabled) |
| `connect()` | `activate` | Enable publishers, start watchdog |
| `disconnect()` | `deactivate` → `cleanup` | Safety action, disable publishers, destroy resources |

### Policy Inference

The `rosetta_client_node` delegates inference to LeRobot's async gRPC policy server. This provides:

- Better GPU memory management
- Support for all LeRobot policy types without code changes
- Consistent behavior between training and deployment

### rosetta_ws Workspace

We provide [rosetta_ws](https://github.com/iblnkn/rosetta_ws), a devcontainer workspace for getting started quickly. Getting ROS2 and LeRobot installed together is not trivial; the workspace handles this setup.

---

## Nodes

Both nodes use **parameter files** (`params/`) as the source of truth for default values. All parameters are exposed as launch arguments, so you can override any parameter at launch time without editing the params file.

```
rosetta/
├── launch/
│   ├── episode_recorder_launch.py
│   └── rosetta_client_launch.py
└── params/
    ├── episode_recorder.yaml    # Default config for Episode Recorder
    └── rosetta_client.yaml      # Default config for Rosetta Client
```

**Parameter loading order:**
1. Params file loads first (provides defaults)
2. Launch arguments override params file values

To see all available launch arguments:

```bash
ros2 launch rosetta episode_recorder_launch.py --show-args
ros2 launch rosetta rosetta_client_launch.py --show-args
```

### Episode Recorder

Records contract-specified topics to rosbag. Uses ROS2 lifecycle management.

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

### Rosetta Client

Runs policy inference via LeRobot's async inference server.

```bash
ros2 launch rosetta rosetta_client_launch.py contract_path:=/path/to/contract.yaml
```

Run policy:

```bash
ros2 action send_goal /rosetta_client/run_policy \
    rosetta_interfaces/action/RunPolicy "{prompt: 'task description'}"
```

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

**Examples:**

```bash
# Run with a different model
ros2 launch rosetta rosetta_client_launch.py \
    contract_path:=/path/to/contract.yaml \
    pretrained_name_or_path:=my-org/my-policy

# Use SmolVLA with a remote server
ros2 launch rosetta rosetta_client_launch.py \
    contract_path:=/path/to/contract.yaml \
    pretrained_name_or_path:=lerobot/smolvla_base \
    policy_type:=smolvla \
    launch_local_server:=false \
    server_address:=192.168.1.100:8080

# Fine-tune inference behavior
ros2 launch rosetta rosetta_client_launch.py \
    contract_path:=/path/to/contract.yaml \
    actions_per_chunk:=50 \
    aggregate_fn_name:=latest_only
```

### port_bags

Converts ROS2 bags to LeRobot datasets using contract-driven decoding.

`port_bags.py` follows LeRobot's dataset porting conventions. For large-scale conversions, parallel processing, and SLURM cluster workflows, see the **[LeRobot Porting Datasets Guide](https://huggingface.co/docs/lerobot/en/porting_datasets_v3)** and substitute `port_bags.py` for `port_droid.py` in the examples.

#### Relationship to LeRobot

`port_bags.py` mirrors the interface of LeRobot's example porters (like `port_droid.py`):

```bash
# LeRobot's port_droid.py
python examples/port_datasets/port_droid.py \
    --raw-dir /data/droid/1.0.1 \
    --repo-id my_org/droid \
    --push-to-hub

# Rosetta's port_bags.py (same pattern + contract)
python -m rosetta.port_bags \
    --raw-dir /data/recordings \
    --contract contract.yaml \
    --repo-id my_org/my_dataset \
    --push-to-hub
```

**Rosetta-specific additions:**

| Argument | Description |
|----------|-------------|
| `--contract` | **(Required)** Rosetta contract YAML that defines ROS2 topic → LeRobot feature mapping |
| `--root` | Override output directory (LeRobot defaults to `~/.cache/huggingface/lerobot`) |
| `--vcodec` | Video codec selection (not in base LeRobot porters) |

#### Basic Usage

```bash
python -m rosetta.port_bags \
    --raw-dir /path/to/bags \
    --contract /path/to/contract.yaml \
    --repo-id my_dataset
```

#### All Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--raw-dir` | Yes | Directory containing bag subdirectories (each with `metadata.yaml`) |
| `--contract` | Yes | Path to Rosetta contract YAML |
| `--repo-id` | No | Dataset name or HuggingFace repo ID. Defaults to `--raw-dir` directory name |
| `--root` | No | Parent directory for datasets. Dataset saved to `root/repo-id`. Defaults to `~/.cache/huggingface/lerobot` |
| `--push-to-hub` | No | Upload to HuggingFace Hub after conversion (flag) |
| `--num-shards` | No | Total shards for parallel processing (SLURM) |
| `--shard-index` | No | Index of this shard (0 to num-shards-1) |
| `--vcodec` | No | Video encoder (default: `libsvtav1`). Options: `libx264` (fast), `libsvtav1` (good compression), `h264_nvenc` (GPU) |

#### Examples

**Convert to local directory:**

```bash
# Bags at /data/recordings, output to /data/lerobot/my_robot_dataset
python -m rosetta.port_bags \
    --raw-dir /data/recordings \
    --contract /home/user/my_robot_contract.yaml \
    --repo-id my_robot_dataset \
    --root /data/lerobot

# Use faster H.264 encoder instead of default AV1
python -m rosetta.port_bags \
    --raw-dir /data/recordings \
    --contract /home/user/my_robot_contract.yaml \
    --repo-id my_robot_dataset \
    --root /data/lerobot \
    --vcodec libx264
```

**Convert and push to HuggingFace:**

```bash
python -m rosetta.port_bags \
    --raw-dir ./recordings \
    --contract ./contract.yaml \
    --repo-id my-org/my-dataset \
    --push-to-hub
```

**Parallel conversion (recommended for faster processing):**

The conversion bottleneck is video encoding (SVT-AV1). Use sharding to process multiple bags in parallel:

```bash
# Using the parallel wrapper script (handles sharding + aggregation)
./scripts/convert_bags_parallel.sh \
    /path/to/bags \
    /path/to/contract.yaml \
    my_dataset \
    /path/to/output \
    4  # number of parallel shards
```

Or manually with sharding:

```bash
# Run multiple shards in parallel (in separate terminals or background)
for i in 0 1 2 3; do
    python -m rosetta.port_bags \
        --raw-dir /path/to/bags \
        --contract /path/to/contract.yaml \
        --repo-id my_dataset_shard_$i \
        --root /path/to/shards \
        --num-shards 4 \
        --shard-index $i &
done
wait

# Then aggregate the shards
python -c "
from lerobot.datasets.aggregate import aggregate_datasets
from pathlib import Path

aggregate_datasets(
    repo_ids=['my_dataset_shard_0', 'my_dataset_shard_1', 'my_dataset_shard_2', 'my_dataset_shard_3'],
    aggr_repo_id='my_dataset',
    roots=[Path('/path/to/shards')] * 4,
    aggr_root=Path('/path/to/output'),
)
"
```

**Large-scale conversion with SLURM:**

```bash
python -m rosetta.port_bags \
    --raw-dir ./recordings \
    --contract ./contract.yaml \
    --repo-id my-org/my-dataset \
    --num-shards 100 \
    --shard-index $SLURM_ARRAY_TASK_ID
```

For SLURM cluster workflows with `datatrove`, aggregation, and upload scripts, see the [LeRobot Porting Datasets Guide](https://huggingface.co/docs/lerobot/en/porting_datasets_v3).

#### Directory Structure

The `--raw-dir` should contain bag directories, each identified by a `metadata.yaml` file:

```
raw-dir/
├── episode_001/
│   ├── metadata.yaml
│   └── episode_001_0.mcap
├── episode_002/
│   ├── metadata.yaml
│   └── episode_002_0.mcap
└── ...
```

The output LeRobot dataset is saved to `{root}/{repo-id}/`:

```
root/
└── my_dataset/
    ├── meta/
    │   ├── info.json
    │   ├── episodes.jsonl
    │   └── ...
    └── data/
        └── ...
```

## Training a Policy

Once you've converted your ROS2 bags to a LeRobot dataset, train with `lerobot-train`.

### Supported Policies

| Policy | Type | Best For |
|--------|------|----------|
| **ACT** | Behavior Cloning | General manipulation, fast training (recommended for beginners) |
| **Diffusion Policy** | Diffusion | Complex multi-modal tasks |
| **SmolVLA** | VLA | Efficient VLA, good for resource-constrained setups |
| **Pi0 / Pi0Fast** | VLA | Physical Intelligence foundation models |
| **Pi0.5** | VLA | Open-world generalization |
| **NVIDIA GR00T N1.5** | VLA | Humanoid and general robotics |
| **X-VLA** | VLA | Cross-embodiment with soft prompts |
| **VQ-BeT** | Behavior Transformer | Discrete action spaces |
| **TDMPC** | Model-based RL | Sample-efficient learning |

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

VLA models are large pre-trained vision-language-action models. Use PEFT/LoRA for efficient fine-tuning:

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

**Available pre-trained models:**

| Policy | Default Pretrained |
|--------|-------------------|
| smolvla | `lerobot/smolvla_base` |
| pi0 | `lerobot/pi0_base` |
| pi0fast | `lerobot/pi0fast_base` |
| pi05 | `lerobot/pi05_base` |
| xvla | `lerobot/xvla-base` |

### Multi-GPU Training

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

### Further Reading

- **[Imitation Learning on Real Robots](https://huggingface.co/docs/lerobot/il_robots)** - Full tutorial
- **[Multi-GPU Training](https://huggingface.co/docs/lerobot/multi_gpu_training)** - Scale with Accelerate
- **[PEFT/LoRA Fine-tuning](https://huggingface.co/docs/lerobot/peft_training)** - Efficient VLA fine-tuning

## Contract Reference

A contract is a YAML file that maps ROS2 topics to LeRobot's observation/action interface.

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
  # State vector
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
      resize: [480, 640]  # [height, width]
```

Multiple topics can share the same `key`. Values are concatenated.

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

## Supported Message Types

| Type | Extracted Fields |
|------|------------------|
| `sensor_msgs/msg/JointState` | position, velocity, effort by joint name |
| `sensor_msgs/msg/Image` | RGB uint8 array |
| `sensor_msgs/msg/CompressedImage` | Decoded to RGB uint8 |
| `geometry_msgs/msg/Twist` | linear.xyz, angular.xyz |
| `nav_msgs/msg/Odometry` | pose, twist fields |
| `sensor_msgs/msg/Joy` | axes, buttons arrays |
| `sensor_msgs/msg/Imu` | orientation, angular_velocity, linear_acceleration |

### Selector Syntax

Dot notation extracts nested fields:

```yaml
# JointState: {field}.{joint_name}
names: [position.shoulder, velocity.shoulder]

# Odometry: nested path
names: [twist.twist.linear.x, pose.pose.position.z]
```

### Alignment Strategies

| Strategy | Behavior |
|----------|----------|
| `hold` | Use most recent message (default) |
| `asof` | Interpolate to exact timestamp |
| `drop` | Skip frame if no message available |

## Extending (Experimental)

> **Note:** Custom encoder/decoder support is **experimental**. 

Add support for unsupported ROS message types by writing custom decoders (ROS → numpy) and encoders (numpy → ROS).

### Method 1: Specify in Contract (Recommended)

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

### Method 2: Global Registration

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

### Function Signatures

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

### When Each Is Used

| Field | Used By | Purpose |
|-------|---------|---------|
| `decoder` on observations | Runtime, `port_bags.py` | Decode incoming sensor data |
| `decoder` on actions | `port_bags.py` | Read recorded actions from bags |
| `encoder` on actions | Runtime | Publish actions to ROS topics |

## License

Apache-2.0
