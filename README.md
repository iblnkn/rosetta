
---

# rosetta — Contract-driven ROS 2 ⇄ LeRobot policy bridge, recorder, and dataset exporter

**rosetta** is a ROS 2 package that standardizes the interface between ROS 2 topics and LeRobot policies using a small YAML **contract**. It provides:

* **PolicyBridge** — runs a pretrained policy live, subscribing to topics as observations and publishing topics as actions, exactly as defined by the contract.
* **EpisodeRecorderServer** — records raw ROS 2 topics required by the contract straight to rosbag2. Uses ROS 2 actions to start/stop recording and appends task prompts to bagfile metadata for later processing.
* **bag_to_lerobot.py** — converts recorded bags into a ready-to-train **LeRobot v3** dataset using the same decoding/resampling logic as live inference.

This keeps **train ↔ serve ↔ record** aligned and minimizes data/shape skew.

> **Status:** Early public release. Target ROS 2 distro: **Humble**. Python ≥ 3.10.

---

## Contents

- [rosetta — Contract-driven ROS 2 ⇄ LeRobot policy bridge, recorder, and dataset exporter](#rosetta--contract-driven-ros-2--lerobot-policy-bridge-recorder-and-dataset-exporter)
  - [Contents](#contents)
      - [Examples turtlebot trained on ACT.](#examples-turtlebot-trained-on-act)
  - [Why contracts?](#why-contracts)
  - [Install \& build](#install--build)
    - [Prerequisites](#prerequisites)
    - [Resolve ROS dependencies](#resolve-ros-dependencies)
    - [Build](#build)
    - [ML dependencies](#ml-dependencies)
  - [Quick start](#quick-start)
      - [0) Install Gazebo and TurtleBot3](#0-install-gazebo-and-turtlebot3)
      - [1) Use the included TurtleBot contract](#1-use-the-included-turtlebot-contract)
      - [2) Run a policy bridge](#2-run-a-policy-bridge)
      - [3) Record an episode](#3-record-an-episode)
      - [4) Export bag(s) to a LeRobot dataset](#4-export-bags-to-a-lerobot-dataset)
      - [5) Train a LeRobot model](#5-train-a-lerobot-model)
  - [Nodes](#nodes)
    - [PolicyBridge node](#policybridge-node)
      - [Key parameters](#key-parameters)
      - [Safety on stop](#safety-on-stop)
    - [EpisodeRecorderServer node](#episoderecorderserver-node)
      - [Key parameters](#key-parameters-1)
  - [Dataset export (`bag_to_lerobot.py`)](#dataset-export-bag_to_lerobotpy)
      - [Notable options](#notable-options)
  - [Contract file](#contract-file)
  - [Extending (decoders/encoders)](#extending-decodersencoders)

---

#### Examples turtlebot trained on ACT. 

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/iblnkn/rosetta/media/Drive up to the red pillar_1.gif" width="100%" alt=ACT policy: drive up to the red pillar/></td>
    <td><img src="https://raw.githubusercontent.com/iblnkn/rosetta/media/Drive up to the red pillar_2.gif" width="100%" alt=ACT policy: drive up to the red pillar/></td>
    <td><img src="https://raw.githubusercontent.com/iblnkn/rosetta/media/Drive up to the red pillar_3.gif" width="100%" alt=ACT policy: drive up to the red pillar/></td>
    <td><img src="https://raw.githubusercontent.com/iblnkn/rosetta/media/Drive up to the red pillar_4.gif" width="100%" alt=ACT policy: drive up to the red pillar/></td>
  </tr>
  <tr>
    <td align="center">ACT policy: "drive up to the red pillar</td>
    <td align="center">ACT policy: "drive up to the red pillar</td>
    <td align="center">ACT policy: "drive up to the red pillar</td>
    <td align="center">ACT policy: "drive up to the red pillar</td>
  </tr>
</table>

---

## Why contracts?

A **contract** is a small YAML that declares **exactly** which topics, message types, fields, rates, and timing rules a policy consumes and what it publishes. rosetta uses the same contract to:

* subscribe & resample observations in the bridge,
* encode actions for publishing,
* and decode & resample bag files for dataset export.

That **single source of truth** eliminates mismatched shapes and prevents timestamps/policies from drifting between training and serving.

---

## Install & build

### Prerequisites

* ROS 2 (**Humble**)
* LeRobot
* System Python **3.10+**
* Runtime ROS dependencies are resolved via `rosdep`

> Policy execution and dataset export additionally require PyTorch and LeRobot; see **ML dependencies** below.

### Resolve ROS dependencies

```bash
# From your workspace root (e.g., ~/ws)
rosdep install --from-paths src --ignore-src -r -y
```

### Build

```bash
colcon build --packages-select rosetta
source install/setup.bash
```

### ML dependencies

For live policies and dataset export:

```bash
# Install LeRobot
pip install lerobot
```

If you don’t need live inference or export, you can skip these Python extras.

---

## Quick start

#### 0) Install Gazebo and TurtleBot3

Install Gazebo (Gazebo Classic—deprecated, but the easiest path here for now):

```bash
sudo apt install ros-humble-gazebo-*
sudo apt install ros-humble-turtlebot3-msgs
sudo apt install ros-humble-turtlebot3
sudo apt install ros-humble-turtlebot3-simulations

```

Create a test workspace and install the TurtleBot3 simulation packages:

```bash
mkdir -p ~/rosetta_ws/src
cd ~/rosetta_ws/src
git@github.com:iblnkn/turtlebot3_simulations.git
sudo apt install python3-colcon-common-extensions
cd ~/rosetta_ws
colcon build --symlink-install
echo 'source ~/rosetta_ws/install/setup.bash' >> ~/.bashrc
source ~/.bashrc
```

Launch TurtleBot3 in Gazebo:

```bash
export TURTLEBOT3_MODEL=burger
ros2 launch rosetta turtlebot3_red_pillar_world.launch.py
```

#### 1) Use the included TurtleBot contract

This package ships `contracts/turtlebot.yaml`, describing RGB, depth, state (joints/odom/imu), and a `cmd_vel` action.

#### 2) Run a policy bridge

```bash
# Edit policy_path to your pretrained model directory, or keep the default,
# which will pull a simple ACT model from Hugging Face.
ros2 launch rosetta turtlebot_policy_bridge.launch.py \
  log_level:=info
```

Send a goal to start/stop the run (fields may vary with your `rosetta_interfaces` version):

```bash
# Start a run with an optional prompt/task.
# (The example ACT model is trained on a single action—drive up to the red pillar—
# and does not ingest the prompt.)
ros2 action send_goal /run_policy rosetta_interfaces/action/RunPolicy \
  "{prompt: 'drive up to the red pillar}"
```

```bash
# Stop the action
ros2 service call /run_policy/cancel std_srvs/srv/Trigger "{}"
```

#### 3) Record an episode

```bash
# Separate node that writes topics to rosbag2 (MCAP) under action control
ros2 launch rosetta turtlebot_recorder_server.launch.py log_level:=info

# Start a timed recording
ros2 action send_goal /record_episode rosetta_interfaces/action/RecordEpisode \
  "{prompt: 'kitchen demo 1'}"

# Cancel
ros2 service call /record_episode/cancel std_srvs/srv/Trigger "{}"
```

#### 4) Export bag(s) to a LeRobot dataset

```bash
# Single bag
ros2 run rosetta bag_to_lerobot -- \
  --bag /path/to/bag_dir \
  --contract $(ros2 pkg prefix rosetta)/share/rosetta/contracts/turtlebot.yaml \
  --out /tmp/lerobot_ds

# Or multiple bags
ros2 run rosetta bag_to_lerobot -- \
  --bags /bags/epi1 /bags/epi2 \
  --contract .../turtlebot.yaml \
  --out /tmp/lerobot_ds
```

Output (videos + parquet) is written under `--out` with the LeRobot v3 layout.

#### 5) Train a LeRobot model

Follow LeRobot guidance for how best to train a model.

```bash
# Example
lerobot-train \
  --dataset.repo_id=$iblnk/turtlebot3_demo \
  --policy.type=act \
  --output_dir=lerobot_models/act/ \
  --job_name=act_turtlebot3_example \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=${HF_USER}/act_turtlebot3_example_policy
```

---

## Nodes

### PolicyBridge node

Runs a pretrained policy at a fixed `rate_hz` from the contract, sampling observations by header or receive time (also contract-controlled), and publishing actions with a safety behavior when stopping.

* **Executable:** `policy_bridge`
* **Action server:** `/run_policy` (`rosetta_interfaces/RunPolicy`)
* **Cancel helper:** `/run_policy/cancel` (`std_srvs/Trigger`)
* **Publisher:** as declared in the contract (e.g., `/cmd_vel Twist`)

#### Key parameters

| Param                  | Type   | Default      | Notes                                                                             |
| ---------------------- | ------ | ------------ | --------------------------------------------------------------------------------- |
| `contract_path`        | string | **required** | Path to YAML/JSON contract.                                                       |
| `policy_path`          | string | **required** | Directory with pretrained policy artifacts (e.g., `config.json`, weights, stats). |
| `policy_device`        | string | `auto`       | `auto`, `cpu`, `cuda[:N]`, or `mps`.                                              |
| `use_chunks`           | bool   | `True`       | Batch-generate `actions_per_chunk` actions per policy call.                       |
| `actions_per_chunk`    | int    | `25`         | Actions per chunk when chunking is enabled.                                       |
| `chunk_size_threshold` | float  | `0.5`        | Low-water mark (0..1 of chunk) before refilling the queue.                        |
| `max_queue_actions`    | int    | `512`        | Max buffered actions.                                                             |
| `use_header_time`      | bool   | `True`       | Prefer `msg.header.stamp` to sample observations.                                 |
| `use_autocast`         | bool   | `False`      | Enable `torch.autocast` when supported.                                           |

#### Safety on stop

The action spec can set `safety_behavior: zeros|hold`. On stop/cancel/timeout, the node sends either a zero action or holds the last action.

---

### EpisodeRecorderServer node

Records a predefined set of topics directly to rosbag2 as they arrive. Recording is controlled via an action so you can start/stop programmatically and attach operator prompts for dataset export.

* **Executable:** `recorder_server`
* **Action server:** `/record_episode` (`rosetta_interfaces/RecordEpisode`)
* **Cancel helper:** `/record_episode/cancel` (`std_srvs/Trigger`)
* **Storage:** MCAP by default (from contract `recording.storage`)

#### Key parameters

| Param                    | Type   | Default         | Notes                                        |
| ------------------------ | ------ | --------------- | -------------------------------------------- |
| `contract_path`          | string | **required**    | Same contract used by PolicyBridge.          |
| `bag_base_dir`           | string | `/tmp/episodes` | Episode directories created under this root. |
| `storage_preset_profile` | string | `""`            | Optional rosbag2 preset (e.g., `zstd_fast`). |
| `storage_config_uri`     | string | `""`            | Optional storage config (file URI or path).  |

On stop, the node amends the bag’s `metadata.yaml` to store `custom_data.lerobot.operator_prompt` for later export.

---

## Dataset export (`bag_to_lerobot.py`)

Converts one or more bag directories into a LeRobot v3 dataset using the same decoders and resamplers as live inference, ensuring shape and dtype parity.

```bash
ros2 run rosetta bag_to_lerobot -- --help
```

#### Notable options

* `--timestamp {contract,bag,header}` — choose the time base before resampling.
* `--no-videos` — write PNG images instead of H.264 MP4.
* `--image-threads / --image-processes` — tune I/O parallelism.
* `--chunk-size --data-mb --video-mb` — size the parquet/video chunks.

Depth images are converted to normalized float in H×W×3 (for LeRobot compatibility) while preserving REP-117 special values (`NaN`, `±Inf`).

---

## Contract file

See `share/rosetta/contracts/turtlebot.yaml` for a complete, documented example. Highlights:

* **observations** — list of streams (RGB, depth, state). Each specifies topic, type, optional `selector.names` for extracting scalars, `image.resize`, and an `align` policy (`hold`/`asof`/`drop`) with `stamp: header|receive`.
* **actions** — what to publish, e.g., `geometry_msgs/Twist` to `/cmd_vel`, with `selector.names` (e.g., `[linear.x, angular.z]`), `from_tensor.clamp`, QoS, and a publish strategy.
* **rate_hz / max_duration_s** — contract rate and episode timeout used across nodes.
* **recording.storage** — default rosbag2 backend (MCAP recommended).

---

## Extending (decoders/encoders)

rosetta exposes tiny registries to convert between ROS messages and numpy/torch tensors:

```python
# Add a decoder for a custom message
from rosetta.common.contract_utils import register_decoder

@register_decoder("my_msgs/msg/Foo")
def _dec_foo(msg, spec):
    ...
    return np.array([...], dtype=np.float32)

# Add an encoder for a custom action type
from rosetta.common.contract_utils import register_encoder

@register_encoder("my_msgs/msg/Bar")
def _enc_bar(names, action_vec, clamp):
    ...
    return msg_instance
```

With a corresponding contract entry, the bridge and exporter will automatically use your converters.
