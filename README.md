
# rosetta — ROS 2 ⇄ LeRobot bridge

**rosetta** is a ROS 2 package that standardizes the interface between ROS 2 topics and LeRobot policies using a small YAML **contract**. It provides:

- **bag_to_lerobot.py** — converts recorded bags into a ready-to-train **LeRobot v3** dataset 

## Install & Build

### Prerequisites

- **ROS 2 Humble** (desktop install recommended)
- **System Python 3.10** (match the system interpreter used to build ROS binaries)
- **rosdep** for runtime ROS deps

> ROS binaries are compiled against the system Python. If you use a virtualenv, base it on the **system interpreter**, and don’t use conda for ROS 2 binaries. See ROS docs: “Using Python Packages with ROS 2”.

#### Create workspace + venv (system Python)

```bash

# venv from the system interpreter (ROS 2 binary-compatible)
python3.10 -m venv ./venv
touch ./venv/COLCON_IGNORE
source ./venv/bin/activate

pip install -r requirements.txt

# Source ROS and build
cd /workspace/rover/ros2
colcon build --packages-select rosetta
```

Verify after install (optional): `which lerobot-train` to confirm CLI is on `PATH`.


### How to run

Single episode: 

`python3 path/to/bag_to_lerobot.py --bag /path/to/bag/folder --contract /path/to/contracts/fomo_test.yaml --out /path/to/outputfolder`

More than one episode: 

`python3 path/to/bag_to_lerobot.py --bags /path/to/episodes/epi1  /path/to/episodes/epi1 /path/to/episodes/epi2 /path/to/episodes/epi3  --contract /path/to/contracts/fomo_test.yaml --out /path/to/outputfolder`

Each bag folder must have an mcap file with its metadata file. In case you trimmed a rosbag and want to obtain a metadata.yaml for it, run: `ros2 bag reindex /mcap_folder/ ` (the mcap file inside mcap_folder/ must be called data_0.mcap)
---

## Contracts

A **contract** is a small YAML that declares which topics, message types, fields, rates, and timing rules a policy consumes and what it publishes. rosetta uses the same contract to:

Important considerations: 
- All cameras of type foxglove_msgs/msg/CompressedVideo must have stamp: foxglove 
- Make sure to define the correct image size in the contract
- Rate_hz is interpreted as an int

---

### Notable options when running 

* `--timestamp {contract,bag,header}` — choose the time base before resampling.
* `--no-videos` — write PNG images instead of MP4.
* `--image-threads / --image-processes` — tune I/O parallelism.
* `--chunk-size --data-mb --video-mb` — size the parquet/video chunks.

Depth images are converted to normalized float in H×W×3 (for LeRobot compatibility) while preserving REP-117 special values (`NaN`, `±Inf`).

---

## Contract file

See `share/rosetta/contracts/fomo_test.yaml`. Highlights:

* **observations** — list of streams (RGB, depth, state). Each specifies topic, type, optional `selector.names` for extracting scalars, `image.resize`, and an `align` policy (`hold`/`asof`/`drop`) with `stamp: header|receive|foxglove`.
* **actions** — what to publish, e.g., `geometry_msgs/Twist` to `/cmd_vel`, with `selector.names` (e.g., `[linear.x, angular.z]`), `from_tensor.clamp`, QoS, and a publish strategy.
* **rate_hz / max_duration_s** — contract rate and episode timeout used across nodes.

## Tasks

Tasks are defined in the metadata.yaml of each mcap at the end: 

```bash
  custom_data: 
    lerobot.operator_prompt: rover_diff_drive
```

If not defined, the task is set as "" 