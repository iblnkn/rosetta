
# rosetta — ROS 2 ⇄ LeRobot bridge

**rosetta** is a ROS 2 package that standardizes the interface between ROS 2 topics and LeRobot policies using a small YAML **contract**. It provides:

- **bag_to_lerobot.py** — converts recorded bags into a ready-to-train **LeRobot v3** dataset 

## Install & Build

#### Create workspace + venv (system Python)

```bash
# Create a venv exclusive for this tool. The .gitignore for the venv is inside rosetta 

cd /workspace/rover/ros2/src/fomo/src/rosetta

python3.10 -m venv ./venv
touch ./venv/COLCON_IGNORE
source ./venv/bin/activate

#The venv es requiered because lerobot installs a version of torch and torchvision different from the one we use
pip install -r requirements.txt

# Source ROS and build
cd /workspace/rover/ros2
colcon build --packages-select rosetta
```

Verify after install (optional): `which lerobot-train` to confirm CLI is on `PATH`.


### How to run

To test the tool you need a rosbag with its corresponding metadata.yaml file. Make sure they are both stored in the same folder. 

Single episode: 

`python3 path/to/bag_to_lerobot.py --bag /path/to/bag/folder --contract /path/to/contracts/fomo_test.yaml --out /path/to/outputfolder`

More than one episode: 

`python3 path/to/bag_to_lerobot.py --bags /path/to/episodes/epi1  /path/to/episodes/epi1 /path/to/episodes/epi2 /path/to/episodes/epi3  --contract /path/to/contracts/fomo_test.yaml --out /path/to/outputfolder`

Each bag folder must have an mcap file with its metadata file. In case you trimmed a rosbag and want to obtain a metadata.yaml for it, run: `ros2 bag reindex /mcap_folder/ ` (the mcap file inside mcap_folder/ must be called data_0.mcap)
---

## Contracts

A **contract** is a small YAML that declares which topics, message types, fields, rates, and timing rules a policy consumes and what it publishes. The current state of the contract should allow you to record datasets.

Important considerations: 
- All cameras of type foxglove_msgs/msg/CompressedVideo must have stamp: foxglove 
- Make sure to define the correct image size in the contract. For instance, resize: [720,1280]
- Rate_hz is interpreted as an int

---

## Tasks

Tasks are defined in the metadata.yaml of each mcap at the end: 

```bash
  custom_data: 
    lerobot.operator_prompt: rover_diff_drive
```

If not defined, the task is set as "" 