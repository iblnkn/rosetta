# From bags to a moving arm

In this tutorial you run the whole Rosetta workflow on a simulated SO-ARM101
in Gazebo. You'll deploy a trained policy, build a dataset from recorded
bags, train and deploy your own policy, then change the contract without
recording anything new.

You need a Linux machine (x86_64 or aarch64) and
[Pixi](https://pixi.sh/latest/installation/). Training in step 6 wants an
NVIDIA GPU. The recording is already done:
[ros-physical-ai/demos](https://github.com/ros-physical-ai/demos) publishes
60 bags, a dataset ported from them, and a policy trained on that dataset.

## 1. Install the demos workspace

```bash
git clone https://github.com/ros-physical-ai/demos && cd demos
pixi install
pixi run install-deps
pixi run build
```

`pixi install` brings ROS 2, Gazebo and LeRobot. `install-deps` clones the
robot description, drivers and Rosetta into `external/`, then installs a
PyTorch build that matches your GPU. The build ends with a colcon summary and
no failed packages.

From here on, every command runs inside a `pixi shell` in the `demos`
directory. Open one now. A shell opened before the build doesn't have
`install/setup.bash` sourced. Check that the build includes Rosetta's ROS
package:

```bash
ros2 pkg prefix rosetta_ros
```

It prints a path under `install/`. If it prints `Package not found`, your
demos checkout builds a Rosetta release from before the `rosetta_ros`
package existed, and the Rosetta commands on this page won't run against it.

In a second terminal, start the Zenoh router and leave it running:

```bash
pixi run zenoh-router
```

## 2. Start the robot

In a new `pixi shell`:

```bash
ros2 launch pai_bringup so_arm_gz_bringup.launch.py
```

A Gazebo window opens with the arm on a table, three cubes and a tray. RViz
opens beside it. Check the four topics the contract uses:

```bash
ros2 topic list | grep -E '^/(joint_states|wrist_camera/image_raw|static_camera/image_raw|forward_position_controller/commands)$'
```

It prints all four. If it prints nothing, the Zenoh router isn't running.

## 3. Read the contract

The contract for this robot ships with the demos. Save its path and open it:

```bash
export CONTRACT=$(ros2 pkg prefix pai_data_collection)/share/pai_data_collection/config/rosetta/so_arm101.yaml
cat $CONTRACT
```

It declares three observation keys, `observation.images.wrist`,
`observation.images.static` and `observation.state`, and one action key,
`action`. Under each key, `channel` names a topic and its message type,
`align` says which message to take at each frame tick, and `apply` says how
the values change. For the joint vectors, `select` picks the fields.

Notice the two `resize: [480, 480]` lines. You'll change those in step 8.
Also notice `rad2deg` on the state and the action: ROS publishes radians,
the dataset stores degrees, and the runner converts back to radians before
it publishes a command.

Load it:

```bash
python3 -c "from rosetta.contract.schema import load_contract; load_contract('$CONTRACT'); print('OK')"
```

It prints `OK`.

## 4. Deploy the published policy

Start the policy runner with the published checkpoint from the Hugging Face
Hub. The first start downloads the weights.

```bash
ros2 launch rosetta_ros policy_runner_launch.py \
    params_file:=$(ros2 pkg prefix pai_data_collection)/share/pai_data_collection/config/rosetta/policy_runner.yaml \
    contract_path:=$CONTRACT \
    pretrained_name_or_path:=francocipollone/rospai_act_sim_arm101_place_cubes_on_tray \
    policy_type:=act \
    use_sim_time:=true
```

Wait for `Policy server ready on 127.0.0.1:8080`, then
`Configured: contract=..., framework=lerobot`. Two other lines tell you
something went differently:

- `Requested policy device 'cuda' but the requested backend is unavailable; using 'cpu' instead.`
  The machine has no usable NVIDIA GPU. The policy runs on the CPU.
- `Policy server failed to start within 120.0s` ends the configure. The
  first download took longer than the demos params file allows. Run the
  launch command again.

In another `pixi shell`, send a goal with the task prompt the bags were
recorded with:

```bash
ros2 action send_goal /run_policy \
    rosetta_interfaces/action/RunPolicy "{prompt: 'place cubes into basket'}"
```

It prints `Goal accepted with ID: ...` and keeps running. Watch Gazebo. The
arm picks up the cubes and puts them on the tray.

Press Ctrl+C in the `send_goal` terminal. It prints `Canceling goal...`,
and may then print `Executor is already spinning`. That second line comes
from the `ros2 action` CLI, and the goal is still cancelled. The runner
stops and sends its last command once more, because the contract's action
channel says `safety: hold`. The arm stays where it stopped.

Put the cubes back:

```bash
./pai_data_collection/scripts/gz_set_cubes_poses.py
```

## 5. Port bags into a dataset

Download the first three bag directories from the
[demos bag folder](https://drive.google.com/drive/folders/1x-vtJqVtTHESkQLZCpj7aSnfekpI3YN4)
into `datasets/bags/`. Each directory holds one `.mcap` file and a
`metadata.yaml`. The first one is 1.7 GB. Port them:

```bash
ros2 run rosetta_ros rosetta_port \
    --raw-dir datasets/bags \
    --contract $CONTRACT \
    --repo-id tutorial_480 \
    --root datasets/lerobot
```

The porter logs one line per bag, such as
`-> 2331 frames from 1781844295_447166031` for the first one, and ends with
a line that starts `Completed: 3/3 episodes (0 failed)`. Look at what it wrote:

```bash
ls datasets/lerobot/tutorial_480/meta/
python3 -c "import json; print(json.load(open('datasets/lerobot/tutorial_480/meta/info.json'))['features']['observation.images.wrist']['shape'])"
```

`meta/` holds `info.json`, `stats.json`, the `episodes` and `tasks` tables,
and `rosetta_contract.yaml`, a copy of the contract. The second command
prints `[480, 480, 3]`, which came from the `resize` line.

## 6. Train your own policy

Train on the published dataset rather than your three episodes. It is all
60 bags, ported. LeRobot downloads it on first use.

```bash
lerobot-train \
    --dataset.repo_id=francocipollone/rospai_sim_arm101_place_cubes_on_tray \
    --policy.type=act \
    --output_dir=outputs/train/act_tutorial \
    --job_name=act_tutorial \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --wandb.enable=false \
    --steps=3000 \
    --batch_size=32 \
    --save_freq=1500 \
    --log_freq=500
```

The checkpoint ends up at
`outputs/train/act_tutorial/checkpoints/last/pretrained_model`.

## 7. Deploy your policy

Stop the policy runner from step 4 with Ctrl+C. It can print a
`KeyboardInterrupt` traceback while it stops the policy server. Reset the
cubes as in step 4, then start the runner again on your checkpoint:

```bash
ros2 launch rosetta_ros policy_runner_launch.py \
    params_file:=$(ros2 pkg prefix pai_data_collection)/share/pai_data_collection/config/rosetta/policy_runner.yaml \
    contract_path:=$CONTRACT \
    pretrained_name_or_path:=outputs/train/act_tutorial/checkpoints/last/pretrained_model \
    policy_type:=act \
    use_sim_time:=true
```

Send the same goal as in step 4. The arm moves under your policy. The
published policy trained for 70,000 steps and yours for 3,000, so compare
how the two handle the cubes.

## 8. Change the contract

Make a copy with smaller images:

```bash
sed 's/resize: \[480, 480\]/resize: [240, 240]/' $CONTRACT > so_arm101_240.yaml
diff $CONTRACT so_arm101_240.yaml
```

The diff shows the two `resize` lines. Port the same three bags with the new
contract:

```bash
ros2 run rosetta_ros rosetta_port \
    --raw-dir datasets/bags \
    --contract so_arm101_240.yaml \
    --repo-id tutorial_240 \
    --root datasets/lerobot

python3 -c "import json; print(json.load(open('datasets/lerobot/tutorial_240/meta/info.json'))['features']['observation.images.wrist']['shape'])"
```

It prints `[240, 240, 3]`. The bags didn't change. A policy trained on this
dataset deploys with `contract_path:=so_arm101_240.yaml`, so the runner
resizes the live camera images to 240 by 240 the same way.

You now have the published policy and your own running on the same
simulated arm, and two datasets ported from one set of bags by two
contracts.

## Next

- [Write a contract](../how-to/write-a-contract.md) for your own robot.
- [Record episodes](../how-to/record-episodes.md) with the episode recorder.
- [About the contract](../explanation/design.md) for the reasoning behind
  the design.
