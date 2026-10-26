# From bags to a moving arm

In this tutorial you run the whole Rosetta workflow on a simulated SO-ARM101
in Gazebo. You'll deploy a trained policy, build a dataset from recorded
bags, train and deploy your own policy, then change the contract without
recording anything new.

You need a Linux machine. Training in step 6 wants a GPU. The recording is
already done:
[ros-physical-ai/demos](https://github.com/ros-physical-ai/demos) publishes
60 bags, a dataset built from them, and a policy trained on that dataset.

## 1. Install the demos workspace

```bash
git clone https://github.com/ros-physical-ai/demos && cd demos
pixi install
pixi run install-deps
pixi run build
```

The build ends with a colcon summary and no failed packages.

Every ROS command below runs inside `pixi shell`. Open one now. In a second
terminal, start the Zenoh router and leave it running:

```bash
pixi run zenoh-router
```

## 2. Start the robot

In a `pixi shell`:

```bash
ros2 launch pai_bringup so_arm_gz_bringup.launch.py
```

A Gazebo window opens with the arm at home and three cubes on the table.
Check the topics:

```bash
ros2 topic list | grep -E 'joint_states|camera|forward_position'
```

You'll see `/joint_states`, `/wrist_camera/image_raw`,
`/static_camera/image_raw` and `/forward_position_controller/commands`.

## 3. Read the contract

The contract for this robot ships with the demos. Save its path and open it:

```bash
export CONTRACT=$(ros2 pkg prefix pai_data_collection)/share/pai_data_collection/config/rosetta/so_arm101.yaml
cat $CONTRACT
```

Three observation keys, `observation.images.wrist`,
`observation.images.static` and `observation.state`, and one action key,
`action`. Under each: `channel` names a topic, `align` says which sample to
take, `apply` says how the values change, and for the joint and command
vectors `select` says which fields. Notice the two `resize: [480, 480]`
lines. You'll change those in step 8.

Load it:

```bash
python -c "from rosetta.contract.schema import load_contract; load_contract('$CONTRACT'); print('OK')"
```

It prints `OK`.

## 4. Deploy the published policy

Start the policy runner with the checkpoint from the Hugging Face Hub. The
first start downloads the weights.

```bash
ros2 launch rosetta policy_runner_launch.py \
    params_file:=$(ros2 pkg prefix pai_data_collection)/share/pai_data_collection/config/rosetta/policy_runner.yaml \
    contract_path:=$CONTRACT \
    pretrained_name_or_path:=francocipollone/rospai_act_sim_arm101_place_cubes_on_tray \
    policy_type:=act \
    use_sim_time:=true
```

Wait until the launch output goes quiet. In another `pixi shell`:

```bash
ros2 action send_goal /run_policy \
    rosetta_interfaces/action/RunPolicy "{prompt: 'place cubes on tray'}"
```

Watch Gazebo. The arm picks up the cubes and puts them on the tray. Press
Ctrl+C in the `send_goal` terminal to stop. The arm holds its last position,
which is the `safety: hold` line in the contract.

Put the cubes back:

```bash
pixi run ./pai_data_collection/scripts/gz_set_cubes_poses.py
```

## 5. Prepare a dataset from the bags

Download three bag directories from the
[demos bag folder](https://drive.google.com/drive/folders/1x-vtJqVtTHESkQLZCpj7aSnfekpI3YN4)
into `datasets/bags/`. Each is a directory containing `metadata.yaml`. Port
them:

```bash
ros2 run rosetta rosetta_port \
    --raw-dir datasets/bags \
    --contract $CONTRACT \
    --repo-id tutorial_480 \
    --root datasets/lerobot
```

The porter logs one line per episode. Look at what it wrote:

```bash
ls datasets/lerobot/tutorial_480/meta/
python -c "import json; print(json.load(open('datasets/lerobot/tutorial_480/meta/info.json'))['features']['observation.images.wrist']['shape'])"
```

`meta/` holds `info.json`, `stats.json`, the episode and task tables, and
`rosetta_contract.yaml`, a copy of the contract. The second command prints
`[480, 480, 3]`, which came from the `resize` line.

## 6. Train your own policy

Train on the full published dataset. It's the same 60 bags, ported the same
way, and it downloads on first use. This step takes a while.

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

Stop the policy runner from step 4 with Ctrl+C. Start it again on your
checkpoint:

```bash
ros2 launch rosetta policy_runner_launch.py \
    params_file:=$(ros2 pkg prefix pai_data_collection)/share/pai_data_collection/config/rosetta/policy_runner.yaml \
    contract_path:=$CONTRACT \
    pretrained_name_or_path:=outputs/train/act_tutorial/checkpoints/last/pretrained_model \
    policy_type:=act \
    use_sim_time:=true
```

Send the same goal as in step 4. The arm moves under your policy. After 3000
steps it'll be rougher than the published one.

## 8. Change the contract

Make a copy with smaller images:

```bash
sed 's/resize: \[480, 480\]/resize: [240, 240]/' $CONTRACT > so_arm101_240.yaml
diff $CONTRACT so_arm101_240.yaml
```

The diff shows the two `resize` lines. Port the same three bags with the new
contract:

```bash
ros2 run rosetta rosetta_port \
    --raw-dir datasets/bags \
    --contract so_arm101_240.yaml \
    --repo-id tutorial_240 \
    --root datasets/lerobot

python -c "import json; print(json.load(open('datasets/lerobot/tutorial_240/meta/info.json'))['features']['observation.images.wrist']['shape'])"
```

It prints `[240, 240, 3]`. The bags didn't change. To train and deploy on
this dataset, repeat steps 6 and 7 with `tutorial_240` and
`so_arm101_240.yaml`.

## Next

- [Write a contract](../how-to/write-a-contract.md) for your own robot.
- [Record episodes](../how-to/record-episodes.md) with the episode recorder.
- [About the contract](../explanation/design.md) for the reasoning behind
  the design.
