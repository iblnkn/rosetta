# Record, convert, train, deploy

The pipeline after the contract exists. Node parameters:
[nodes](../reference/nodes.md).

## Record episodes

```bash
ros2 launch rosetta episode_recorder_launch.py contract_path:=/path/to/contract.yaml
```

### Keyboard controller (recommended)

Run it in a second terminal while the recorder is running. Keys: `r` start,
`s` stop and save, `d` discard, `t` edit prompt, `q` quit.

```bash
ros2 run rosetta episode_keyboard_node
```

### ROS 2 action

For scripted workflows, trigger recording directly:

```bash
ros2 action send_goal /record_episode \
    rosetta_interfaces/action/RecordEpisode "{prompt: 'task description'}"
```

Stop with Ctrl-C, or via the cancel service, which is what a dashboard button
wants since Foxglove can call services and cannot call actions:

```bash
ros2 service call /episode_recorder/cancel_recording std_srvs/srv/Trigger
```

Both save the bag. The goal ends `CANCELED`, which is how an untimed recording
ends and not an error. `max_duration_s` on the goal bounds an episode instead.

**How many episodes?** Plan on recording 50 to 200+ demonstrations depending on
task complexity. Vary the starting conditions across takes: a policy trained on
one object position learns that position.

The recorder captures every topic on the graph by default, keeping one
`image_transport` stream per camera. Any valid rosbag2 file containing the
contract's topics works, so `ros2 bag record` and third-party tools are fine
too.

## Convert bags to a dataset

```bash
rosetta_port \
    --raw-dir ./datasets/bags \
    --contract ./contract.yaml \
    --repo-id my-org/my-dataset \
    --root ./datasets/lerobot
```

`rosetta_port` runs the same `StreamBuffer` resampling as live inference, so the
offline dataset matches what the robot sees at runtime. One bag directory is one
episode. Observation topics must be present in the bag; actions, rewards, and
signals may be missing and zero-fill.

Because the bag preserves raw data, re-run the porter with an updated contract
(changing keys, `fps`, alignment, adding topics) without re-recording. Other
flags: `--framework`, `--num-shards` / `--shard-index` for parallel invocations,
`--vcodec`, `--push-to-hub`, `--no-embed-contract`.

For large-scale conversions and SLURM workflows, see the
[LeRobot Porting Datasets Guide](https://huggingface.co/docs/lerobot/en/porting_datasets_v3)
and substitute `rosetta_port` for `port_droid.py`.

## Train

```bash
lerobot-train \
    --dataset.repo_id=my-org/my-dataset \
    --policy.type=act \
    --output_dir=outputs/train/act_my_robot \
    --policy.device=cuda \
    --wandb.enable=true
```

ACT trains fast and is the recommended start. To fine-tune a VLA, use
[PEFT/LoRA](https://huggingface.co/docs/lerobot/peft_training)
(`--policy.path=lerobot/smolvla_base --peft.method_type=LORA --peft.r=64`).
LeRobot also supports [multi-GPU training](https://huggingface.co/docs/lerobot/multi_gpu_training),
resuming with `--config_path=.../train_config.json --resume=true`, and
`huggingface-cli upload` to push a checkpoint.

## Deploy

```bash
# Terminal 1
ros2 launch rosetta policy_runner_launch.py \
    contract_path:=/path/to/contract.yaml \
    pretrained_name_or_path:=my-org/my-policy
```

```bash
# Terminal 2
ros2 action send_goal /run_policy \
    rosetta_interfaces/action/RunPolicy "{prompt: 'task description'}"
```

**Without a contract file.** Datasets ported with default settings embed the
contract. Leave `contract_path` empty and the node resolves it from the
checkpoint chain. A non-empty `contract_path` is used as given and never
compared against the checkpoint's own.

**Remote inference.** Set `launch_local_server:=false` and point `server_address`
at a machine running the policy server, which has no ROS 2 dependency and can run
on any machine with a GPU. This lets a resource-constrained robot offload
inference over the network.

**With LeRobot's own tools.** The `policy_runner_node` is optional. Installing
`lerobot_robot_rosetta` is the whole setup, since LeRobot's CLIs discover it by
the `lerobot_robot_*` name prefix:

```bash
lerobot-record --robot.type=rosetta --robot.config_path=contract.yaml
```

`actions_per_chunk` is the first tuning knob: larger chunks mean fewer inference
calls and a less reactive robot. `aggregate_fn_name` blends an arriving chunk
with the one still executing; set `latest_only` if the action vector contains a
`kind: binary` or `kind: quaternion` slice, since every other option
interpolates linearly and LeRobot does not read `kind`.

## Human in the loop

`hil_launch.py` runs the manager, a policy runner, an optional reward
classifier, and the recorder together, muxing teleop intervention against policy
control. Declare the leader device in the contract's
[`teleop:` section](../reference/contract.md#teleop) first.

```bash
ros2 launch rosetta hil_launch.py contract_path:=/path/to/contract.yaml

ros2 action send_goal /manage_episode \
    rosetta_interfaces/action/ManageEpisode "{prompt: 'task description'}"

ros2 service call /hil_manager/set_intervention std_srvs/srv/SetBool "{data: true}"
ros2 service call /hil_manager/end_episode std_srvs/srv/SetBool "{data: false}"
```

The result reports how the episode ended (`termination_reason`) and whether the
robot did the task (`outcome`), independently. Nothing here deletes a bag.
