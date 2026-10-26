# Deploy a policy

Run a trained checkpoint on the robot with the policy runner. Parameters
are under [policy_runner_node](../reference/nodes.md#policy_runner_node).

## Run

```bash
ros2 launch rosetta policy_runner_launch.py \
    contract_path:=robot.yaml \
    pretrained_name_or_path:=outputs/train/my_policy/checkpoints/last/pretrained_model \
    policy_type:=act
```

`pretrained_name_or_path` also accepts a Hugging Face Hub model id. `policy_type` has to
match the checkpoint. Add `use_sim_time:=true` for a simulator.

In a second terminal, start a run with the prompt you recorded with:

```bash
ros2 action send_goal /run_policy \
    rosetta_interfaces/action/RunPolicy "{prompt: 'place cubes on tray'}"
```

Ctrl+C on the client stops the run, and the runner publishes each action
channel's `safety` value.

## Let the checkpoint bring its contract

Leave `contract_path` empty. The runner reads `train_config.json` next to
the checkpoint, finds the training dataset, and loads
`meta/rosetta_contract.yaml` from it. The policy then runs on the contract
it was trained with.

```bash
ros2 launch rosetta policy_runner_launch.py \
    contract_path:= \
    pretrained_name_or_path:=<hf_user>/<model>
```

If you pass `contract_path`, it's used as given. The runner doesn't compare
it with the checkpoint's.

## Run inference on another machine

The policy server doesn't need ROS 2. On the GPU machine:

```bash
python -m lerobot_rosetta.policy_server \
    --host=0.0.0.0 --port=8080 \
    --policy-type=act \
    --pretrained-name-or-path=<checkpoint> \
    --policy-device=cuda
```

`--policy-type`, `--pretrained-name-or-path` and `--policy-device` preload
the model and go together. Without them the server loads the model on the
first request.

On the robot:

```bash
ros2 launch rosetta policy_runner_launch.py \
    contract_path:=robot.yaml \
    launch_local_server:=false \
    server_address:=<gpu-host>:8080
```

## Tune the chunking

Set these in a params file and pass it as `params_file:=`.

`actions_per_chunk` is how many actions come back per inference. More means
fewer calls and a less reactive robot. `chunk_size_threshold` is how empty
the action queue gets before the next observation goes out.
`aggregate_fn_name` is how a new chunk merges with the one still executing.
Use `latest_only` if the action vector has a `binary` or `quaternion` slice,
since the other options interpolate.

## Use LeRobot's own tools instead

With `lerobot_robot_rosetta` installed, LeRobot's CLIs see the robot as
`--robot.type=rosetta`:

```bash
lerobot-record --robot.type=rosetta --robot.config_path=robot.yaml ...
lerobot-replay --robot.type=rosetta --robot.config_path=robot.yaml ...
```

The plugin creates a ROS 2 node, so ROS 2 has to be installed and sourced. On
connect it enforces LeRobot's limit of one numeric observation key and one
action key, then waits up to 5 s for every observation stream to deliver.
