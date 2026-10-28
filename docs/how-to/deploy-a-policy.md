# Deploy a policy

Run a trained checkpoint on the robot with the policy runner. Its parameters
are under [policy_runner_node](../reference/nodes.md#policy_runner_node).

You need a checkpoint (see [Train a policy](train.md)), the contract its
dataset was ported with, and the robot or simulator publishing the contract's
observation topics.

## Start the runner

```bash
ros2 launch rosetta_ros policy_runner_launch.py \
    contract_path:=robot.yaml \
    pretrained_name_or_path:=outputs/train/my_policy/checkpoints/last/pretrained_model \
    policy_type:=act
```

- `contract_path` is the contract to run on. Leave it out to use the
  contract the checkpoint was trained with, as in
  [Use the contract from training](#use-the-contract-from-training).
- `pretrained_name_or_path` is a local checkpoint directory or a Hugging Face
  Hub model id. The package params file sets a demo checkpoint, so always pass
  it too.
- `policy_type` has to match the `--policy.type` the checkpoint was trained
  with.
- Add `use_sim_time:=true` for a simulator.

The launch file configures and activates the node. Configure starts a local
policy server and waits for it to load the model, for up to
`server_startup_timeout_sec` (120 s by default). Raise it in a params file if
the first launch has to download weights from the Hub. `policy_device` is
`cuda` by default. The server falls back to `cpu`, with a warning in its log,
when its machine has no CUDA.

With LeRobot 0.6.1, every run logs
`obs_similarity_atol not supported in this LeRobot version; ignoring.`
LeRobot's client has no such option, so the parameter has no effect.

## Start and stop a run

In a second terminal, start a run with the prompt the episodes were recorded
with:

```bash
ros2 action send_goal --feedback /run_policy \
    rosetta_interfaces/action/RunPolicy "{prompt: 'place cubes on tray'}"
```

Feedback reports `published_actions` and `queue_depth`. A run lasts until
it's cancelled, unless the goal sets `max_duration_s` or the node sets
`default_max_duration_s`.

Ctrl+C on `ros2 action send_goal` cancels the run. The client may print
`Executor is already spinning` as it exits. The cancel still reaches the
runner, which stops and publishes each action channel's `safety` value (see
[Set the action's safety](write-a-contract.md#set-the-actions-safety)).

## Use the contract from training

`rosetta_port` copies the contract into the dataset as
`meta/rosetta_contract.yaml`. Leave `contract_path` out and the runner finds
that file through the checkpoint's `train_config.json`, so it runs on exactly
the contract the policy was trained with:

```bash
ros2 launch rosetta_ros policy_runner_launch.py \
    pretrained_name_or_path:=outputs/train/my_policy/checkpoints/last/pretrained_model \
    policy_type:=act
```

The node logs `Using model-embedded contract (via dataset chain):` with the
path it found. If it finds none, configure fails with `contract_path parameter
required`. [Contract fallback](../reference/nodes.md#contract-fallback) lists
each link it follows.

A `contract_path` you pass is used as given and isn't compared with the
checkpoint's dataset.

## Run inference on another machine

The policy server can run on a GPU machine while the runner stays on the
robot. The server imports only LeRobot, so the GPU machine doesn't need ROS 2.
It needs LeRobot with its `async` extra, plus any extra the policy type
needs, and the `policies/lerobot/lerobot_rosetta` directory of a Rosetta clone
on `PYTHONPATH`. On the GPU machine:

```bash
PYTHONPATH=rosetta/policies/lerobot/lerobot_rosetta python3 -m lerobot_rosetta.policy_server \
    --host=0.0.0.0 --port=8080 \
    --policy-type=act \
    --pretrained-name-or-path=<checkpoint> \
    --policy-device=cuda
```

`--policy-type`, `--pretrained-name-or-path` and `--policy-device` preload the
model before the server accepts connections. They go together. Without them
the server loads the model when the runner first connects.

On the robot, point the runner at the server and name the same checkpoint:

```bash
ros2 launch rosetta_ros policy_runner_launch.py \
    contract_path:=robot.yaml \
    pretrained_name_or_path:=<checkpoint> \
    policy_type:=act \
    launch_local_server:=false \
    server_address:=<gpu_host>:8080
```

At the start of each run the runner sends its own `policy_type`,
`pretrained_name_or_path` and `policy_device` to the server, and the server
loads that model if it differs from the one it holds. So `<checkpoint>` has to
be a path that exists on the GPU machine, or a Hub repo ID. The server
resolves `policy_device` on its own machine, so a robot without CUDA can still
ask for `cuda`.

The server listens on an unencrypted gRPC port and unpickles what clients send.
Only expose it on a network you trust.

## Tune the chunking

The runner asks the server for a chunk of actions at a time and queues them.
To change how, put the parameters in a params file under a `policy_runner`
section. It only needs the keys it changes:

```yaml
policy_runner:
  ros__parameters:
    actions_per_chunk: 20
    aggregate_fn_name: latest_only
```

Pass it with `params_file:=tuning.yaml`. A file with no section that applies
to `policy_runner` fails the launch.

| Parameter | Launch default | Meaning |
|---|---|---|
| `actions_per_chunk` | `30` | Actions requested per inference. More means fewer calls and a less reactive robot. |
| `chunk_size_threshold` | `0.95` | The next observation goes out once the queue holds this fraction of a chunk or less. Range 0 to 1. |
| `aggregate_fn_name` | `weighted_average` | How a new chunk combines with queued actions for the same steps: `weighted_average` (0.3 old, 0.7 new), `average`, `conservative` (0.7 old, 0.3 new), or `latest_only`. |

Use `latest_only` if the action vector has a `binary` or `quaternion` slice.
The other options blend old and new values, which turns an open or closed
gripper command into something in between and a unit quaternion into one that
isn't. The defaults in the table come from the package params file that the
launch file loads. Running the node without it uses the adapter's own defaults
(see [LeRobot adapter parameters](../reference/nodes.md#lerobot-adapter-parameters)).

## Run with LeRobot's own tools

With `lerobot_robot_rosetta` installed, LeRobot's CLIs list the robot as
`--robot.type=rosetta`, configured by the contract:

```bash
lerobot-record --robot.type=rosetta --robot.config_path=robot.yaml ...
lerobot-replay --robot.type=rosetta --robot.config_path=robot.yaml ...
```

The plugin creates a ROS 2 node, so ROS 2 has to be installed and sourced. On
connect it applies the same contract limits as the runner (one numeric
observation key, one action key, a `select` on every numeric source), then
waits up to 5 s for every observation stream to deliver. After 5 s it carries
on with a warning, and the first frames may be zero-filled.
