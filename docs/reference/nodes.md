# Nodes

The `rosetta_ros` package installs four executables. Three are nodes and are
described on this page: `episode_recorder_node`, `policy_runner_node` and
`episode_keyboard_node`. The fourth, `rosetta_port`, ports bags into datasets
and has its own page, [rosetta_port](rosetta-port.md). The human-in-the-loop
manager, `hil_manager_node`, is in
[`rosetta_ros_hil`](https://github.com/iblnkn/rosetta/tree/main/robots/ros/rosetta_ros_hil).
The message, service and action fields are on [Interfaces](interfaces.md).

Every launch file prints its arguments and defaults:

```bash
ros2 launch rosetta_ros episode_recorder_launch.py --show-args
```

## Parameter layering

`episode_recorder_launch.py` and `policy_runner_launch.py` build the node's
parameters from three layers. A later layer wins.

1. The package params file, `params/episode_recorder.yaml` or
   `params/policy_runner.yaml`.
2. The file given as `params_file`. It only needs the keys it changes.
3. The launch arguments. An empty launch argument keeps the value from the
   files. `contract_path` is always passed.

A params file can hold sections keyed `/**`, `/**/<node>`, `<node>` (or
`/<node>`) and `/<namespace>/<node>`. Every section that names the node
applies, merged in that order, so the more specific key wins. A file that has
sections but none for the node fails the launch with `no section applies to
node`. The layers are passed as dictionaries, so both launch files also work
when another launch file includes them under a namespace.

`ros2 launch` rejects an empty value (`contract_path:=` is a malformed launch
argument), so these launch files always load a contract file. The checkpoint
fallback for an empty `contract_path` is only reachable with `ros2 run`.

## Lifecycle

`episode_recorder_node` and `policy_runner_node` are lifecycle nodes.
`episode_keyboard_node` is a plain node. Both lifecycle launch files take
`configure` and `activate` (both default `true`), which drive those
transitions at start.

| Transition | Effect |
|---|---|
| `configure` | Load the contract and create subscriptions, inactive publishers, the action server and the services. On any error the node tears down what it built and returns to `unconfigured`. If that teardown also fails, the node is finalized. |
| `activate` | Activate publishers, then accept work. |
| `deactivate` | Stop accepting work, stop work in progress (waiting up to 5 s), publish the safety action, then deactivate publishers. The recorder has no safety action. |
| `cleanup` | Fails while work is in progress. Otherwise releases everything `configure` created. |
| `shutdown` | Stop, wait, publish the safety action, release. |

Ctrl+C runs the `shutdown` transition before the process exits, so an open bag
is closed and the safety action is published.

A node runs one piece of work at a time, started by an action goal or by its
start service. Work requested while busy or while the node is not active is
rejected. A rejected start service returns `accepted: false` with `message:
Rejected: already busy` or `Rejected: node not active`.

The goal's terminal state follows its `termination_reason`:

| `termination_reason` | Goal state |
|---|---|
| any, after a client cancel | `CANCELED` |
| `error`, `node_deactivated` | `ABORTED` |
| `stopped`, `timeout`, `completed` | `SUCCEEDED` |

Each node has a `~/start_*` service that starts the same work without a goal,
and a `~/cancel_*` service. For goal-started work the cancel service sends a
cancel to the node's own action server, so the goal ends `CANCELED` exactly as
if an action client had cancelled it. For service-started work it stops the
work with `termination_reason: cancelled`. A cancel service with nothing
running returns `success: false`.

Work started by a service has no goal, so it has no action feedback or result.
Its progress is on `~/status` and its result on `~/result`, which every node
publishes however the work started.

## episode_recorder_node

Records bags. Node name `episode_recorder`. Launch file
`episode_recorder_launch.py`.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `contract_path` | string | `""` | Contract file. Required: configure fails with `contract_path parameter required` when empty. The launch file defaults to `contracts/so_101.yaml`. Read-only. |
| `bag_base_dir` | string | `datasets/bags` | Directory for bags. Relative paths resolve against the current directory. `~` expands. Created at configure. Read-only. |
| `storage_id` | string | `mcap` | rosbag2 storage plugin, passed unchecked. Read-only. |
| `record_all` | bool | `true` | Also record every other topic on the graph, discovered at each recording start. Read-only. |
| `exclude_topics` | string[] | `[""]` (none) | Regexes, `ros2 bag record --exclude` syntax. Discovered topics that match are skipped. Read-only. |
| `include_topics` | string[] | `[""]` (none) | Regexes. Discovered topics that match are recorded despite `exclude_topics` and the camera rule. Read-only. |
| `default_prompt` | string | `""` | Prompt used when a goal or service call leaves `prompt` empty. Read at each recording start. |
| `default_max_duration_s` | double | `0.0` | Seconds. Used when the goal's `max_duration_s` is zero or less. Zero or less records until stopped. Read at each recording start. |
| `feedback_rate_hz` | double | `2.0` | Hz, 0.1 to 1000. Rate of action feedback and `~/status`. Read at each recording start. |
| `embed_contract` | bool | `true` | Copy the contract file's text, comments included, into each bag's `metadata.yaml`. Read-only. |
| `use_sim_time` | bool | `false` | Stamp messages with sim time and record `/clock` as a contract topic. |

Launch arguments: `params_file`, `contract_path`, `bag_base_dir`,
`storage_id`, `use_sim_time`, `log_level`, `configure`, `activate`.
`params/episode_recorder.yaml` sets the same values as the defaults in the
table.

| Name | Kind | Type |
|---|---|---|
| `record_episode` | action server | `rosetta_interfaces/action/RecordEpisode` |
| `~/start_recording` | service | `rosetta_interfaces/srv/StartRecording` |
| `~/cancel_recording` | service | `std_srvs/srv/Trigger` |
| `~/delete_last_bag` | service | `std_srvs/srv/Trigger` |
| `~/status` | publisher | `rosetta_interfaces/msg/RecordEpisodeStatus`, reliable, depth 10 |
| `~/result` | publisher | `rosetta_interfaces/msg/RecordEpisodeResult`, reliable, transient-local, depth 1 |
| contract topics | subscriptions | The types the contract declares, with its QoS |
| discovered topics | subscriptions | With `record_all`, per recording |

With the default namespace these are `/record_episode` and
`/episode_recorder/...`.

`~/delete_last_bag` deletes the last recording's bag directory, including a
failed partial bag. Called during a recording, it marks that recording for
deletion when it ends. It returns `success: false` with `No bag to delete`
when there is none. Starting a new recording makes the previous bag no longer
deletable.

`~/status` carries the feedback fields at `feedback_rate_hz` while a recording
runs. `~/result` carries the result fields once per recording, and a late
subscriber gets the last one.

### Recorded topics

- Contract topics are every source in every frame section, the `teleop`
  `input` and `feedback` channels, the `tasks` and `adjunct` channels, and
  `/clock` under `use_sim_time`. `teleop.events` is not a contract topic. A
  topic declared twice with different types or QoS fails configure.
- The recorder subscribes to contract topics at configure, so a recording
  captures them from its first instant. A type that does not import fails
  configure. Transient-local contract topics, such as `/tf_static`, are the
  exception: they are subscribed at each recording start instead, so their
  latched messages land in every bag, and a type that does not import fails
  that recording.
- A contract topic with no messages is marked `(!)` in the per-topic summary
  logged when a recording ends. Nothing stops a recording from starting
  without it.
- With `record_all`, topics on the graph that are not contract topics are
  discovered at each recording start. `/rosout` and `/parameter_events` are
  skipped unless `include_topics` matches them. A topic whose type does not import is skipped with a
  warning. The subscription QoS is reliable or transient-local only when
  every publisher offers it.
- Each camera records one stream. For a raw `sensor_msgs/msg/Image` topic and
  its `image_transport` re-encodings, a stream named in the contract or
  matching `include_topics` wins. Otherwise the first stream not excluded, in
  the order `/compressed`, `/zstd`, `/theora`, `/compressedDepth`, raw.

### Bags

- Each recording is one bag directory, `<bag_base_dir>/<seconds>_<nanoseconds>`,
  named from the wall clock at start and zero-padded to ten and nine digits.
- The bag timestamp of each message is the node clock at receipt. Under
  `use_sim_time` that is sim time. The message's own `header.stamp` is
  unchanged.
- After the bag closes, `metadata.yaml` gets `custom_data` keys:
  `lerobot.operator_prompt` when the prompt is non-empty,
  `rosetta.contract_yaml` with `embed_contract`, and `rosetta.goal_id` (the
  goal UUID) for goal-started recordings.
- A failed bag write ends the recording with `termination_reason: error` and
  the failure in `message`.
- `termination_reason` is one of `stopped`, `timeout`, `cancelled`,
  `node_deactivated`, `error`. `bag_path` is set on every path, including
  errors.

## episode_keyboard_node

Drives the recorder's services from a terminal. Node name `episode_keyboard`.
Launch file `episode_keyboard_launch.py`. It needs an interactive terminal.
Without one it exits with `Failed to enable raw terminal mode. Are you running
in an interactive terminal?`

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `recorder_ns` | string | `/episode_recorder` | Prefix of the recorder's services. The keyboard calls `<recorder_ns>/start_recording`, `<recorder_ns>/cancel_recording` and `<recorder_ns>/delete_last_bag`. |
| `default_prompt` | string | `""` | Initial prompt for recordings. The `t` key changes it. |

Launch arguments: `recorder_ns`, `default_prompt`.

| Key | Action |
|---|---|
| `r`, right arrow | Call `start_recording` with the current prompt. |
| `s`, left arrow | Call `cancel_recording`. The bag is kept, and the recording ends with `termination_reason: cancelled`. |
| `d`, Backspace | Call `cancel_recording` if recording, then `delete_last_bag`. |
| `t` | Edit the prompt. Enter applies, Esc cancels. |
| `h`, `?` | Print the key help. |
| `q` | Quit. |

Letter keys also work in upper case. Beyond the topics and services every
rclpy node has, the node has only the three service clients.

## policy_runner_node

Runs a policy on the live robot through a framework adapter. Node name
`policy_runner`. Launch file `policy_runner_launch.py`.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `contract_path` | string | `""` | Contract file. Empty resolves it from `pretrained_name_or_path`, as in [Contract fallback](#contract-fallback). The launch argument defaults to empty, which keeps the params-file value. Read-only. |
| `pretrained_name_or_path` | string | `""` | Checkpoint, a local path or a Hugging Face Hub repo ID. Read by the adapter and by the contract fallback. Read-only. |
| `framework` | string | `lerobot` | Adapter, by its name in the `rosetta.policy_runners` entry-point group. Read-only. |
| `is_classifier` | bool | `false` | Publish the contract's `rewards` section as the output instead of `actions`. Read-only. |
| `default_prompt` | string | `""` | Task used when a goal or service call leaves `prompt` empty. Read at each run start. |
| `default_max_duration_s` | double | `0.0` | Seconds. Used when the goal's `max_duration_s` is zero or less. Zero or less runs until stopped. Read at each run start. |
| `feedback_rate_hz` | double | `2.0` | Hz, 0.1 to 1000. Rate of action feedback and `~/status`. Read at each run start. |
| `use_sim_time` | bool | `false` | Use the `/clock` time. The LeRobot adapter then paces observations and actions on it at the contract `fps`. |

Launch arguments: `params_file`, `contract_path`,
`pretrained_name_or_path`, `policy_type`, `policy_device`, `server_address`,
`launch_local_server`, `use_sim_time`, `output_prefix`, `log_level`,
`configure`, `activate`.

| Name | Kind | Type |
|---|---|---|
| `run_policy` | action server | `rosetta_interfaces/action/RunPolicy` |
| `~/start_policy` | service | `rosetta_interfaces/srv/StartPolicy` |
| `~/cancel_policy` | service | `std_srvs/srv/Trigger` |
| `~/status` | publisher | `rosetta_interfaces/msg/RunPolicyStatus`, reliable, depth 10 |
| `~/result` | publisher | `rosetta_interfaces/msg/RunPolicyResult`, reliable, transient-local, depth 1 |
| action topics | publishers | The contract's `actions` topics, or its `rewards` topics with `is_classifier` |
| observation topics | subscriptions | The contract's `observations` topics |

With the default namespace these are `/run_policy` and `/policy_runner/...`.
Under `rosetta_ros_hil`'s `hil_launch.py` the runner is in namespace
`robot_policy`, so the action is `/robot_policy/run_policy`.

`output_prefix` publishes every contract action and reward topic under a
prefix: with `so_101.yaml`, `output_prefix:=/policy` publishes
`/leader_arm/joint_states` as `/policy/leader_arm/joint_states`. The launch
file reads the topics from `contract_path`, so a prefix needs a contract file.
[Route commands through a mux](../how-to/route-commands.md) covers the uses.

While the node is active, a watchdog publishes each action channel's `safety`
command (`zeros`, `hold`, or nothing for `none`) when no action has been
published for two frame periods on the node clock. It disarms after firing
until the next action. A contract whose actions are all `safety: none` has no
watchdog.

The run ends with `termination_reason`:

- `completed` when the adapter's run ends on its own without error.
- `timeout`, `cancelled` or `node_deactivated` when the node stopped it for
  that reason.
- `error` when the adapter raised or reported a failure, with the detail in
  `message`.

### Contract fallback

With an empty `contract_path`, configure follows
`pretrained_name_or_path` to its `train_config.json`, then to the training
dataset's `root` or `repo_id`, then to `meta/rosetta_contract.yaml` there.
A link that names an existing local directory is read from it. Any other
link is fetched from the Hub. A missing link fails configure, and a Hub miss
also logs a warning naming it. Each inline `decoder:` or
`encoder:` in a contract found this way logs a warning before the contract
loads, because loading it imports that code. A non-empty `contract_path` is
used as given and never compared with the checkpoint's contract.

### LeRobot adapter parameters

The `lerobot` adapter declares these at configure, so `ros2 param list` shows
them only on a configured node. A value from the launch layers takes effect
when the adapter declares the parameter. The second column is the adapter's
default and the third is what `params/policy_runner.yaml` sets.

| Parameter | Adapter default | Params file | Meaning |
|---|---|---|---|
| `pretrained_name_or_path` | `""` | `iblnk/act-turtlebot3_demo` | Checkpoint. Empty starts the server without a model, which then loads on the first run. With `is_classifier`, a `reward_classifier` checkpoint. |
| `policy_type` | `""` | `""` | Policy type sent to the server, one of LeRobot's `SUPPORTED_POLICIES` (0.6.1: `act`, `smolvla`, `diffusion`, `tdmpc`, `vqbet`, `pi0`, `pi05`, `groot`). Empty reads it at configure from the checkpoint's `config.json`, or uses `reward_classifier` with `is_classifier`. A checkpoint the runner cannot read fails configure, so set it when the checkpoint exists only on a remote server. The Rosetta servers load the type the checkpoint records and ignore this value. Read-only. |
| `policy_device` | `cuda` | `cuda` | Torch device, such as `cuda`, `cuda:1`, `mps`, `xpu`, `cpu`. Sent to the server unchanged. The server resolves it on its own machine: a backend that machine lacks falls back to `cpu` with a warning in the server log, and a name LeRobot does not know is passed through. Empty fails each run with `policy_device cannot be empty`. Read-only. |
| `server_address` | `127.0.0.1:8080` | `127.0.0.1:8080` | `host:port` of the policy server. Read-only. |
| `launch_local_server` | `true` | `true` | Start a policy server on `server_address` at configure. Read-only. |
| `server_startup_timeout_sec` | `120.0` | `120.0` | Seconds configure waits for the local server to accept connections. The server loads the model before it listens, so this covers the model load and any download. Read-only. |
| `actions_per_chunk` | `50` | `30` | Actions requested per inference. Read at each run start. |
| `chunk_size_threshold` | `0.5` | `0.95` | Fraction, 0 to 1. Queue fill at which the next observation is sent. Read at each run start. |
| `aggregate_fn_name` | `weighted_average` | `weighted_average` | How an overlapping chunk merges with the queue: `weighted_average` (0.3 old, 0.7 new), `latest_only`, `average`, `conservative` (0.7 old, 0.3 new). Read at each run start. |

The adapter refuses a contract at configure when it has more than one
numeric observation key, more than one action key, or a numeric stream
without `select`. LeRobot's live path merges all numeric values into one
`observation.state` and one `action`, and names them from `select`.

The local server runs `lerobot_rosetta.policy_server`, or
`lerobot_rosetta.classifier_server` with `is_classifier`. Its output goes to a
`rosetta_policy_server_*.log` file in the system temp directory, and a server
that exits during startup fails configure with the end of that log. A server
that died is started again at the next run. Cleanup and shutdown stop it.

## Nodes inside the LeRobot plugins

The LeRobot `Robot` and `Teleoperator` plugins start their own lifecycle
nodes when a LeRobot tool connects them:

| Plugin | Node name |
|---|---|
| `lerobot_robot_rosetta` (`--robot.type=rosetta`) | `rosetta_<id>`. The id defaults to the contract's `robot_type`. |
| `lerobot_teleoperator_rosetta` (`--teleop.type=rosetta_teleop`) | `rosetta_teleop_<id>`. The id defaults to `<robot_type>_teleop`. |

The robot plugin's node subscribes to the observation topics and publishes the
action topics. The teleop plugin's node subscribes to the `teleop` input and
events topics and publishes the feedback topics. Neither node has Rosetta
services or actions, only the standard lifecycle, parameter and logger
services. Inside `policy_runner_node`, the robot
plugin uses the runner's node and starts none of its own.
