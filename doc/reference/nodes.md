# Nodes

Three executables in the `rosetta` package. The recorder and policy runner
launch files read a params file from `params/` and expose deployment values as
launch arguments. An empty launch argument keeps the params-file value, except
`contract_path`, which is passed as given. The human-in-the-loop manager is a
separate app built on these nodes, in
[`rosetta_hil`](https://github.com/iblnkn/rosetta_hil).

```bash
ros2 launch rosetta <launch_file> --show-args
```

## Lifecycle

`episode_recorder_node` and `policy_runner_node` are lifecycle nodes. `episode_keyboard_node` is a plain node.

| Transition | Effect |
|---|---|
| `configure` | Load the contract, create subscriptions and inactive publishers. An error returns the node to `unconfigured`, or finalizes it if teardown fails too. |
| `activate` | Enable publishers, then accept goals. |
| `deactivate` | Stop accepting goals, stop in-progress work (waits up to 5 s), publish the safety action, disable publishers. |
| `cleanup` | Refused while work is in progress. Otherwise release resources. |
| `shutdown` | Stop, secure, release. |

Both lifecycle launch files take `configure` (default `true`) and
`activate` (default `true`), which drive the transitions at start.

A node runs one goal at a time. A goal sent while busy or while not active is
rejected. Work stopped by deactivate ends `ABORTED` with `termination_reason:
node_deactivated`. A cancelled goal ends `CANCELED`. Each node also exposes a
`~/start_*` service that starts the same work without a goal, and a
`~/cancel_*` service that cancels the running goal the way a client would.
A cancel service with nothing running returns `success: false`.

## episode_recorder_node

Records bags. Node name `episode_recorder`. Launch
`episode_recorder_launch.py`.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `contract_path` | string | `""` | Required. The launch file defaults to `contracts/so_101.yaml`. |
| `bag_base_dir` | string | `datasets/bags` | Relative to the current directory. |
| `storage_id` | string | `mcap` | Passed to rosbag2 unchecked. Any installed storage plugin works. |
| `record_all` | bool | `true` | Record every topic on the graph. |
| `exclude_topics` | string[] | none | Regex list, `ros2 bag record --exclude` syntax. |
| `include_topics` | string[] | none | Regex list. Matches are recorded despite `exclude_topics` and the camera rule. |
| `default_prompt` | string | `""` | Used when a goal or service leaves `prompt` empty. |
| `default_max_duration_s` | double | `0.0` | Zero or less records until stopped. |
| `feedback_rate_hz` | double | `2.0` | 0.1 to 1000. |
| `embed_contract` | bool | `true` | Write the contract text into `metadata.yaml`. |
| `use_sim_time` | bool | `false` | Also records `/clock`. |

Launch arguments: `params_file`, `contract_path`, `bag_base_dir`,
`storage_id`, `use_sim_time`, `log_level`, `configure`, `activate`.

| Interface | Type |
|---|---|
| `record_episode` | action `rosetta_interfaces/action/RecordEpisode` |
| `~/start_recording` | service `rosetta_interfaces/srv/StartRecording` |
| `~/cancel_recording` | service `std_srvs/srv/Trigger` |
| `~/delete_last_bag` | service `std_srvs/srv/Trigger`. Deletes the last bag, including a failed partial bag. During a recording, it deletes that recording once it ends. |

With the default namespace these are `/record_episode` and
`/episode_recorder/...`.

Topic rules:

- Contract topics are every source in every section, plus `tasks` and
  `adjunct` channels, plus `/clock` under `use_sim_time`. The recorder
  subscribes to them at configure. A type that does not import fails
  configure, or the first episode for a transient-local topic. A contract
  topic with no messages at episode end is logged with `(!)`. Nothing stops
  an episode from starting without it.
- With `record_all`, every other topic on the graph is discovered at each
  episode start. A topic is skipped if it matches `exclude_topics`, unless it
  matches `include_topics`. `/rosout` and `/parameter_events` are always
  skipped. QoS is adapted to what every publisher offers.
- One stream per raw `sensor_msgs/msg/Image` topic and its transports, in
  preference `/compressed`, `/zstd`, `/theora`, `/compressedDepth`, raw. A
  transport named in the contract or in `include_topics` wins.
- Transient-local topics such as `/tf_static` are re-subscribed per episode so
  their latched messages land in every bag.
- The bag stamp is the node clock at receipt. Under `use_sim_time` that is sim
  time.
- A write failure ends the episode `ABORTED` with `termination_reason: error`.

Output: `<bag_base_dir>/<seconds>_<nanoseconds>/`, zero-padded to ten and
nine digits. After close, `metadata.yaml` holds `custom_data` keys
`rosetta.contract_yaml` with `embed_contract`, `lerobot.operator_prompt` when
the prompt is non-empty, and `rosetta.goal_id` for goals.

Result `termination_reason`: `stopped`, `timeout`, `cancelled`,
`node_deactivated`, `error`. `bag_path` is set on every path.

## episode_keyboard_node

Drives the recorder from a terminal. Needs a TTY. Node name
`episode_keyboard`. Launch `episode_keyboard_launch.py`.

| Parameter | Default | Meaning |
|---|---|---|
| `recorder_ns` | `/episode_recorder` | Where the recorder's services live. |
| `default_prompt` | `""` | Prompt for the next episode. |

| Key | Calls |
|---|---|
| `r`, right arrow | `start_recording` with the current prompt |
| `s`, left arrow | `cancel_recording`. The bag is kept. |
| `d`, backspace | `cancel_recording` if recording, then `delete_last_bag` |
| `t` | Edit the prompt. Enter applies, Esc cancels. |
| `h`, `?` | Help |
| `q` | Quit |

## policy_runner_node

Runs a policy on the live robot. Node name `policy_runner`. Launch
`policy_runner_launch.py`.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `contract_path` | string | `""` | Empty resolves the contract from the checkpoint. The launch file defaults to `contracts/so_101.yaml`. |
| `framework` | string | `lerobot` | Adapter, by entry-point name under `rosetta.policy_runners`. |
| `is_classifier` | bool | `false` | Serve the `rewards` section as output instead of `actions`. |
| `default_prompt` | string | `""` | |
| `default_max_duration_s` | double | `0.0` | |
| `feedback_rate_hz` | double | `2.0` | |

Contract resolution with an empty `contract_path`: `pretrained_name_or_path`,
then its `train_config.json`, then the training dataset's root or repo id,
then `meta/rosetta_contract.yaml` there. Local paths first, then the Hugging
Face Hub. A missing link is an error. A non-empty `contract_path` is used as
given and never compared with the checkpoint's.

The LeRobot adapter adds these. Defaults in the second column are the
adapter's. `params/policy_runner.yaml` sets the third.

| Parameter | Adapter default | Params file | Meaning |
|---|---|---|---|
| `pretrained_name_or_path` | `""` | `iblnk/act-turtlebot3_demo` | Local path or Hub model id. |
| `policy_type` | `act` | `act` | One of `act`, `smolvla`, `diffusion`, `tdmpc`, `vqbet`, `pi0`, `pi05`. Must match the checkpoint. |
| `policy_device` | `cuda` | `cuda` | Falls back to `cpu` with a warning if the backend is missing. Empty means `cpu`. |
| `server_address` | `127.0.0.1:8080` | same | |
| `launch_local_server` | `true` | `true` | Start `python -m lerobot_rosetta.policy_server` at configure, with the model preloaded when `pretrained_name_or_path` is set. Restarted per run if it died. |
| `server_startup_timeout_sec` | `120.0` | `120.0` | Configure waits this long for the server socket. |
| `actions_per_chunk` | `50` | `30` | Actions returned per inference. |
| `chunk_size_threshold` | `0.5` | `0.95` | Queue fill ratio at which the next observation is sent. |
| `aggregate_fn_name` | `weighted_average` | same | How a new chunk merges with the queue: `weighted_average`, `latest_only`, `average`, `conservative`. |
| `obs_similarity_atol` | `1.0` | `-1.0` | Negative disables. Ignored by stock LeRobot 0.6.0. |

Launch arguments: `params_file`, `contract_path`,
`pretrained_name_or_path`, `policy_type`, `server_address`,
`launch_local_server`, `use_sim_time`, `log_level`, `configure`, `activate`.
`policy_device` is not a launch argument. Set it in the params file.

With `use_sim_time`, observations and actions pace on the sim clock at the
contract `fps`.

| Interface | Type |
|---|---|
| `run_policy` | action `rosetta_interfaces/action/RunPolicy` |
| `~/start_policy` | service `rosetta_interfaces/srv/StartPolicy` |
| `~/cancel_policy` | service `std_srvs/srv/Trigger` |

Under `rosetta_hil`'s `hil_launch.py` the runner sits in namespace
`robot_policy`, so the action is `/robot_policy/run_policy`.
