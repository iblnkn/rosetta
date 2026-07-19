# Nodes

All nodes read parameter files (`params/`) as defaults. All parameters are also exposed as launch arguments, which override the defaults. Run `ros2 launch rosetta <launch_file> --show-args` to see all options.

## episode_recorder_node

Records contract-specified topics to rosbag2. Launch: `episode_recorder_launch.py`. Params: `params/episode_recorder.yaml`.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `contract_path` | `contracts/so_101.yaml` | Path to contract YAML (launch-arg default, required at the node level) |
| `bag_base_dir` | `datasets/bags` | Rosbag output dir. Relative paths resolve against the launch cwd (like `ros2 bag record`) |
| `storage_id` | `mcap` | Rosbag format: `mcap` (recommended) or `sqlite3` |
| `default_max_duration` | `0.0` | Max episode duration in seconds. `0.0` records until stopped |
| `feedback_rate_hz` | `2.0` | Recording feedback publish rate |
| `record_all` | `true` | Record every topic on the graph, not only contract topics |
| `exclude_topics` | `[]` | Regex list of topics to skip when `record_all` is on |
| `embed_contract` | `true` | Embed the exact contract text into bag metadata |
| `log_level` | `info` | Logging level: `debug`, `info`, `warn`, `error` (launch argument) |
| `configure` | `true` | Auto-configure on startup (launch argument) |
| `activate` | `true` | Auto-activate on startup (launch argument) |

### Interface

| Name | Type | Purpose |
|------|------|---------|
| `record_episode` (action) | `rosetta_interfaces/action/RecordEpisode` | Start an episode. `prompt` field sets the task label |
| `~/cancel_recording` (service) | `std_srvs/srv/Trigger` | Stop and save the running episode |
| `~/start_recording` (service) | `rosetta_interfaces/srv/StartRecording` | Start for callers without action support |
| `~/delete_last_bag` (service) | `std_srvs/srv/Trigger` | Remove the most recently saved bag |

### Topic recording

By default, the recorder records **every topic** on the ROS2 graph, not only those declared in the contract, equivalent to `ros2 bag record -a`. Contract topics (observations, actions, etc.) must be present. Only `/rosout` and `/parameter_events` are excluded automatically. `exclude_topics` (regex list) excludes more. `record_all: false` records only contract-declared topics. Rationale: [Record raw, decide late](../explanation/record-raw-decide-late.md).

## episode_keyboard_node

Keyboard control for the recorder. Launch: `episode_keyboard_launch.py`.

| Key | Action |
|-----|--------|
| `r` / `→` | Start recording |
| `s` / `←` | Stop and save |
| `d` / `⌫` | Discard episode (stop + delete bag) |
| `t` | Edit task prompt for the next episode |
| `h` / `?` | Help |
| `q` | Quit |

| Launch argument | Default | Description |
|----------|---------|-------------|
| `recorder_ns` | `/episode_recorder` | Namespace of the recorder node |
| `default_prompt` | `` | Initial task prompt used when starting recordings |

## policy_runner_node

Wraps a policy framework's inference pipeline in ROS2 actions. Launch: `policy_runner_launch.py`. Params: `params/policy_runner.yaml`.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `contract_path` | `contracts/so_101.yaml` | Path to contract YAML. Optional when the checkpoint's training dataset embeds one |
| `pretrained_name_or_path` | *(see params file)* | HuggingFace model ID or local path |
| `framework` | `lerobot` | Policy framework adapter, resolved by entry-point name |
| `is_classifier` | `false` | Publish the reward section as the action output (HIL reward classifier) |
| `server_address` | `127.0.0.1:8080` | Policy server address |
| `policy_type` | `act` | Policy type: `act`, `smolvla`, `diffusion`, `pi0`, `pi05`, etc. |
| `policy_device` | `cuda` | Inference device: `cuda`, `cpu`, `mps`, or `cuda:0` |
| `actions_per_chunk` | `30` | Actions per inference chunk |
| `chunk_size_threshold` | `0.95` | When to request new chunk (0.0-1.0) |
| `aggregate_fn_name` | `weighted_average` | Chunk aggregation: `weighted_average`, `latest_only`, `average`, `conservative` |
| `feedback_rate_hz` | `2.0` | Execution feedback publish rate |
| `launch_local_server` | `true` | Auto-start policy server subprocess (at configure, with model preload) |
| `server_startup_timeout_sec` | `120.0` | Max wait for the server to come up (covers model preload, raise for cold HF downloads) |
| `obs_similarity_atol` | `-1.0` | Observation filtering tolerance (-1.0 to disable)* |
| `log_level` | `info` | Logging level: `debug`, `info`, `warn`, `error` (launch argument) |
| `configure` | `true` | Auto-configure on startup (launch argument) |
| `activate` | `true` | Auto-activate on startup (launch argument) |

### Interface

| Name | Type | Purpose |
|------|------|---------|
| `run_policy` (action) | `rosetta_interfaces/action/RunPolicy` | Run the policy. `prompt` field sets the task |

The action's relative name is `run_policy`. A launch-file namespace prefixes the full name, e.g. `/robot_policy/run_policy` in the HIL launch.

### Contract resolution

When `contract_path` is empty, the node resolves the contract from the checkpoint: `pretrained_name_or_path` → `train_config.json` → the training dataset (local path first, then the Hub) → the dataset's embedded `meta/rosetta_contract.yaml`. Datasets ported with the default `embed_contract` settings deploy with no separate contract file. If no link in the chain resolves, the node errors and asks for `contract_path`.

## hil_manager_node

Orchestrates human-in-the-loop episodes. Launch: `hil_launch.py`. Params: `params/hil_manager.yaml` (the "super YAML" covering all four HIL nodes).

### Interface

| Name | Type | Purpose |
|------|------|---------|
| `manage_episode` (action) | `rosetta_interfaces/action/ManageEpisode` | Run a managed episode end to end |
| `~/start_episode` (service) | `rosetta_interfaces/srv/StartHILEpisode` | Start for callers without action support |
| `~/stop_episode` (service) | `std_srvs/srv/Trigger` | Stop the running episode |
| `~/set_intervention` (service) | `std_srvs/srv/SetBool` | Toggle teleop intervention mux |
| `~/set_reward_override` (service) | `std_srvs/srv/SetBool` | Force the reward signal |
| `~/clear_reward_override` (service) | `std_srvs/srv/Trigger` | Clear the forced reward |

See [Set up teleop and HIL](../how-to/teleop-and-hil.md) for the launch topology.

## ROS2 lifecycle

All Rosetta nodes are lifecycle nodes:

| Transition | Effect |
|------------|--------|
| `configure` | Create subscriptions (start buffering), create publishers (disabled) |
| `activate` | Enable publishers, start watchdog |
| `deactivate` → `cleanup` | Safety action, disable publishers, destroy resources |
