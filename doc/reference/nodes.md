# Nodes

Both Rosetta nodes read parameter files (`params/`) as defaults. A launch file
exposes the deployment-specific subset as launch arguments (paths, storage
format, log level, lifecycle autostart); everything else is set in the params
YAML. Run `ros2 launch rosetta <launch_file> --show-args` to see the options.

## episode_recorder_node

Records contract-specified topics to rosbag2. Launch: `episode_recorder_launch.py`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `contract_path` | `contracts/so_101.yaml` | Path to contract YAML (launch arg) |
| `bag_base_dir` | `datasets/bags` | Directory for rosbag output, relative to the launch cwd (launch arg) |
| `storage_id` | `mcap` | Rosbag format: `mcap` (recommended) or `sqlite3` (launch arg) |
| `default_prompt` | `""` | Task label used when a goal leaves `prompt` empty |
| `default_max_duration_s` | `0.0` | Max episode duration. `0.0` records until stopped |
| `feedback_rate_hz` | `2.0` | Recording feedback publish rate |
| `record_all` | `true` | Record every topic on the graph, not only contract topics |
| `exclude_topics` | `[]` | Regex list of topics to skip when `record_all` is on |
| `include_topics` | `[]` | Regex list to always record, overriding `exclude_topics` |
| `embed_contract` | `true` | Embed the contract text into bag metadata |

Actions and services: `record_episode` (action), `~/start_recording`,
`~/cancel_recording`, `~/delete_last_bag`.

### Topic recording

By default the recorder records **every topic** on the ROS2 graph, not just
those declared in the contract, so you never lose data you might need later.
This behaves like `ros2 bag record -a`. Contract topics are required to be
present. Only `/rosout` and `/parameter_events` are excluded automatically;
`exclude_topics` excludes more, and `record_all: false` records only
contract-declared topics.

Cameras are the exception: per camera the recorder keeps one `image_transport`
stream, preferring `/compressed` > `/zstd` > `/theora` > `/compressedDepth` >
raw. `image_transport` republishers encode only while subscribed, so recording
all of them makes the camera node encode every frame several ways at once.

## episode_keyboard_node

Keyboard control for the recorder. Launch: `episode_keyboard_launch.py`, with
`recorder_ns` (default `/episode_recorder`) and `default_prompt` arguments.

| Key | Action |
|-----|--------|
| `r` / `→` | Start recording |
| `s` / `←` | Stop and save |
| `d` / `⌫` | Discard episode (stop + delete bag) |
| `t` | Edit task prompt for the next episode |
| `h` / `?` | Help |
| `q` | Quit |

## policy_runner_node

Wraps a policy framework's inference pipeline in ROS2 actions. Launch:
`policy_runner_launch.py`. The first block is declared by the node; the second
by the resolved `framework` adapter, `lerobot_rosetta` here.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `contract_path` | `contracts/so_101.yaml` | Optional when the checkpoint's dataset embeds one (launch arg) |
| `framework` | `lerobot` | Policy framework adapter, resolved by entry-point name |
| `is_classifier` | `false` | Publish the reward section as the action output |
| `default_prompt` | `""` | Task used when a goal leaves `prompt` empty |
| `default_max_duration_s` | `0.0` | Max run duration. `0.0` runs until stopped |
| `feedback_rate_hz` | `2.0` | Execution feedback publish rate |

| LeRobot adapter parameter | Default | Description |
|-----------|---------|-------------|
| `pretrained_name_or_path` | *(see params file)* | HuggingFace model ID or local path (launch arg) |
| `server_address` | `127.0.0.1:8080` | Policy server address (launch arg) |
| `policy_type` | `act` | `act`, `smolvla`, `diffusion`, `pi0`, `pi05`, etc. (launch arg) |
| `policy_device` | `cuda` | `cuda`, `xpu`, `mps`, `cpu`, or `cuda:0` (falls back to `cpu` if unavailable) |
| `actions_per_chunk` | `30` | Actions per inference chunk |
| `chunk_size_threshold` | `0.95` | When to request a new chunk (0.0-1.0) |
| `aggregate_fn_name` | `weighted_average` | `weighted_average`, `latest_only`, `average`, `conservative` |
| `launch_local_server` | `true` | Auto-start the policy server at configure, with model preload (launch arg) |
| `server_startup_timeout_sec` | `120.0` | Max wait for the server, covering model preload |
| `obs_similarity_atol` | `-1.0` | Observation filtering tolerance. Ignored by stock LeRobot v0.6.0, where the filter is hardcoded on |
| `sim_time_multiplier` | `1.0` | Scales contract `fps` before handing it to LeRobot, which paces on wall time |

Actions and services: `run_policy` (action), `~/start_policy`,
`~/cancel_policy`. A launch namespace prefixes the action name, as in
`/robot_policy/run_policy` under `hil_launch.py`.

When `contract_path` is empty, the node resolves the contract from the
checkpoint: `pretrained_name_or_path` → `train_config.json` → the training
dataset → `meta/rosetta_contract.yaml`.

## hil_manager_node

Orchestrates human-in-the-loop episodes. Launch: `hil_launch.py`, which wires
the manager, a policy runner (namespace `robot_policy`), an optional reward
classifier (namespace `reward_classifier`), and the recorder. Params:
`params/hil_manager.yaml` covers all four.

Action: `manage_episode`. Services: `~/start_episode`, `~/end_episode` and
`~/set_intervention` and `~/set_reward_override` (`SetBool`),
`~/cancel_episode` and `~/clear_reward_override` (`Trigger`).

`~/end_episode` is the deliberate, labelled end: the goal succeeds and the
verdict lands in `outcome`. `~/cancel_episode` abandons the take. Neither
deletes a bag.

## ROS2 lifecycle

All Rosetta nodes are lifecycle nodes.

| Transition | Effect |
|------------|--------|
| `configure` | Create subscriptions (start buffering), create publishers (disabled) |
| `activate` | Enable publishers, start watchdog, open goal acceptance |
| `deactivate` → `cleanup` | Safety action, disable publishers, destroy resources |

Goals are accepted only while `active`, one at a time per node. A deactivate
stops in-progress work and ends its goal `ABORTED` with
`termination_reason: node_deactivated`.
