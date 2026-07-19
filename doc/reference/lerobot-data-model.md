# LeRobot Data Model

This page covers LeRobot's internal data model in detail. You do not need this to get started. Refer back here for key conventions, feature types, or policy compatibility.

Facts on this page are verified against LeRobot v0.6.0, the version the workspace pins. Policy configs change between releases.

## Key system

**LeRobot keys are flat dictionary strings using dots as a naming convention.** `observation.state.joint_position` is a single string key, not a nested lookup. The only hard rule is **no forward slashes** (`/`) in key names.

Keys work at any depth:

```python
# These are all valid, independent LeRobot feature keys:
"observation.state"                              # (14,) float64
"observation.state.joint_position"               # (7,)  float32
"observation.state.gripper_position"             # (1,)  float32
"observation.state.imu.orientation"              # (4,)  float64
"observation.environment_state"                  # (25,) float64
"observation.environment_state.object_positions" # (12,) float32
"observation.images.front"                       # (480, 640, 3) video
"observation.images.wrist.left"                  # (480, 640, 3) video
"action"                                         # (8,) float32
"action.arm"                                     # (6,) float32
"action.gripper"                                 # (1,) float32
"action.base"                                    # (2,) float32
```

There is no parent-child relationship between these keys. `observation.state` and `observation.state.joint_position` coexist as independent features with different shapes. They share a prefix, nothing more.

### How LeRobot classifies keys

Keys are free-form strings. `dataset_to_policy_features` (in `lerobot/utils/feature_utils.py`) classifies dataset features into `FeatureType`s, which determine how policies process each feature:

| Rule | FeatureType | How policies use it |
|------|-------------|---------------------|
| dtype is `image` or `video` (any key) | `VISUAL` | Fed through vision encoder |
| key is exactly `observation.environment_state` | `ENV` | Separate encoder projection (privileged sim state) |
| key starts with `observation` (everything else) | `STATE` | Robot state encoder |
| key starts with `action` | `ACTION` | Policy output / training target |
| anything else | dropped | Not passed to the policy |

Note two things. `VISUAL` is decided by dtype, not by key name: `observation.images.*` is convention, not mechanism. And keys matching no rule are silently dropped from the policy's view. The `FeatureType` enum also defines `LANGUAGE` and `REWARD`, but this classification never produces them: language tensors (`observation.language.*`) are created by the tokenizer processor at runtime, not read from the dataset schema.

So `observation.state.imu`, `observation.state.joint_position`, and `observation.state` all classify as `STATE`, and `action.arm` and `action.gripper` both classify as `ACTION`.

### Convention vs. compatibility

LeRobot's key system has two layers:

1. **The dataset format** accepts any key string. `observation.state.fake_sensor.special_data` or `my_custom_thing` both store fine.
2. **Built-in policies** look for specific keys by exact match. ACT, SmolVLA, and Pi0 all expect `observation.state` and `action` as single combined vectors.

The [DROID dataset](https://huggingface.co/datasets/lerobot/droid) demonstrates the recommended approach for richness plus compatibility: **store split sub-keys alongside combined keys**:

```python
# Split sub-keys (rich, self-documenting):
"observation.state.joint_position":     {"shape": (7,)}
"observation.state.cartesian_position": {"shape": (6,)}
"observation.state.gripper_position":   {"shape": (1,)}

# Combined key (policy-compatible):
"observation.state":                    {"shape": (8,)}  # joints + gripper

# Same pattern for actions:
"action.joint_position":    {"shape": (7,)}
"action.gripper_position":  {"shape": (1,)}
"action":                   {"shape": (8,)}  # joints + gripper
```

The sub-keys preserve semantic meaning and enable richer downstream analysis. The combined keys keep existing policies working without modification.

## EnvTransition

LeRobot defines a [Universal Data Container](https://huggingface.co/docs/lerobot/introduction_processors#envtransition-the-universal-data-container) called `EnvTransition`, descended from the classic Gymnasium `step()` return (`observation, reward, terminated, truncated, info`).

The `EnvTransition` TypedDict defines seven top-level slots: `observation`, `action`, `reward`, `done`, `truncated`, `info`, and `complementary_data`. The contract makes the mapping between ROS2 and the EnvTransition's semantic categories explicit. No core policy uses all components.

### Observation (`observation.*`)

Everything the robot senses. Sub-divided by modality:

```
observation.
├── state                           # Robot proprioception (joints, EEF pose)
│   ├── joint_position              #   Optional: split out joint positions
│   ├── cartesian_position          #   Optional: split out EEF pose
│   └── gripper_position            #   Optional: split out gripper
│
├── environment_state               # External/privileged state (sim only)
│   ├── object_positions            #   Optional: sub-key for object poses
│   └── contact_forces              #   Optional: sub-key for forces
│
├── images.                         # Camera feeds (stored as MP4 video)
│   ├── top                         #   Overhead / third-person view
│   ├── front                       #   Front-facing view
│   ├── left / right                #   Side views
│   ├── wrist.left / wrist.right    #   Wrist-mounted cameras
│   └── wrist.top / wrist.bottom    #   Wrist camera orientations
│
└── language                        # Tokenized text (generated by processor)
    ├── tokens                      #   Token IDs (int tensor)
    └── attention_mask              #   Attention mask (bool tensor)
```

**`observation.state`** vs **`observation.environment_state`**: semantically distinct. `state` is the robot's proprioception, what the robot knows about its own body (joint angles, gripper width, EEF pose). `environment_state` is privileged information about the external world (object positions, contact forces), usually available only in simulation. They carry different `FeatureType`s (`STATE` vs `ENV`) and policies encode them with separate projections.

### Action (`action*`)

Motor commands the robot executes:

```
action                              # Combined action vector (policy-compatible)
├── joint_position                  # Optional: split out joint commands
├── cartesian_position              # Optional: split out EEF commands
├── gripper_position                # Optional: split out gripper
├── base                            # Optional: mobile base velocity
└── arm1.fingers                    # Optional: arbitrary depth is allowed
```

Most built-in policies expect a single `action` key. If you split into sub-keys, also provide the combined `action` for compatibility (see the DROID pattern above).

### Task and language

These serve different purposes and coexist:

| Concept | Key(s) | Type | Purpose |
|---------|--------|------|---------|
| **Task string** | `task` | `str` | Human-readable label: `"pick up the red block"` |
| **Language tokens** | `observation.language.tokens` | `Tensor (int)` | Tokenized text for VLA forward pass |
| **Language mask** | `observation.language.attention_mask` | `Tensor (bool)` | Attention mask for tokenized text |

The **flow** between them: the dataset stores a `task_index` (int) per frame, resolved to a `task` string via `meta/tasks.parquet`. How the string reaches the policy depends on the policy:

- **Pre-tokenized** (SmolVLA, Pi0, Pi0Fast, Pi0.5, X-VLA): LeRobot's `TokenizerProcessorStep` reads the `task` string from `complementary_data` and produces `observation.language.tokens` and `observation.language.attention_mask` tensors. The policy consumes these tensors.
- **Internally tokenized** (GR00T, Wall-X): The raw `task` string is passed directly to the policy, which tokenizes it through its own VLM backbone.

`task` is always a single string per frame. `subtask` is a recognized complementary data key.

### Reward and episode signals

RL signals and episode boundaries:

```
next.reward                         # Scalar float: RL reward signal
next.done                           # Bool: episode terminated naturally (goal reached, failure)
next.truncated                      # Bool: episode ended artificially (time limit)
```

These use the `next.` prefix because they describe the outcome *after* taking the action.

### Complementary data

Per-frame metadata flowing through training without being a model input:

```
task                                # Task description string (resolved from task_index)
task_index                          # int64: index into meta/tasks.parquet
episode_index                       # int64: which episode this frame belongs to
frame_index                         # int64: position within the episode
index                               # int64: global frame index
timestamp                           # float32: time in seconds
observation.state_is_pad            # bool tensor: padding flag for state
observation.images.front_is_pad     # bool tensor: padding flag per image key
action_is_pad                       # bool tensor: padding flag for action
```

The `*_is_pad` flags mark which frames in a temporal window are real vs. padded (used when a policy looks at past frames before enough exist).

Every dataset gets the five default features (`timestamp`, `frame_index`, `episode_index`, `index`, `task_index`) automatically. Do not declare them.

### Info

The `info` slot in `EnvTransition` is **runtime-only**, never persisted to datasets. The slot carries transient signals like teleop events used during live recording and policy execution. LeRobot's `TeleopEvents` enum defines `success`, `failure`, `rerecord_episode`, `is_intervention`, and `terminate_episode`. For persistent metadata, use `complementary_data` instead.

Note: `meta/info.json` in the dataset directory is unrelated. That file stores the dataset schema (features, fps, robot_type), not per-frame data.

## Data types

Each feature key maps to a specific data type. LeRobot datasets support:

| Data Type | LeRobot dtype | Shape | Description | Example Keys |
|-----------|--------------|-------|-------------|-------------|
| **Float vector** | `float32` / `float64` | `(N,)` | Continuous values: joints, poses, velocities | `observation.state`, `action` |
| **Image** | `video` | `(H, W, 3)` | RGB uint8 frames, stored as MP4 | `observation.images.*` |
| **String** | `string` | `(1,)` | Text labels, prompts | `task`, `language_instruction` |
| **Boolean** | `bool` | `(1,)` or `(N,)` | Binary flags | `next.done`, `action_is_pad` |
| **Integer** | `int32` / `int64` | `(1,)` or `(N,)` | Discrete values, indices | `task_index`, `episode_index` |

In the Rosetta contract, dtype is usually **auto-detected** from the ROS2 message type:

| ROS2 Message Type | Auto dtype | Output |
|-------------------|-----------|--------|
| `sensor_msgs/msg/JointState` | `float64` | Selected position/velocity/effort values |
| `sensor_msgs/msg/CompressedImage` | `video` | RGB uint8 `(H, W, 3)` |
| `sensor_msgs/msg/Image` | `video` | RGB uint8 `(H, W, 3)` |
| `geometry_msgs/msg/Twist` | `float64` | Selected linear/angular components |
| `geometry_msgs/msg/TwistStamped` | `float64` | Selected twist components |
| `nav_msgs/msg/Odometry` | `float64` | Selected pose/twist fields |
| `sensor_msgs/msg/Imu` | `float64` | Orientation, angular vel, linear accel |
| `sensor_msgs/msg/Joy` | `float32` | Selected axes/buttons |
| `control_msgs/msg/MultiDOFCommand` | `float64` | Selected DOF values |
| `trajectory_msgs/msg/JointTrajectory` | `float64` | Selected first-point fields |
| `std_msgs/msg/Float32` | `float32` | Scalar `(1,)` |
| `std_msgs/msg/Float64` | `float64` | Scalar `(1,)` |
| `std_msgs/msg/Int32` | `int32` | Scalar `(1,)` |
| `std_msgs/msg/Int64` | `int64` | Scalar `(1,)` |
| `std_msgs/msg/String` | `string` | Text `(1,)` |
| `std_msgs/msg/Bool` | `bool` | Boolean `(1,)` |
| `std_msgs/msg/Float32MultiArray` | `float32` | Vector `(N,)` |
| `std_msgs/msg/Float64MultiArray` | `float64` | Vector `(N,)` |
| `std_msgs/msg/Int32MultiArray` | `int32` | Vector `(N,)` |

Override the auto-detected dtype with the `dtype` field in the contract, or use a [custom decoder](../how-to/add-custom-codecs.md) for non-standard message types.

## Policy feature compatibility

Each LeRobot policy implements its own `validate_features()` and accesses batch keys differently. There is no single enforced schema. Accepted keys depend on the policy. This table summarizes the requirements from the modeling code in `lerobot/src/lerobot/policies/`:

| Feature | ACT | SmolVLA | Pi0 | Pi0-Fast | Pi0.5 | GR00T N1.7 | Wall-X | X-VLA |
|---------|:---:|:-------:|:---:|:--------:|:-----:|:----------:|:------:|:-----:|
| **Type** | BC | VLA | VLA | VLA | VLA | VLA | VLA | VLA |
| **`observation.state`** | optional | **required** | optional* | optional* | optional* | optional* | optional* | **required**† |
| **`observation.environment_state`** | optional | - | - | - | - | - | - | - |
| **`observation.images.*`** | multi | multi | multi | multi | multi | multi | multi | multi |
| **`task` string** | - | **required**‡ | **required**‡ | **required**‡ | **required**‡ | **required**‡ | **required**‡ | **required**‡ |
| **`action`** | **required** | **required** | **required** | **required** | **required** | **required** | **required** | **required** |
| **VLM backbone** | - | SmolVLM2 (0.5B) | PaliGemma (3B, 0.7B variant) | PaliGemma (3B) | PaliGemma (3B, 0.7B variant) | Cosmos-Reason2 / Qwen3-VL | Qwen 2.5-VL | Florence2 |
| **RTC support** | - | yes | yes | yes | yes | - | - | - |
| **Max state dim** | any | 32 | 32 | 32 | 32 | 132 | 20 | 32 |
| **Max action dim** | any | 32 | 32 | 32 | 32 | 132 | 20 | 20 |
| **Image size** | any | 512×512 | 224×224 | 224×224 | 224×224 | 256×256 | any | any |
| **Max language tokens** | - | 48 | 48 | 200 | 200 | - | 768 | 64 |
| **Chunk size (default)** | 100 | 50 | 50 | 50 | 50 | 40 | 32 | 32 |
| **Async inference** | yes | yes | yes | - | yes | yes | - | - |

*\*Missing state is auto-padded with zeros by `validate_features`, so these policies run without a state key.*
*†X-VLA requires state under the default `use_proprio: true`. Set `use_proprio: false` to run without state.*
*‡"Required" is functional, not enforced: these policies are language-conditioned, and a missing `task` degrades output rather than raising an error.*

**Key dimensions:**

- **Max images**: All "multi" policies handle N cameras, configured at init time. No policy has unlimited image capacity. ACT concatenates image features, so the practical limit depends on the model's hidden dimension. VLA policies (Pi0 family, SmolVLA, Wall-X) feed images through a VLM, so the VLM context window constrains the image count. Setups with 2 to 3 cameras rarely hit these limits.
- **Max language tokens**: Maximum token count the policy's tokenizer keeps from your task string (`tokenizer_max_length`). Longer prompts truncate. GR00T delegates tokenization to the backbone's own processor and declares no limit.
- **Chunk size**: Future action steps the policy predicts per inference call (`chunk_size`). Larger chunks mean fewer inference calls and less reactivity. Most policies build architecture (positional embeddings, pre-allocated tensors) to match the configured `chunk_size` at init time.
- **RTC (Real-Time Chunking)**: An [inference wrapper](https://huggingface.co/docs/lerobot/rtc) improving real-time performance by overlapping action chunks with continuous re-planning. In v0.6.0, the policies carrying an `rtc_config` field are SmolVLA and the Pi0 family (plus evo1 and molmoact2 outside this table).
- **Async inference**: Whether the policy is in LeRobot's gRPC-based asynchronous inference server allowlist. `SUPPORTED_POLICIES` in `async_inference/constants.py` lists `act`, `smolvla`, `diffusion`, `tdmpc`, `vqbet`, `pi0`, `pi05`, `groot`. [Async](https://huggingface.co/docs/lerobot/rtc) decouples observation collection from action computation, useful for high-frequency control loops. Pi0-Fast, Wall-X, and X-VLA implement `predict_action_chunk()` and are technically compatible, but are not on the allowlist.

**VLA language pipeline**: All VLA policies require a `task` string (e.g. `"pick up the red block"`). In Rosetta, the string comes from the `prompt` argument when recording or running a policy. Tokenization into tensors happens automatically, either by LeRobot's `TokenizerProcessorStep` (a pipeline step running before the policy sees the data) or inside the policy. From a Rosetta/ROS2 perspective, **you provide the task prompt and nothing else**.

**Subtask support**: LeRobot's `lerobot-annotate` [tool](https://huggingface.co/spaces/lerobot/annotate) adds subtask annotations to recorded episodes (e.g. marking "reach for object", "grasp", "lift" within a longer task). Annotations store as `language_events` columns in the dataset's parquet files. **No current action policy consumes subtask annotations.** [SARM](https://huggingface.co/docs/lerobot/sarm) (a reward model) uses them to compute progress scores for [RA-BC](https://huggingface.co/docs/lerobot/sarm) weighted training of Pi0, Pi0.5, and SmolVLA.

### Supported policies

| Policy | Type | Best For |
|--------|------|----------|
| [**ACT**](https://huggingface.co/docs/lerobot/act) | Behavior Cloning | General manipulation, fast training (recommended for beginners) |
| [**SmolVLA**](https://huggingface.co/docs/lerobot/smolvla) | VLA | Efficient VLA, good for resource-constrained setups |
| [**Pi0**](https://huggingface.co/docs/lerobot/pi0) / [**Pi0Fast**](https://huggingface.co/docs/lerobot/pi0fast) | VLA | Physical Intelligence foundation models |
| [**Pi0.5**](https://huggingface.co/docs/lerobot/pi05) | VLA | Open-world generalization |
| [**NVIDIA GR00T N1.7**](https://huggingface.co/docs/lerobot/groot) | VLA | Humanoid and general robotics |
| [**Wall-X**](https://huggingface.co/docs/lerobot/walloss) | VLA | Qwen 2.5-VL backbone, multi-embodiment |
| [**X-VLA**](https://huggingface.co/docs/lerobot/xvla) | VLA | Cross-embodiment with soft prompts |
