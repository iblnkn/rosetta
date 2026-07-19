<p align="center">
  <img alt="Rosetta" src="media/rosetta_logo.png" width="100%">
</p>
<!-- <p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/ROS2-Jazzy-blue" alt="ROS2">
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python 3.10+">
</p> -->

**Rosetta** sits between pub/sub robots and policy-learning frameworks: one contract per robot turns messy ROS2 topics into the clean fixed-rate frames that frameworks like [LeRobot](https://github.com/huggingface/lerobot) consume — for recording, training, and live inference.

## Table of Contents

- [Recent Changes](#recent-changes)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
  - [What is LeRobot?](#what-is-lerobot)
  - [What is Rosetta?](#what-is-rosetta)
- [Architecture](#architecture)
- [The Contract](#the-contract)
- [Recording Episodes](#recording-episodes)
  - [Why Record to Bag Files?](#why-record-to-bag-files)
- [Converting Bags to Datasets](#converting-bags-to-datasets)
- [Training a Policy](#training-a-policy)
  - [Supported Policies](#supported-policies)
- [Deploying Policies](#deploying-policies)
- [Contract Reference](#contract-reference)
  - [Minimal Example](#minimal-example)
  - [Observations](#observations)
  - [Actions](#actions)
  - [Operators](#operators)
  - [Teleop](#teleop)
  - [Tasks, Rewards, and Signals](#tasks-rewards-and-signals)
  - [Adjunct Topics](#adjunct-topics)
  - [Select Syntax](#select-syntax)
  - [Alignment Strategies](#alignment-strategies)
  - [Supported Message Types](#supported-message-types)
  - [Custom Encoders/Decoders](#custom-encodersdecoders)
- [LeRobot Data Model Reference](#lerobot-data-model-reference)
  - [Key System](#key-system)
  - [EnvTransition](#envtransition)
  - [Data Types](#data-types)
  - [Policy Feature Compatibility](#policy-feature-compatibility)
- [License](#license)

<a id="recent-changes"></a>
<details>
<summary><strong>Recent Changes</strong></summary>

- **Codec rename & fail-fast registration (breaking):** "codec" is now "codec" everywhere — `rosetta/frames/codecs.py` → `rosetta/frames/codecs.py`, `discover_codecs` → `discover_codecs`, the plugin entry-point group `rosetta.codecs` → `rosetta.codecs`, and the bundled example `rosetta.examples.stone_codecs` → `rosetta.examples.stone_codecs` (update inline `decoder:`/`encoder:` paths that referenced it). `encode_value` is now payload-first — `encode_value(action_vec, spec, stamp_ns=None)` — matching `decode_value(msg, spec)` and the registered encoder callables. Registration fails fast: `register_decoder` validates its `dtype` against the contract vocabulary, and `override=True` with nothing registered is an error (a plugin overriding a renamed/removed built-in fails loudly instead of silently registering fresh). A plugin that fails to import now latches that failure — every later discovery call re-raises the original error until the process restarts (a rescan could never recover: partial registrations from the failed module would collide as duplicates and mask the real error). The dtype vocabulary (`SUPPORTED_NUMERIC_DTYPES`/`SPECIAL_DTYPES`) lives in `frames/codecs.py` beside the registry; schema re-exports it. Load-time additions: a teleop input whose dtype is non-numeric is rejected (its decoded value re-encodes onto a numeric action topic), and Joy selectors reject negative indices. A hub-resolved sidecar contract that declares inline `decoder:`/`encoder:` paths now logs a prominent warning naming each one.
- **Operator & serve-path hardening (breaking):** the encode path now refuses non-finite commands — `encode_value` raises `NonFiniteActionError` on NaN/Inf (after the inverse pipeline, on every action channel), and `TopicBridge.publish_frame` encodes the whole frame before publishing any of it, dropping a non-finite frame atomically with a throttled error (the watchdog's per-channel `safety` covers persistence; `safety: hold` can never re-send a bad frame). `clamp` is dict-only (`{min: lo, max: hi}` — the `[lo, hi]` list form no longer loads), rejects bool/string bounds, and preserves the input dtype (no more silent uint8→float64 promotion). `resize` requires two integers in `[1, 8192]` (floats/bools/strings were silently truncated or crashed load with raw errors before) and is rejected at load on non-image streams (it used to load and then crash on every message). `_nearest_resize` now rounds sampling indices symmetrically instead of truncating (≤ 1 source-pixel shift vs previously ported datasets; record and serve stay pixel-identical by construction). Bare operators (`rad2deg`) reject stray argument payloads at load.
- **Teleop input/feedback targeting (breaking):** `teleop.input`/`teleop.feedback` are now each a *list* of independently-targeted sources instead of a single block: every entry names the topic it corresponds to in `actions`/`observations` via `target`/`origin` (validated at load time), so a contract can teleop one action, several, or none. Schema: `Teleop.input`/`.feedback` changed from `FrameEntry | None` to `tuple[TeleopInputSource | TeleopFeedbackSource, ...]` (new dataclasses in `contract/schema.py`). `hil_manager_node` now decodes each teleop input and encodes+publishes it onto its `target` using the contract's normal decode/encode machinery (previously it republished the raw message to every command publisher unchanged, which only worked when every action shared one message type). The porter additionally records each entry as its own `teleop.input.<action key>` / `teleop.feedback.<observation key>` diagnostic column.
- **Online/offline split (breaking):** `rosetta.robots.ros2.port` and `rosetta.robots.ros2.bag_frames` moved to `rosetta.robots.ros2.offline.{port,bag_frames}`; everything else in `rosetta.robots.ros2` (decoders/encoders, `topic_bridge`, nodes) is the live path. The `rosetta_port` console script and its CLI are unchanged.
- **Spec composition (breaking):** runtime specs now carry `source: Source` — the exact declaration they were resolved from — and the flat passthrough fields are gone: `spec.topic` → `spec.source.channel.topic`, `spec.msg_type` → `spec.source.channel.type`, `spec.align` → `spec.source.align`, `spec.qos` → `spec.source.channel.qos`, `spec.decoder`/`spec.encoder` → `spec.source.channel.*`, `spec.safety_behavior` → `spec.source.channel.safety`, `spec.kind` → `spec.source.kind`. Computed fields (`key`, `names`, `fps`, `dtype`, `namespace`, `operators`, `is_image`, `image_resize`) stay flat. A forgotten copy is no longer expressible, so the field-set parity guard is retired (the test now pins computed derivations + the identity guarantee). Behavior fix: reward-as-action specs now honor a `kind` declared on a reward channel (the flat design silently dropped it to `continuous`).
- **Frame I/O rename (breaking):** "stream" now means exactly one thing — the decoded, timelined sample sequence of a single channel *before* alignment (the `StreamSpec`/`StreamBuffer` sense, matching Rerun semantics). The post-align robot surface is not a stream (it's bidirectional), so it was renamed: `rosetta.frames.streams` → `rosetta.frames.protocols`, `FrameStream` → `FrameIO` (a `FrameIO` is both a `FrameSource` and a `FrameSink`, like Python's `TextIO`); `PolicyRunner.run()`'s first parameter is now `frames: FrameIO`. `FrameSource`/`FrameSink` are unchanged.
- **Contract naming split (breaking):** "Spec" now means exactly one thing — a resolved runtime stream spec. The `StreamSpec` family (`StreamSpec`, `ObservationStreamSpec`, `ActionStreamSpec`) moved from `rosetta.contract.schema` to `rosetta.contract.specs` (import from there, or from the top-level `rosetta`); the declaration-side dataclasses in `contract/schema.py` dropped the suffix to mirror their YAML stanzas — `ChannelSpec` → `Channel`, `AlignSpec` → `Align`, `SourceSpec` → `Source`, `TaskSpec` → `Task`, `TeleopSpec` → `Teleop`, `TeleopEventsSpec` → `TeleopEventMap` (not `TeleopEvents`: that name is lerobot's event enum). `FrameEntry`, `Contract`, enums, and the contract YAML are unchanged.
- **Operators rename (breaking):** `rosetta/contract/ops.py` → `rosetta/contract/operators.py`; "op" is now "operator" everywhere — `Op` → `Operator`, `OpContext` → `OperatorContext`, `register_op` → `register_operator`, `build_op` → `build_operator`, `OP_REGISTRY` → `OPERATOR_REGISTRY`, and the plugin entry-point group `rosetta.ops` → `rosetta.operators`. Resolved specs carry `operators` (was `ops`). The contract YAML is unchanged (`apply:` lists and operator names stay as they were).
- **Contract format v2 (breaking):** sections are now mappings keyed by frame key (v1's flat entry lists no longer load); each entry is a `channel: {topic, type, qos, ...}` block plus a **mandatory** `align: {strategy, timeline}` (`tolerance_ms` with `asof`), then `select`/`apply`/`kind`. A list value under one key declares ordered sources (concatenation for observations, action splitting). `serve: {safety}` moved to `channel.safety`; `stamp: receive|header` became open timeline names (`timeline: receive|header`) chosen per source — including actions, replacing the deleted global `timestamp_source`. New required top-level fields: `robot_interface: ros2` and `fps`. Validation replaces fallback: unknown keys anywhere, a missing align, or a timeline the channel's message type cannot provide (e.g. `header` on `std_msgs`) are load-time errors, and a header-timeline message arriving unstamped is dropped at ingest instead of silently falling back to receive time. Top-level `x-*` keys are ignored (YAML anchor holders, e.g. shared QoS). `clamp` accepts `{min, max}`. See `contracts/stone.yaml` for the annotated tour.
- **Restructure (breaking):** the package tree now states the architecture — `rosetta.core` split into `rosetta.contract` (`contract.py` → `contract/schema.py`, `contract_utils.py` → `contract/specs.py`, plus `ops.py`, `errors.py`) and `rosetta.frames` (`codecs.py` → `frames/codecs.py` (since renamed back to `frames/codecs.py`, see above), `frame_layout.py` → `frames/layout.py`, `resample.py` → `frames/resample.py`, naming helpers → `frames/naming.py`); `rosetta.ros2` → `rosetta.robots.ros2`; `rosetta.backends.protocols` → `rosetta.policies`. New `rosetta.frames.streams` defines `FrameSource`/`FrameSink`/`FrameStream` — `PolicyRunner.run()` now takes a `FrameStream` instead of a `TopicBridge`, so framework adapters no longer import rclpy. Entry-point groups (`rosetta.dataset_writers`, `rosetta.policy_runners`) are unchanged.
- **Node rename (breaking):** `rosetta_client_node` → `policy_runner_node` (class `PolicyRunnerNode`, default node name `policy_runner`, launch `policy_runner_launch.py`, params `policy_runner.yaml`); its `backend` parameter is now `framework`, and `rosetta_port`'s `--backend` flag is now `--framework`
- **Contract:** `name` → `robot_type`, `rate_hz` → `fps`
- **Nodes:** `PolicyBridge` → `rosetta_client_node`, `EpisodeRecorderServer` → `episode_recorder_node`
- **Actions:** relative names (`run_policy`, `record_episode`) prefixed by the launch-file namespace — `/run_policy` and `/record_episode` in the default launches; `/robot_policy/run_policy` and `/reward_classifier/run_policy` in the HIL launch
- **Launch:** `turtlebot_policy_bridge.launch.py` → `rosetta_client_launch.py`, `turtlebot_recorder_server.launch.py` → `episode_recorder_launch.py`
- **Conversion:** `bag_to_lerobot.py` → `rosetta.robots.ros2.port` (the `rosetta_port` console script) (now processes directories, supports sharding)
- **Inference:** Policy loading moved to LeRobot's async gRPC server
- **New:** `lerobot_teleoperator_rosetta` (experimental)

</details>

---

## Quick Start

```
  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
  │  DEFINE  │     │  RECORD  │     │ CONVERT  │     │  TRAIN   │     │  DEPLOY  │
  │ Contract │────▶│  Demos   │────▶│ Dataset  │────▶│  Policy  │────▶│ on Robot │
  └──────────┘     └──────────┘     └──────────┘     └──────────┘     └──────────┘
```

[**Define**](#the-contract) a contract that maps your ROS2 topics to [LeRobot](https://github.com/huggingface/lerobot) features, [**record**](#recording-episodes) demos to bag files, [**convert**](#converting-bags-to-datasets) them to a LeRobot dataset, [**train**](#training-a-policy) a policy, and [**deploy**](#deploying-policies) it back to your robot.

> **Getting started?** The [rosetta_ws](https://github.com/iblnkn/rosetta_ws) devcontainer handles the non-trivial setup of getting ROS2, Rosetta, and LeRobot installed together.

**1. Define** a [contract](#the-contract) for your robot:

```yaml
# my_contract.yaml
robot_type: my_robot
robot_interface: ros2
fps: 30

observations:
  observation.state:
    channel: {topic: /joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: header}
    select: [position.j1, position.j2]

  observation.images.cam:
    channel: {topic: /camera/image_raw/compressed,
              type: sensor_msgs/msg/CompressedImage}
    align: {strategy: hold, timeline: header}
    apply: [resize: [480, 640]]

actions:
  action:
    channel: {topic: /cmd, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: header}
    select: [position.j1, position.j2]
```

**2. Record** demonstrations to rosbag:

```bash
# Terminal 1: Start the recorder
ros2 launch rosetta episode_recorder_launch.py contract_path:=contract.yaml
```

```bash
# Terminal 2: Keyboard controller (r=start, s=save, d=discard, t=set prompt, q=quit)
ros2 run rosetta episode_keyboard_node
```

> **How many episodes?** Plan on recording **50–200+ demonstrations** depending on task complexity. More diverse, high-quality demonstrations tend to produce better policies. For practical data collection tips, see [Collecting Your Dataset](https://abenstirling.com/lerobot/) and [Improving Your Robotics AI Model](https://docs.phospho.ai/learn/improve-robotics-ai-model).

**3. Convert** bags to LeRobot dataset:

```bash
python -m rosetta.robots.ros2.offline.port \
    --raw-dir ./datasets/bags \
    --contract my_contract.yaml \
    --repo-id my-org/my-dataset \
    --root ./datasets/lerobot
```

**4. Train** with LeRobot:

```bash
lerobot-train \
    --dataset.repo_id=my-org/my-dataset \
    --policy.type=act \
    --output_dir=outputs/train/my_policy
```

**5. Deploy** the trained policy:

```bash
# Terminal 1: Start the client
ros2 launch rosetta policy_runner_launch.py \
    contract_path:=contract.yaml \
    pretrained_name_or_path:=my-org/my-policy
```

```bash
# Terminal 2: Run the policy
ros2 action send_goal /run_policy \
    rosetta_interfaces/action/RunPolicy "{prompt: 'pick up red block'}"
```

(The action is served under the relative name `run_policy`; a launch-file namespace prefixes it, e.g. `/robot_policy/run_policy` in the HIL launch.)

---

## Core Concepts

### What is LeRobot?

[LeRobot](https://github.com/huggingface/lerobot) is Hugging Face's open-source framework for [robot learning](https://huggingface.co/spaces/lerobot/robot-learning-tutorial). It provides tools for recording demonstrations, training policies (ACT, Diffusion Policy, VLAs like SmolVLA and Pi0), and deploying them on hardware. LeRobot defines a standard dataset format (v3) built on Parquet files and MP4 videos, with a growing ecosystem of community-contributed datasets and models on the [Hugging Face Hub](https://huggingface.co/datasets?other=LeRobot).

### What is Rosetta?

Robots are messy; policies want structure. A **contract** (one YAML per robot) declares how the robot's pub/sub topics become clean fixed-rate **frames** — and back. The **robot side** (`rosetta.robots`) adapts each pub/sub ecosystem onto that frame stream (ROS2 today). The **policy side** (`rosetta.policies`) adapts each learning framework (LeRobot, vla_foundry, starvla, ...) to consume it, for dataset writing and live policy execution. Either side swaps out without touching the other, because both speak only frames.

The same frame machinery (`rosetta.frames`) runs live inference and offline bag conversion, so training data matches inference input sample-for-sample by construction.

## Architecture

Inside the `rosetta` package, the tree tells the story:

```
rosetta/
├── contract/    # the declaration: one YAML per robot (schema, specs, operators)
├── frames/      # the interlingua: layout, resampling, codecs, stream protocols
├── robots/      # robot side: pub/sub ecosystems (ros2/ today)
└── policies/    # policy side: DatasetWriter + PolicyRunner seams, entry-point loading
```

The data-path vocabulary is three words, in pipeline order: a **channel** is a declared endpoint in the robot interface's dialect (`schema.Channel`); a **stream** is the decoded, timelined sample sequence of one channel *before* alignment (`StreamSpec` describes it, `StreamBuffer` lands it on the clock); a **frame** is one synchronized sample of every contract key per clock tick, *after* alignment (`FrameIO` — a `FrameSource` + `FrameSink` — is the bidirectional frame surface a `PolicyRunner` drives).

Inside `contract/`, the split is *say vs. do*: `schema.py` is what the contract **says** — the typed document model of the YAML (`Channel`, `Align`, `Source`, `FrameEntry`, `Contract`), validation, and `load_contract()`. `specs.py` is what the runtime **consumes** — the resolved stream specifications (`StreamSpec` family) produced by the `iter_*_specs` pass. A spec is composed, not copied: it carries `source` (the exact declaration `Source` it was resolved from — read `spec.source.channel.topic`, `spec.source.align`, `spec.source.kind` through it) plus the computed fields the YAML never states (`names`, `dtype`, `namespace`, `operators`, image geometry). Downstream code takes `list[StreamSpec]` and never re-reads the document; "spec" always means one of these resolved runtime objects, and a declaration fact can never silently go missing from one — there is no copy step to forget.

The workspace consists of several packages; framework adapters register into `rosetta.policies` entry points:

| Package | Purpose |
|---------|---------|
| `rosetta` | Core library, nodes, bag conversion |
| [`rosetta_interfaces`](https://github.com/iblnkn/rosetta_interfaces) | ROS2 action/service definitions |
| [`lerobot_rosetta`](https://github.com/iblnkn/lerobot-rosetta) | LeRobot backend adapter — dataset writer, policy runner, inference servers |
| [`lerobot_robot_rosetta`](https://github.com/iblnkn/lerobot-robot-rosetta) | LeRobot Robot plugin (discovered by LeRobot) |
| [`lerobot_teleoperator_rosetta`](https://github.com/iblnkn/lerobot-teleoperator-rosetta) | LeRobot Teleoperator plugin (experimental) |

```
rosetta/
├── launch/
│   ├── episode_keyboard_launch.py
│   ├── episode_recorder_launch.py
│   ├── hil_launch.py
│   └── policy_runner_launch.py
└── params/
    ├── episode_recorder.yaml    # Default config for Episode Recorder
    ├── hil_manager.yaml         # HIL launch "super YAML" (all four nodes)
    └── policy_runner.yaml       # Default config for the policy runner
```

### LeRobot Plugin Architecture

The `lerobot_robot_rosetta` and `lerobot_teleoperator_rosetta` packages implement LeRobot's [Robot](https://huggingface.co/docs/lerobot/integrate_hardware) and [Teleoperator](https://huggingface.co/docs/lerobot/integrate_hardware#adding-a-teleoperator) interfaces. They follow LeRobot's [plugin conventions](https://huggingface.co/docs/lerobot/integrate_hardware#using-your-own-lerobot-devices-) (`lerobot_robot_*` and `lerobot_teleoperator_*` prefixes) for auto-discovery when installed.

**Typical LeRobot robots** (like `so101_follower`) communicate directly with hardware:
- Motors via serial/CAN (`FeetechMotorsBus`, `DynamixelMotorsBus`)
- Cameras via USB/OpenCV
- The `Robot` class IS the hardware interface

**Rosetta robots** are ROS2 lifecycle nodes:
- Subscribe to ROS2 topics for observations
- Publish to ROS2 topics for actions
- Hardware drivers exist elsewhere in the ROS2 graph
- The contract YAML defines topic-to-feature mapping

**Important:** Because `lerobot_robot_rosetta` creates a ROS2 lifecycle node internally, **your system needs ROS2 installed** to use it, even when invoking it through LeRobot's standard CLI tools. When `policy_runner_node` launches inference, the chain is: `policy_runner_node` (ROS2 node) → LeRobot `RobotClient` → `lerobot_robot_rosetta` (also a ROS2 node) → your robot's ROS2 topics. Both the convenience node and the robot plugin are ROS2 nodes running in the same ROS2 graph.

This means any ROS2 robot can use LeRobot's tools. Define a contract and use `--robot.type=rosetta`.

### ROS2 Lifecycle Integration

LeRobot's `connect()` / `disconnect()` map to ROS2 lifecycle transitions:

| LeRobot Method | Lifecycle Transition | Effect |
|----------------|---------------------|--------|
| - | `configure` | Create subscriptions (start buffering), create publishers (disabled) |
| `connect()` | `activate` | Enable publishers, start watchdog |
| `disconnect()` | `deactivate` → `cleanup` | Safety action, disable publishers, destroy resources |

### Policy Inference

The `policy_runner_node` delegates inference to a gRPC policy server (`lerobot_rosetta.policy_server`, a thin preload/cache wrapper over LeRobot's `lerobot.async_inference.policy_server`). The server has no ROS2 dependency and can run on any machine with LeRobot and a GPU. Benefits:

- Better GPU memory management
- Support for all LeRobot policy types without code changes
- Consistent behavior between training and deployment
- Can run on a remote machine, letting a resource-constrained robot offload inference over the network

When `launch_local_server` is `true`, the server is started — and the model fully loaded — at node **configure** time, so the first `run_policy` goal costs the same as any other. The configure transition blocks until the model is up (bounded by `server_startup_timeout_sec`), GPU memory is held from startup, and later goals reuse the loaded model instead of re-reading the checkpoint on every handshake.

### rosetta_ws Workspace

We provide [rosetta_ws](https://github.com/iblnkn/rosetta_ws), a devcontainer workspace for getting started quickly. Getting ROS2 and LeRobot installed together is not trivial; the workspace handles this setup.

---

## The Contract

The contract defines the translation between ROS 2 topics and the keys LeRobot expects.

| ROS2 Side | | LeRobot Side |
|-----------|---|-------------|
| `/front_camera/image_raw/compressed` | &rarr; | `observation.images.front` |
| `/follower_arm/joint_states` (position fields) | &rarr; | `observation.state` |
| `/imu/data` (orientation, angular_velocity) | &rarr; | `observation.state.imu` |
| `/leader_arm/joint_states` (position fields) | &larr; | `action` |
| `/base_controller/cmd_vel` (linear, angular) | &larr; | `action.base` |
| `/task_prompt` (String) | &rarr; | `task` |
| `/reward_signal` (Float64) | &rarr; | `next.reward` |

On the ROS2 side, data lives in typed messages on named topics with rich structure (headers, arrays, nested fields). On the LeRobot side, data lives in flat dictionaries with dot-separated string keys and numpy/tensor values. The contract maps one to the other, handling type conversion, field extraction, timestamp alignment, and resampling.

Every frame-clock entry reads as one pipeline — `channel (provides) → align (chooses a timeline; mandatory) → select → apply → the mapping key`:

```yaml
observation.state:
  channel: {topic: /follower_arm/joint_states, type: sensor_msgs/msg/JointState}
  align: {strategy: hold, timeline: header}
  select: [position.shoulder_pan, position.shoulder_lift, position.elbow,
           position.wrist_pitch, position.wrist_roll, position.wrist_yaw]
```

At each timestep, this:
1. **Subscribes** to `/follower_arm/joint_states` (a `JointState` message) — the channel block is exactly what a different pub/sub ecosystem would replace
2. **Aligns** the stream onto the frame clock using the message's `header` timeline (holding the latest value)
3. **Extracts** the named fields using dot notation (`position.shoulder_pan` → `msg.position[msg.name.index("shoulder_pan")]`)
4. **Assembles** a numpy array: `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]` (dtype `float64`)
5. **Stores** it under the key `observation.state` in the LeRobot dataset

**Multi-source concatenation**: a **list** value under one key declares ordered sources whose values concatenate into a single feature vector:

```yaml
observations:
  observation.state:
    - channel: {topic: /arm/joint_states, type: sensor_msgs/msg/JointState}
      align: {strategy: hold, timeline: header}
      select: [position.j1, position.j2, position.j3]
    - channel: {topic: /gripper/state, type: std_msgs/msg/Float32}
      align: {strategy: hold, timeline: receive}
      # Result: observation.state = [j1, j2, j3, gripper] (4D vector)
```
This is important because, as shown in [Policy Feature Compatibility](#policy-feature-compatibility), all of the available core policies depend on explicit names for most keys. If you have multiple ros2 topics you would like to use for observation, the most straightforward way to achieve this is to declare them as ordered sources of one key.

The same topic may also appear in several sources (e.g. position and orientation slices of one pose topic) — each source keeps its own selector and buffer.

Multi-source keys are validated at load (`ContractValidationError` otherwise):

- every source of a multi-source key must have a `select` (concatenation needs static dims to lay out the combined vector);
- all sources of a key must resolve to the **same** dtype (set `dtype:` in the channel explicitly to align them);
- images and strings cannot share a key; give each its own key.

For images, each image key must be unique; image features cannot be concatenated.

Layout is purely by declaration order — there is no separate ordering key (e.g. `position: 2`). Order is already the one thing a YAML list gives you for free, and it's fully expressive: any position for any source, reordered with a one-line diff. An explicit ordering field would just duplicate that ordering in a second place with no new capability, only a new way for the two to drift out of sync.

A minimal contract typically only needs `observations` and `actions`. See the full [Contract Reference](#contract-reference) for all options, and the [LeRobot Data Model Reference](#lerobot-data-model-reference) for how keys, features, and policies interact.

---

## Recording Episodes

The `episode_recorder_node` is a convenience node that records contract-specified topics to [rosbag2](https://github.com/ros2/rosbag2) files. It reads the contract to determine which topics to subscribe to, then lets you start and stop recording via ROS2 actions, with feedback on duration and message count.

**This node is not the only way to record compatible bags.** Any method that produces a valid rosbag2 file containing the contract's topics will work, including `ros2 bag record`, custom scripts using `rosbag2_py`, or third-party recording tools. The `episode_recorder_node` makes this convenient within the ROS2 ecosystem: you define your topics once in the contract, and it handles subscription setup, bag lifecycle, and action-based control. It may also be useful standalone for any workflow where you need to define a set of topics and start/stop recording programmatically via ROS2 actions.

> Both Rosetta nodes use parameter files (`params/`) as defaults. All parameters are also exposed as launch arguments, which override the defaults. Run `ros2 launch rosetta <launch_file> --show-args` to see all options.

```bash
ros2 launch rosetta episode_recorder_launch.py contract_path:=/path/to/contract.yaml
```

### Controlling Recording

#### Option A — Keyboard controller (recommended)

The `episode_keyboard_node` lets you start, stop, and discard episodes with single key presses. Run it in a **second terminal** while the recorder is running:

```bash
ros2 run rosetta episode_keyboard_node
```

Or via launch (supports optional arguments):

```bash
ros2 launch rosetta episode_keyboard_launch.py \
    default_prompt:="pick up the cube"
```

| Key | Action |
|-----|--------|
| `r` / `→` | Start recording |
| `s` / `←` | Stop and save |
| `d` / `⌫` | Discard episode (stop + delete bag) |
| `t` | Edit task prompt for the next episode |
| `h` / `?` | Help |
| `q` | Quit |

Launch arguments for `episode_keyboard_launch.py`:

| Argument | Default | Description |
|----------|---------|-------------|
| `recorder_ns` | `/episode_recorder` | Namespace of the recorder node |
| `default_prompt` | `` | Initial task prompt used when starting recordings |

#### Option B — ROS2 action

For scripted or automated workflows, trigger recording directly via the action interface:

```bash
ros2 action send_goal /record_episode \
    rosetta_interfaces/action/RecordEpisode "{prompt: 'task description'}"
```

Stop by sending `Ctrl-C` to the `send_goal` command, or via the cancel service:

```bash
ros2 service call /episode_recorder/cancel_recording std_srvs/srv/Trigger
```

**Parameters** (all available as launch arguments):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `contract_path` | `contracts/so_101.yaml` | Path to contract YAML |
| `bag_base_dir` | `datasets/bags` | Rosbag output dir; relative paths resolve against the launch cwd (like `ros2 bag record`) |
| `storage_id` | `mcap` | Rosbag format: `mcap` (recommended) or `sqlite3` |
| `default_max_duration` | `300.0` | Max episode duration in seconds |
| `feedback_rate_hz` | `2.0` | Recording feedback publish rate |
| `log_level` | `info` | Logging level: `debug`, `info`, `warn`, `error` |
| `configure` | `true` | Auto-configure on startup |
| `activate` | `true` | Auto-activate on startup |

**Examples:**

```bash
# Override output directory
ros2 launch rosetta episode_recorder_launch.py \
    contract_path:=/path/to/contract.yaml \
    bag_base_dir:=/data/recordings

# Change max duration and storage format
ros2 launch rosetta episode_recorder_launch.py \
    contract_path:=/path/to/contract.yaml \
    default_max_duration:=600.0 \
    storage_id:=sqlite3
```

### Why Record to Bag Files?

Rosetta records demonstrations to [rosbag2](https://github.com/ros2/rosbag2) files first, then converts them to LeRobot datasets in a separate step. This is a deliberate design choice with several benefits:

- **Preserves raw data.** Bag files store every message at its original rate and timestamp, with no alignment, downsampling, or lossy transformation. This means you can reprocess the same recordings later with a different contract (changing feature keys, adjusting resampling rates, adding new topics) without re-recording.
- **Familiar to ROS2 users.** Bag files are the standard data format in the ROS2 ecosystem, with mature tooling for [recording, playback, inspection](https://docs.ros.org/en/jazzy/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html), and analysis. Any tool that works with bag files works with your recorded data.
- **Stores data beyond what LeRobot needs.** Bags can include topics that don't map to any LeRobot feature: diagnostics, TF trees, debug streams, extra sensors. This data is preserved for analysis, debugging, or future use even though it isn't part of the training dataset.
- **Leverages MCAP.** Rosetta defaults to [MCAP](https://mcap.dev/) storage, which provides [high-performance](https://mcap.dev/guides/benchmarks/rosbag2-storage-plugins) random-access reads, efficient compression, and broad ecosystem support beyond ROS2.
- **Write-optimized for live recording.** Bag files (especially MCAP) are designed for high-throughput sequential writes with minimal overhead, well-suited for capturing live sensor data. LeRobot datasets (Parquet + MP4) are read-optimized for training but involve more overhead when writing live, including in-memory buffering and post-episode video encoding.


## Converting Bags to Datasets

The porter (`rosetta_port`, i.e. `python -m rosetta.robots.ros2.offline.port`) converts rosbag2 files to LeRobot datasets using the contract for key mapping, timestamp alignment, resampling, and dtype conversion. It applies the same `StreamBuffer` resampling logic used during live inference, ensuring your offline dataset matches what the robot would see at runtime.

While you could write your own conversion script using the primitives in `rosetta.contract` / `rosetta.frames` (contract loader, stream buffers) and `rosetta.robots.ros2` (decoders), the porter handles the full pipeline: reading bags, applying the contract, encoding video, building the LeRobot dataset structure, and optionally pushing to the Hub. Because the raw bag preserves all data without transformation, you can re-run the porter with an updated contract (changing keys, adjusting `fps`, adding or removing features) without re-recording.


### Relationship to LeRobot

The porter mirrors the interface of LeRobot's example porters (like `port_droid.py`):

```bash
# LeRobot's port_droid.py
python examples/port_datasets/port_droid.py \
    --raw-dir /data/droid/1.0.1 \
    --repo-id my_org/droid \
    --push-to-hub

# Rosetta's rosetta.robots.ros2.offline.port (same pattern + contract)
python -m rosetta.robots.ros2.offline.port \
    --raw-dir ./datasets/bags \
    --contract contract.yaml \
    --repo-id my_org/my_dataset \
    --root ./datasets/lerobot
```

**Rosetta-specific additions:**

| Argument | Description |
|----------|-------------|
| `--contract` | **(Required)** Rosetta contract YAML that defines ROS2 topic → LeRobot feature mapping |
| `--root` | Override output directory (LeRobot defaults to `~/.cache/huggingface/lerobot`) |
| `--vcodec` | Video codec selection (not in base LeRobot porters) |

### Basic Usage

```bash
python -m rosetta.robots.ros2.offline.port \
    --raw-dir ./datasets/bags \
    --contract ./contract.yaml \
    --repo-id my_dataset \
    --root ./datasets/lerobot
```

 For additional information on large-scale conversions, parallel processing, and SLURM cluster workflows, see the **[LeRobot Porting Datasets Guide](https://huggingface.co/docs/lerobot/en/porting_datasets_v3)** and substitute `rosetta_port` for `port_droid.py` in the examples.



## Training a Policy

Once you've converted your ROS2 bags to a LeRobot dataset, [train a policy](https://huggingface.co/docs/lerobot/il_robots#train-a-policy) with `lerobot-train`.


### Quick Start: ACT

```bash
lerobot-train \
    --dataset.repo_id=my-org/my-dataset \
    --policy.type=act \
    --output_dir=outputs/train/act_my_robot \
    --policy.device=cuda \
    --wandb.enable=true
```

### Fine-tuning VLA Models

VLA models are large pre-trained vision-language-action models. Use [PEFT](https://huggingface.co/docs/peft/index)/[LoRA](https://huggingface.co/docs/peft/task_guides/lora_based_methods) for [efficient fine-tuning](https://huggingface.co/docs/lerobot/peft_training):

```bash
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=my-org/my-dataset \
    --policy.output_features=null \
    --policy.input_features=null \
    --steps=100000 \
    --batch_size=32 \
    --peft.method_type=LORA \
    --peft.r=64
```

### Multi-GPU Training

LeRobot supports [training on multiple GPUs](https://huggingface.co/docs/lerobot/multi_gpu_training) using [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index):

```bash
accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --mixed_precision=fp16 \
    $(which lerobot-train) \
    --dataset.repo_id=my-org/my-dataset \
    --policy.type=act \
    --batch_size=32
```

### Resume Training

```bash
lerobot-train \
    --config_path=outputs/train/my_run/checkpoints/last/pretrained_model/train_config.json \
    --resume=true
```

### Upload to HuggingFace Hub

```bash
huggingface-cli upload my-org/my-policy \
    outputs/train/my_run/checkpoints/last/pretrained_model
```


### Supported Policies

| Policy | Type | Best For |
|--------|------|----------|
| [**ACT**](https://huggingface.co/docs/lerobot/act) | Behavior Cloning | General manipulation, fast training (recommended for beginners) |
| [**SmolVLA**](https://huggingface.co/docs/lerobot/smolvla) | VLA | Efficient VLA, good for resource-constrained setups |
| [**Pi0**](https://huggingface.co/docs/lerobot/pi0) / [**Pi0Fast**](https://huggingface.co/docs/lerobot/pi0fast) | VLA | Physical Intelligence foundation models |
| [**Pi0.5**](https://huggingface.co/docs/lerobot/pi05) | VLA | Open-world generalization |
| [**NVIDIA GR00T N1.5**](https://huggingface.co/docs/lerobot/groot) | VLA | Humanoid and general robotics |
| [**Wall-X**](https://huggingface.co/docs/lerobot/walloss) | VLA | Qwen 2.5-VL backbone, multi-embodiment |
| [**X-VLA**](https://huggingface.co/docs/lerobot/xvla) | VLA | Cross-embodiment with soft prompts |

## Deploying Policies

The `policy_runner_node` is a convenience node that wraps LeRobot's inference pipeline in ROS2 actions. It lets you start and stop policy execution via `ros2 action send_goal`, with feedback on inference progress. It can optionally launch a local LeRobot gRPC policy server as a subprocess, or connect to a remote one.

Launch Client:

```bash
ros2 launch rosetta policy_runner_launch.py contract_path:=/path/to/contract.yaml
```

Run policy:

```bash
ros2 action send_goal /run_policy \
    rosetta_interfaces/action/RunPolicy "{prompt: 'task description'}"
```

**Remote inference:** When `launch_local_server` is `false`, the node connects to a gRPC policy server at `server_address`. The server has no ROS2 dependency and can run on any machine with a GPU, completely independent of your robot's ROS2 environment. This lets a resource-constrained robot offload inference to a remote GPU server. To pre-warm the remote server so even the first goal skips the model load:

```bash
python -m lerobot_rosetta.policy_server --host=0.0.0.0 --port=8080 \
    --policy-type=act --pretrained-name-or-path=my-org/my-policy --policy-device=cuda
```

(Stock `lerobot.async_inference.policy_server` also works, but reloads the checkpoint on every goal.)

**Parameters** (all available as launch arguments):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `contract_path` | `contracts/so_101.yaml` | Path to contract YAML |
| `pretrained_name_or_path` | *(see params file)* | HuggingFace model ID or local path |
| `server_address` | `127.0.0.1:8080` | Policy server address |
| `policy_type` | `act` | Policy type: `act`, `smolvla`, `diffusion`, `pi0`, `pi05`, etc. |
| `policy_device` | `cuda` | Inference device: `cuda`, `cpu`, `mps`, or `cuda:0` |
| `actions_per_chunk` | `30` | Actions per inference chunk |
| `chunk_size_threshold` | `0.95` | When to request new chunk (0.0-1.0) |
| `aggregate_fn_name` | `weighted_average` | Chunk aggregation: `weighted_average`, `latest_only`, `average`, `conservative` |
| `feedback_rate_hz` | `2.0` | Execution feedback publish rate |
| `launch_local_server` | `true` | Auto-start policy server subprocess (at configure, with model preload) |
| `server_startup_timeout_sec` | `120.0` | Max wait for the server to come up (covers model preload; raise for cold HF downloads) |
| `obs_similarity_atol` | `-1.0` | Observation filtering tolerance (-1.0 to disable)* |
| `log_level` | `info` | Logging level: `debug`, `info`, `warn`, `error` |
| `configure` | `true` | Auto-configure on startup |
| `activate` | `true` | Auto-activate on startup |

*\*`obs_similarity_atol`: The policy server filters observations that are "too similar" (L2 norm of state difference < threshold). The default threshold (1.0) assumes joint states change significantly between frames. Many robots have smaller movements, causing most observations to be skipped. Set to `-1.0` to disable filtering.*

**Example:**

```bash
# Run with a pretrained model
ros2 launch rosetta policy_runner_launch.py \
    contract_path:=/path/to/contract.yaml \
    pretrained_name_or_path:=my-org/my-policy
```

**This node is not the only way to deploy.** You can run inference using LeRobot's standard CLI tools directly with the Rosetta robot plugin:

```bash
# Standard LeRobot deployment, no policy_runner_node needed
lerobot-record --robot.type=rosetta --robot.config_path=contract.yaml
```

The `lerobot_robot_rosetta` / `lerobot_teleoperator_rosetta` distributions
follow LeRobot's third-party plugin naming convention (`lerobot_robot_*`,
`lerobot_teleoperator_*`), so LeRobot CLIs and the async robot client
auto-discover them when installed — no manual import or registration step.

See [Imitation Learning on Real Robots](https://huggingface.co/docs/lerobot/il_robots) for LeRobot's native deployment workflow. The `policy_runner_node` adds ROS2 action-based lifecycle management on top of this, which is convenient if your workflow is already ROS2-centric.

---

## Contract Reference

A contract is a YAML file that maps ROS2 topics to LeRobot's observation/action interface. The contract currently maps to the full LeRobot `EnvTransition` interface:

| Contract Section | EnvTransition Slot | Status |
|-----------------|-------------------|--------|
| `observations` | `observation.*` | Supported |
| `actions` | `action*` | Supported |
| `tasks` | `complementary_data.task` | Supported |
| `rewards` | `next.reward` | Supported |
| `signals` | `next.done`, `next.truncated` | Supported |
| `complementary_data` | `complementary_data.*` | Supported |

Not every section needs to be filled for every robot. A minimal contract only needs `observations` and `actions`. To see which keys are required or accepted by different policies, see [Policy Feature Compatibility](#policy-feature-compatibility).

### Minimal Example

```yaml
robot_type: my_robot
robot_interface: ros2
fps: 30

observations:
  observation.state:
    channel: {topic: /joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: header}
    select: [position.j1, position.j2]

actions:
  action:
    channel: {topic: /joint_commands, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: header}
    select: [position.j1, position.j2]
```

`robot_type`, `robot_interface` (only `ros2` today), and `fps` are required.
Top-level keys starting with `x-` are ignored — use them to hold shared YAML
anchors (e.g. an `x-qos:` block of reusable QoS profiles; see
`contracts/stone.yaml`).

### Observations

```yaml
observations:
  # State vector (with all optional fields shown)
  observation.state:
    channel:
      topic: /joint_states
      type: sensor_msgs/msg/JointState
      qos: {reliability: best_effort, depth: 10}
      dtype: float64            # optional; defaults to the codec's native dtype
    align:
      strategy: hold            # hold | asof | drop — mandatory, no default
      timeline: header          # a timeline the channel provides — mandatory
    select: [position.j1, velocity.j1]
    apply: [rad2deg]            # optional operator pipeline (see Operators)
```

Data on a channel can carry several timestamps at once; the robot interface
produces them as named **timelines** and `align.timeline` selects one by
name. Every ros2 channel provides `receive` (arrival time at the node); a
message type carrying a std_msgs `Header` also provides `header` (sensor
time — more accurate, but requires publishers to stamp correctly and hosts
to be time-synced). Naming a timeline the channel cannot provide is a
load-time error, and a header-timeline message that arrives unstamped is
dropped at ingest — never silently re-timed.

`align.strategy` picks how samples land on the frame clock: `hold` carries
the last value forward, `asof` holds only within `tolerance_ms` (required,
and only valid, with `asof`), `drop` gaps out anything older than one frame.

```yaml

  # Camera (resize is an operator in the apply pipeline; encoding hints live in the channel)
  observation.images.camera:
    channel: {topic: /camera/image_raw/compressed,
              type: sensor_msgs/msg/CompressedImage}
    align: {strategy: hold, timeline: header}
    apply: [resize: [224, 224]]  # [height, width]
```

A list value under one key declares ordered sources whose values are
concatenated (see [The Contract](#the-contract)).

### Actions

```yaml
actions:
  action:
    channel:
      topic: /joint_commands
      type: sensor_msgs/msg/JointState
      qos: {reliability: reliable, depth: 10}
      safety: hold              # none (default) | hold | zeros
    align: {strategy: hold, timeline: header}
    select: [position.j1, position.j2]
    apply: [rad2deg]            # only serveable operators allowed on actions
```

`channel.safety` is the stop behavior published by the watchdog and on
deactivate. The default is `none` because a fabricated zero command is only a
safe stop under velocity control — under position control (the common case)
it commands a slam to the zero pose. Opt in explicitly per channel: `zeros`
for velocity-controlled robots (e.g. a Twist base), `hold` where re-sending
the last command is safe.

Actions read the same pipeline right-to-left: recording decodes from the
channel, serving encodes to it. A list value splits one action vector across
channels in order (see `contracts/stone.yaml`), each with its own safety
behavior and align.

### Field kinds (`kind`)

`kind` is an optional tag on an observation or action spec that describes what
the value's representation is. LeRobot ignores it. The vla_foundry / starVLA
adapters use it to pick per-group normalization and rotation handling. The
default leaves every existing contract unchanged.

`kind` is a single token (default `continuous`):

| value | dims | meaning |
|---|---|---|
| `continuous` (default) | any | plain scalar/vector (joints, positions, velocities) |
| `quaternion` | 4 | rotation `[x, y, z, w]` |
| `euler_rpy` | 3 | roll / pitch / yaw |
| `axis_angle` | 3 | rotation vector |
| `rotation_6d` | 6 | 6-D continuous rotation |
| `binary` | any | discrete on/off (e.g. gripper) |

**Keys stay canonical.** Don't encode the type in the key. `action.binary`
becomes a separate LeRobot feature and breaks policies that read `action`.
Keep the canonical key and split a mixed vector into one spec per kind. The
specs share the key and concatenate into one flat feature, each with its own
`kind`:

```yaml
observations:
  observation.state:                # canonical key; one flat vector to LeRobot
    - channel: {topic: /ee_pose, type: geometry_msgs/msg/PoseStamped}
      align: {strategy: hold, timeline: header}
      select: [pose.position.x, pose.position.y, pose.position.z]
      # kind: continuous (default)
    - channel: {topic: /ee_pose, type: geometry_msgs/msg/PoseStamped}
      align: {strategy: hold, timeline: header}
      select: [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
      kind: quaternion
    - channel: {topic: /gripper, type: sensor_msgs/msg/JointState}
      align: {strategy: hold, timeline: header}
      select: [position.gripper]
      kind: binary
```

`kind` is per-source, so a mixed vector carries one tag per slice.
Validation runs at contract load: the dim count must match the `kind`
(`quaternion` is 4, etc.), and an untagged `x/y/z/w` run warns ("looks like a
quaternion, set `kind: quaternion`") so a rotation isn't silently min-max'd.

### Backend notes (LeRobot / vla_foundry / starVLA)

One contract drives all three frameworks. Only `kind` is VLA-specific
(LeRobot ignores it). A few differences matter:

- **Training is framework-native for all three.** Rosetta produces the dataset
  and hosts deploy. You train with each framework's own tools: LeRobot with
  `lerobot-train` on the dataset, vla_foundry with its trainer on the tar shards,
  starVLA from the starVLA repo after pasting the generated `rosetta_dataconfig.py`.
- **Deploy topology.** LeRobot runs a gRPC policy-server subprocess. vla_foundry
  loads the model in-process in the ROS node (heavy on the robot). starVLA is a
  websocket client to a separate server process (GPU box, possibly remote).
- **State/action layout.** LeRobot keeps one feature per contract key, so
  multi-key state like `observation.state` + `observation.environment_state` stays
  separate. The VLA writers concatenate all state into one `observation.state`
  column (starVLA preserves the per-`kind` slices via `meta/modality.json`). A
  model needing separate state inputs loses that split on the VLA path.
- **Sections consumed.** The VLA writers use observations and actions only. The
  `rewards` / `signals` / `info` / `complementary_data` / `adjunct` sections
  (LeRobot RL and record-only) are not part of the VLA dataset.
- **`fps` consistency.** `fps` sets the control rate and action time-horizon. Keep
  it identical across record, train, and deploy or the policy degrades.

### Train/deploy skew

Rosetta aims to make what a policy trains on match what it sees live. Offline
conversion (`bag_frames`) and online inference (`topic_bridge`) run the same
`StreamBuffer` resampling, `aggregate_frame`, and operator pipeline from the
same contract, so decode, `select`, `apply` operators, alignment, and key
aggregation match
across all three frameworks.

Downstream, per framework:

- **LeRobot.** Normalization is policy-internal, with the same dataset stats at
  train and infer time. The only residual is the video codec (mp4 dataset vs raw
  online), inherent to any video dataset.
- **vla_foundry.** Inference reuses the training `RoboticsProcessor` and
  normalizer, so input and output normalization match training. Make sure the
  online image-history matches the writer's `image_indices` windowing.
- **starVLA.** The server un-normalizes the output. The stock server leaves
  input-state normalization to the client. The Rosetta runner stays starVLA-free
  and sends raw state, so a state-conditioned model would see skew. Use
  `scripts/rosetta_starvla_server.py` (the default `Dockerfile.starvla` entrypoint),
  which normalizes input state server-side with the training transform.
  Vision+language-only models send no state and are unaffected. Also keep the
  runner's `image_width/height` equal to the model's training `obs_image_size`.

### Operators

`apply` is an ordered operator pipeline run after `select` (field projection).
On the **record/decode** path operators run front-to-back via their forward
direction;
on the **serve/encode** path (policy command → ROS) they run back-to-front
via their inverse. (Select/apply are pure per-message transforms, so they
commute with alignment — the runtime applies them once per message at
ingest, which is observationally identical to the contract's
channel → align → select → apply reading.)

Each operator declares an **invertibility tier** that governs where it may run:

| Tier | Meaning | On actions? | Round-trip gate |
|------|---------|:-----------:|:---------------:|
| `FORWARD_ONLY` | decode/build only; lossy and one-way | rejected at load | — |
| `BIDIRECTIONAL` | runs both ways but lossy (it applies a bound) | allowed | — |
| `BIJECTIVE` | inverse exactly undoes forward | allowed | verified at load |

An action's `apply` may only contain serveable operators (`BIDIRECTIONAL` or
`BIJECTIVE`). A `FORWARD_ONLY` operator on an action is rejected at contract
load. A `BIJECTIVE` operator is round-trip verified at load: it will not load
unless
`inverse(forward(x)) == x`, so a wrong inverse fails at load instead of
corrupting actions silently.

| Operator | Form | Tier | Notes |
|----|------|------|-------|
| `rad2deg` | `rad2deg` | `BIJECTIVE` | radians (ROS) ↔ degrees (dataset) |
| `clamp` | `clamp: {min: lo, max: hi}` | `BIDIRECTIONAL` | clip element-wise, preserving the input dtype; on actions this bounds the outgoing command (lossy, so not bijective) |
| `resize` | `resize: [h, w]` | `FORWARD_ONLY` | nearest-neighbor image resize (integers, ≤ 8192); image observations only; declares the stream's output geometry |

```yaml
apply: [rad2deg, clamp: {min: -180, max: 180}]   # convert then bound
apply: [resize: [224, 224]]                      # image resize (observations only)
```

Operators bound values; they do not repair them. `clamp` passes NaN through
(`np.clip` semantics), so the encode path itself refuses non-finite values: a
NaN/Inf command from a diverged policy is never published — the frame is
dropped whole (no partial frame across a multi-channel action) with a
throttled error, and if the condition persists the watchdog applies each
channel's declared `safety` behavior. A recovered policy resumes seamlessly.

Add a built-in capability by registering a new operator in
`rosetta/contract/builtin_operators.py`; the framework
(`rosetta/contract/operators.py`) and the contract schema do not change:

```python
from rosetta.contract.operators import register_operator, Operator, Invertibility

@register_operator("my_op", kind=Invertibility.BIJECTIVE)
class MyOperator(Operator):
    def forward(self, arr): ...
    def inverse(self, arr): ...   # round-trip verified at contract load
```

Registration enforces the tier's promises up front: a serveable tier
(`BIDIRECTIONAL`/`BIJECTIVE`) without an `inverse` is rejected at import, and
a name that is already registered raises unless you pass `override=True`
(same rule as the codec registries) — a plugin cannot silently shadow a
built-in like `clamp`. An operator that fixes image geometry declares it by
setting `self.output_hw = (h, w)`; image observations require some operator
in their pipeline to declare it (built-in `resize` does).

**Bring your own operator as a plugin**, no fork needed. Package your operator
and advertise it under the `rosetta.operators` entry-point group. Rosetta
discovers and imports it at contract load, so the contract references it by
name only (no module paths in the YAML):

```toml
# in your plugin's pyproject.toml
[project.entry-points."rosetta.operators"]
my_operators = "my_pkg.my_operators"   # imported once; its @register_operator calls run
```
```yaml
apply: [my_op]             # resolves to the plugin's registered operator
```

### Teleop

For human-in-the-loop recording with a leader arm or other input device.
Teleop uses fixed **role** sections — `input`, `events`, `feedback` — instead
of frame keys. `input`/`feedback` are each a *list* of independently-targeted
sources, not a single block: every entry names the actions/observations
section topic it corresponds to (`target`/`origin`), validated at load time,
so a contract can teleop one action, several, or none, without touching the
`actions`/`observations` sections themselves:

```yaml
teleop:
  input:
    - target: /arm/joint_commands   # names an existing action channel's topic
      channel: {topic: /leader_arm/joint_states, type: sensor_msgs/msg/JointState}
      align: {strategy: hold, timeline: header}
      select: [position.j1, position.j2]

  events:                 # edge-triggered; no align — events are not resampled
    channel: {topic: /joy, type: sensor_msgs/msg/Joy}
    select:               # event_name -> button/axis path
      is_intervention: buttons.5
      success: buttons.0
      end_success: buttons.6
      end_failure: buttons.7
      failure: buttons.1

  feedback:
    - origin: /arm/joint_states      # names an existing observation channel's topic
      channel: {topic: /leader_arm/effort_feedback, type: sensor_msgs/msg/JointState}
      align: {strategy: hold, timeline: receive}
      select: [effort.j1, effort.j2]
```

`hil_manager_node` drives each `input` entry live: it decodes the
teleop message and encodes+publishes it onto `target` using the same
decode/encode machinery as everything else in the contract (so a leader
device with different fields, units, or a resampled timeline than the action
it drives still works correctly — not a raw byte-for-byte republish). The
action's own topic therefore already carries the human's command during
teleop; `input`/`feedback` are additionally exposed as their own
`teleop.input.<action key>` / `teleop.feedback.<observation key>` dataset
columns by the porter, for diagnostics. Feedback runs the mirror direction
(observation -> encode -> publish to the human device) regardless of mux
state. Feedback channels never declare `safety` (a teleop device gets no
fabricated commands — declaring it is a load error). See `contracts/stone.yaml`
for a full worked example.

### Tasks, Rewards, and Signals

These sections are optional. Use them when your workflow requires task prompts from ROS2 topics, RL reward signals, or episode termination signals.

```yaml
tasks:
  task:                       # not on the frame clock — no align
    channel: {topic: /task_prompt, type: std_msgs/msg/String}

rewards:
  next.reward:                # extended sections: dtype is mandatory
    channel: {topic: /reward, type: std_msgs/msg/Float64, dtype: float64}
    align: {strategy: hold, timeline: receive}

signals:
  next.done:
    channel: {topic: /episode_done, type: std_msgs/msg/Bool, dtype: bool}
    align: {strategy: hold, timeline: receive}

  next.truncated:
    channel: {topic: /episode_truncated, type: std_msgs/msg/Bool, dtype: bool}
    align: {strategy: hold, timeline: receive}
```

The extended sections (`rewards`, `signals`, `info`, `complementary_data`)
are ordinary frame entries with three extra rules: `dtype` is mandatory, they
are never images, and they are record-only (written to bags and datasets,
never fed to a policy at inference).

Task labels are **per-frame**: during conversion, each frame's `task` is the latest string received on a task topic at or before that frame (hold semantics), so the task can change mid-episode and LeRobot stores a per-frame `task_index` accordingly. Frames before the first task message — or recordings with no task topic at all — fall back to the `prompt` argument passed when recording, so you don't need a ROS2 topic for single-task episodes. At inference, the task comes from the `RunPolicy` goal.

### Topic Recording

By default, the episode recorder records **every topic** on the ROS2 graph, not just those declared in the contract. Contract topics (observations, actions, etc.) are required to be present, but everything else is captured automatically so you never lose data you might need later. This behaves like `ros2 bag record -a`.

To exclude topics, pass the `exclude_topics` parameter to the recorder node.

```python
# In a launch file
Node(
    package='rosetta',
    executable='episode_recorder_node',
    parameters=[{
        'contract_path': '/path/to/contract.yaml',
        'exclude_topics': ['/camera/.*/debug', '/diagnostics'],
    }],
)
```

Only `/rosout`, `/parameter_events`, and the recorder's own service topics are excluded automatically.

To disable auto-recording and only record contract-declared topics, set the recorder's `record_all` parameter to `false` (in `params/episode_recorder.yaml` or per-launch).

### Adjunct Topics

Adjunct topics are recorded to the bag file but have no LeRobot feature mapping. Unlike auto-discovered topics, adjunct topics are considered **required** to be present at record time.

```yaml
adjunct:
  - channel: {topic: /tf, type: tf2_msgs/msg/TFMessage}
  - channel: {topic: /diagnostics, type: diagnostic_msgs/msg/DiagnosticArray}
  - channel: {topic: /imu/raw, type: sensor_msgs/msg/Imu}
```

Because the bag preserves this data, you can always add a contract mapping for these topics later and re-run the porter without re-recording.

### Select Syntax

`select` is a flat list of dot-notation paths that extract nested fields from
ROS2 messages:

```yaml
# JointState: {field}.{joint_name}
select: [position.shoulder, velocity.shoulder]

# Odometry: nested path
select: [twist.twist.linear.x, pose.pose.position.z]
```

### Alignment Strategies

| Strategy | Behavior |
|----------|----------|
| `hold` | Use most recent message, no matter how old |
| `asof` | Use most recent message only if within the `tolerance_ms` window, otherwise a gap. Useful for rejecting stale data |
| `drop` | Use most recent message only if it arrived within the current step/frame window, otherwise a gap |

Every frame-clock entry declares one explicitly — there is no default.

**What a gap looks like.** Before **warmup**, no frames are emitted at all:
recording and inference start only once every *observation* stream has
produced at least one sample (actions and extended sections may legitimately
start late or publish sparsely). After warmup, a stream with no sample at a
tick **zero-fills** at its static dim — a zero vector for numerics, a zero
image, `""` for strings — so every frame always has the declared shape. This
is deliberate and currently not configurable: live inference cannot skip a
tick (the policy needs an observation every step; staleness is handled by
missing-stream logging and the action safety watchdog, not by the frame
shape), and offline, dropping frames would silently break the `fps` grid the
dataset declares. Bag conversion and the live bridge share this behavior, so
a gap looks identical in training data and at inference. Note that with
`hold`, post-warmup gaps essentially never occur — zero-fill shows up with
`asof`/`drop` and on sparse extended streams (e.g. a reward published once
per episode holds after its first message, zero-filled before it).

### Supported Message Types

| Type | Extracted Fields |
|------|------------------|
| `sensor_msgs/msg/JointState` | position, velocity, effort by joint name |
| `sensor_msgs/msg/Image` | RGB uint8 array |
| `sensor_msgs/msg/CompressedImage` | Decoded to RGB uint8 |
| `geometry_msgs/msg/Twist` | linear.xyz, angular.xyz |
| `nav_msgs/msg/Odometry` | pose, twist fields |
| `sensor_msgs/msg/Joy` | axes, buttons arrays |
| `sensor_msgs/msg/Imu` | orientation, angular_velocity, linear_acceleration |
| `std_msgs/msg/Float32` | Scalar float32 |
| `std_msgs/msg/Float64` | Scalar float64 |
| `std_msgs/msg/String` | Text string |
| `std_msgs/msg/Bool` | Boolean |
| `std_msgs/msg/Float64MultiArray` | Vector float64 |

**When to write `dtype`.** A stream's dtype resolves by precedence — explicit
`channel.dtype` > `video` (image keys) > `float64` (custom decoders) > the
codec registry's native dtype — so most entries never declare it. You write
it in exactly three situations:

1. **Custom decoder** — the registry can't know what your function returns
   (without it, custom-decoded streams are assumed `float64`);
2. **Multi-source keys** — all sources of one key must resolve to a single
   dtype, and different codecs have different natives (e.g. turtlebot3 pins
   `dtype: float64` on `/imu` and `/odom` to match the JointState codec);
3. **Extended sections** (`rewards`/`signals`/`info`/`complementary_data`) —
   mandatory, since record-only columns have no other type source.

### Custom Encoders/Decoders

Add support for ROS message types beyond the built-ins by writing custom
decoders (ROS → numpy) and encoders (numpy → ROS). Codecs are keyed by message
type in a registry. There are three ways to register yours.

#### Method 1: Plugin via entry points (recommended)

Package your codecs and advertise them under the `rosetta.codecs` entry-point
group. Rosetta discovers and imports the module at contract load, running its
`@register_*` decorators, so the contract names the type only, with no module
paths in the YAML:

```python
# my_pkg/my_codecs.py
import numpy as np
from rosetta.frames.codecs import register_decoder, register_encoder

@register_decoder("my_msgs/msg/MyCustomSensor", dtype="float64")
def decode_my_sensor(msg, spec):
    return np.array([msg.field1, msg.field2], dtype=np.float64)

@register_encoder("my_msgs/msg/MyCustomCommand")
def encode_my_command(values, spec, stamp_ns=None):
    from my_msgs.msg import MyCustomCommand
    msg = MyCustomCommand()
    msg.field1, msg.field2 = float(values[0]), float(values[1])
    return msg
```
```toml
# in your plugin's pyproject.toml
[project.entry-points."rosetta.codecs"]
my_codecs = "my_pkg.my_codecs"
```
```yaml
observations:
  observation.state:
    channel: {topic: /my_sensor,
              type: my_msgs/msg/MyCustomSensor}  # codec self-registered; no path needed
    align: {strategy: hold, timeline: receive}
```

Registering a second codec for a type that already has one is an error, so two
plugins can't silently conflict over a type. To replace a built-in (e.g. you
wrote a better `sensor_msgs/msg/Image` decoder), pass `override=True`:

```python
@register_decoder("sensor_msgs/msg/Image", dtype="video", override=True)
def my_better_image_decoder(msg, spec): ...
```

#### Method 2: Inline path in the contract (per-spec override)

Point a single spec directly at a codec function. It applies to that spec
only, overrides the registry there, and needs no packaging. Use it for a one-off,
or to use a different decoder on one topic while the registry default applies
elsewhere:

```yaml
actions:
  action:
    channel:
      topic: /my_command
      type: my_msgs/msg/MyCustomCommand
      decoder: my_package.codecs:decode_my_command  # module:function (reading bags)
      encoder: my_package.codecs:encode_my_command  # for publishing
    align: {strategy: hold, timeline: receive}
```

The module must be importable; paths are validated at contract load time.

> **Trust model — a contract is code-equivalent.** Loading a contract *imports*
> every `decoder:`/`encoder:` module it names (that's what validates the path)
> and, at runtime, invokes those functions on robot message data. Only load
> contracts you trust. This matters most for the policy runner's sidecar
> fallback, which can fetch `rosetta_contract.yaml` from a Hugging Face Hub
> model/dataset repo: a contract downloaded from a third-party repo is treated
> as trusted input, exactly like a launch file.

> **Round-trip safety:** every encoder/decoder pair is round-trip tested
> (`decode(encode(v)) == v`). A new pair must declare its round-trip behavior
> (`register_encoder(..., roundtrip=False)` for a lossy encoder) and add a sample
> to the test suite, or the tests fail.

#### Function Signatures

**Decoder:** Converts ROS message → numpy array

```python
def my_decoder(msg, spec) -> np.ndarray:
    # msg: ROS message instance
    # spec.names: list of selected field paths from the contract
    # spec.source.channel.type: ROS message type string
    return np.array([...], dtype=np.float64)
```

**Encoder:** Converts numpy array → ROS message

```python
def my_encoder(values, spec, stamp_ns=None):
    # values: numpy array of action values (decode_value/encode_value already ran
    #         spec.operators in the serve/inverse direction, e.g. clamp/deg2rad,
    #         before your encoder sees them)
    # spec.names: list of selected field paths from the contract
    # stamp_ns: optional timestamp in nanoseconds
    msg = MyMessage()
    # ... populate msg from values ...
    return msg
```

#### When Each Is Used

| Field | Used By | Purpose |
|-------|---------|---------|
| `decoder` on observations | Runtime, porter | Decode incoming sensor data |
| `decoder` on actions | porter | Read recorded actions from bags |
| `encoder` on actions | Runtime | Publish actions to ROS topics |

---


## LeRobot Data Model Reference

This section covers LeRobot's internal data model in detail. You don't need this to get started. Refer back here when you need to understand key conventions, feature types, or policy compatibility.

### Key System


**LeRobot keys are flat dictionary strings that use dots as a naming convention.** `observation.state.joint_position` is a single string key, not a nested lookup. The only hard rule is **no forward slashes** (`/`) in key names.

This means you can create keys at any depth:

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

There is no parent-child relationship between these keys. `observation.state` and `observation.state.joint_position` can coexist as completely independent features with different shapes. They just happen to share a prefix.

#### How LeRobot classifies keys

While keys are free-form strings, LeRobot policies use **prefix matching** to classify them into feature types. This classification determines how policies process each feature:

| Prefix | FeatureType | How policies use it |
|--------|-------------|---------------------|
| `observation.images.*` or `observation.image` | `VISUAL` | Fed through vision encoder |
| `observation.environment_state` (exact) | `ENV` | Separate encoder projection (privileged sim state) |
| `observation.*` (everything else under observation) | `STATE` | Robot state encoder |
| `observation.language.*` | `LANGUAGE` | Tokenized text for VLA forward pass |
| `action*` | `ACTION` | Policy output / training target |
| `next.reward` | `REWARD` | RL reward signal |

This means `observation.state.imu`, `observation.state.joint_position`, and `observation.state` are all classified as `STATE`. Similarly, `action.arm` and `action.gripper` are both `ACTION`.

#### Convention vs. compatibility

LeRobot's key system has two layers:

1. **The dataset format** accepts any key string. You can store `observation.state.fake_sensor.special_data` or `my_custom_thing` and it works.
2. **Built-in policies** look for specific keys by exact match. ACT, SmolVLA, and Pi0 all expect `observation.state` and `action` as single combined vectors.

The [DROID dataset](https://huggingface.co/datasets/lerobot/droid) demonstrates the recommended approach when you need both richness and compatibility: **store split sub-keys alongside combined keys**:

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

### EnvTransition

LeRobot defines a [Universal Data Container](https://huggingface.co/docs/lerobot/introduction_processors#envtransition-the-universal-data-container) that descends from the classic Gymnasium `step()` return (`observation, reward, terminated, truncated, info`), called `EnvTransition`.

The `EnvTransition` TypedDict defines six top-level slots. The contract aims to make explicit the mapping between ROS2 and the semantic categories defined by the EnvTransition. No core policy currently leverages all components.

#### Observation (`observation.*`)

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

**`observation.state`** vs **`observation.environment_state`**: These are semantically distinct. `state` is the robot's proprioception, i.e. what the robot knows about its own body (joint angles, gripper width, EEF pose). `environment_state` is privileged information about the external world (object positions, contact forces), typically only available in simulation. They have different `FeatureType`s (`STATE` vs `ENV`) and policies encode them with separate projections.

#### Action (`action*`)

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

#### Task and Language

These serve different purposes and can coexist:

| Concept | Key(s) | Type | Purpose |
|---------|--------|------|---------|
| **Task string** | `task` | `str` | Human-readable label: `"pick up the red block"` |
| **Language tokens** | `observation.language.tokens` | `Tensor (int)` | Tokenized text for VLA forward pass |
| **Language mask** | `observation.language.attention_mask` | `Tensor (bool)` | Attention mask for tokenized text |

The **flow** between them: the dataset stores a `task_index` (int) per frame, which resolves to a `task` string via `meta/tasks.parquet`. How that string reaches the policy depends on the policy:

- **Pre-tokenized** (SmolVLA, Pi0, Pi0Fast, Pi0.5, X-VLA): LeRobot's `TokenizerProcessorStep` reads the `task` string and produces `observation.language.tokens` and `observation.language.attention_mask` tensors. The policy consumes these tensors.
- **Internally tokenized** (GR00T, Wall-X): The raw `task` string is passed directly to the policy, which tokenizes it through its own VLM backbone (Eagle 2.5 for GR00T, Qwen 2.5-VL for Wall-X).

`task` is always a single string per frame. `subtask` is a recognized complementary data key.

#### Reward and Episode Signals

RL signals and episode boundaries:

```
next.reward                         # Scalar float: RL reward signal
next.done                           # Bool: episode terminated naturally (goal reached, failure)
next.truncated                      # Bool: episode ended artificially (time limit)
```

These use the `next.` prefix because they describe the outcome *after* taking the action.

#### Complementary Data

Per-frame metadata that flows through training but isn't a model input:

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

The `*_is_pad` flags mark which frames in a temporal window are real vs. padded (used when a policy looks at multiple past frames and some haven't occurred yet).

The five default features (`timestamp`, `frame_index`, `episode_index`, `index`, `task_index`) are automatically added to every dataset. You don't need to declare them.

#### Info

The `info` slot in `EnvTransition` is **runtime-only** and is not persisted to datasets. It carries transient signals like teleop events (`is_intervention`, `end_success`, `end_failure`) used during live recording and policy execution. If you need persistent metadata, use `complementary_data` instead.

Note: `meta/info.json` in the dataset directory is unrelated; it stores the dataset schema (features, fps, robot_type), not per-frame data.

### Data Types

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
| `nav_msgs/msg/Odometry` | `float64` | Selected pose/twist fields |
| `sensor_msgs/msg/Imu` | `float64` | Orientation, angular vel, linear accel |
| `std_msgs/msg/Float32` | `float32` | Scalar `(1,)` |
| `std_msgs/msg/Float64` | `float64` | Scalar `(1,)` |
| `std_msgs/msg/String` | `string` | Text `(1,)` |
| `std_msgs/msg/Bool` | `bool` | Boolean `(1,)` |
| `std_msgs/msg/Float64MultiArray` | `float64` | Vector `(N,)` |

You can override the auto-detected dtype with the `dtype` field in the contract, or use a [custom decoder](#custom-encodersdecoders-experimental) for non-standard message types.

### Policy Feature Compatibility

Each LeRobot policy implements its own `validate_features()` and accesses batch keys differently. There is no single enforced schema; what keys a policy accepts depends on the policy. This table summarizes the actual requirements based on the modeling code in `lerobot/src/lerobot/policies/`:

| Feature | ACT | SmolVLA | Pi0 | Pi0-Fast | Pi0.5 | GR00T N1.5 | Wall-X | X-VLA |
|---------|:---:|:-------:|:---:|:--------:|:-----:|:----------:|:------:|:-----:|
| **Type** | BC | VLA | VLA | VLA | VLA | VLA | VLA | VLA |
| **`observation.state`** | optional | **required** | optional | - | - | optional | **required** | optional |
| **`observation.environment_state`** | optional | - | - | - | - | - | - | - |
| **`observation.images.*`** | multi | multi | multi | multi | multi | multi | multi | multi |
| **`task` string** | - | **required** | **required** | **required** | **required** | **required** | **required** | **required** |
| **`action`** | **required** | **required** | **required** | **required** | **required** | **required** | **required** | **required** |
| **VLM backbone (params)** | - | SmolVLM2 (0.5B) | PaliGemma (3B / 0.7B) | PaliGemma (3B) | PaliGemma (3B / 0.7B) | Eagle 2.5 (3B) | Qwen 2.5-VL (7B) | Florence2 (0.7B / 0.2B) |
| **RTC support** | - | yes | yes | yes | yes | - | - | - |
| **Max state dim** | any | 32 | 32 | 32 | - | 64 | 20 | 32 |
| **Max action dim** | any | 32 | 32 | 32 | 32 | 32 | 20 | 20 |
| **Image size** | any | 512×512 | 224×224 | 224×224 | 224×224 | 224×224 | any | any |
| **Max language tokens** | - | 48 | 48 | 200 | 48 | 4096 | 768 | 64 |
| **Chunk size (default) [max]** | (100) | (50) | (50) | (50) | (50) | (50) [1024] | (32) | (32) [512] |
| **Async inference** | yes | yes | yes | - | yes | yes | - | - |



**Key dimensions:**

- **Max images**: All "multi" policies dynamically handle N cameras, configured at init time. However, no policy has truly unlimited image capacity. ACT concatenates image features, so the practical limit depends on the model's hidden dimension. VLA policies (Pi0 family, SmolVLA, Wall-X) feed images through a VLM, so the number of images is constrained by the VLM context window. For most robotics setups (2-3 cameras), this is probably not a bottleneck.
- **Max language tokens**: Maximum number of tokens the policy's tokenizer will keep from your task string. Longer prompts get truncated.
- **Chunk size**: Number of future action steps the policy predicts per inference call. Larger chunks mean fewer inference calls but less reactivity. Most policies build architecture (positional embeddings, pre-allocated tensors) to match the configured `chunk_size` at init time.
- **RTC (Real-Time Chunking)**: An [inference wrapper](https://huggingface.co/docs/lerobot/rtc) that improves real-time performance by overlapping action chunks with continuous re-planning. Only works with flow-matching policies (Pi0 family + SmolVLA).
- **Async inference**: Whether the policy is in LeRobot's gRPC-based asynchronous inference server allowlist (`SUPPORTED_POLICIES` in `async_inference/constants.py`). [Async](https://huggingface.co/docs/lerobot/rtc) decouples observation collection from action computation, which is useful for high-frequency control loops. Pi0-Fast, Wall-X, and X-VLA all implement `predict_action_chunk()` and are technically compatible, but haven't been added to the allowlist yet.

**VLA language pipeline**: All VLA policies require a `task` string (e.g., `"pick up the red block"`). In Rosetta, this comes from the `prompt` argument when recording or running a policy. The string gets tokenized into tensors automatically, either by LeRobot's `TokenizerProcessorStep` (a pipeline step that runs before the policy sees the data) or by the policy itself internally. From a Rosetta/ROS2 perspective, **you just provide the task prompt**.

**Subtask support**: LeRobot provides a `lerobot-annotate` [tool](https://huggingface.co/spaces/lerobot/annotate) for adding subtask annotations to recorded episodes (e.g., marking "reach for object", "grasp", "lift" within a longer task). These annotations are stored as `language_instruction` columns in the dataset. However, **no current action policy consumes subtask annotations**. They are used by [SARM](https://huggingface.co/docs/lerobot/sarm) (a reward model) to compute progress scores for [RA-BC](https://huggingface.co/docs/lerobot/sarm) weighted training of Pi0, Pi0.5, and SmolVLA.

#### What this means for your contract

The keys you define in your Rosetta contract determine which policies you can train with. Some practical guidance:

**Maximum compatibility**: if you want your dataset to work with the widest range of policies:

```yaml
observations:
  observation.state:                # Required by: SmolVLA, Wall-X
    channel: {topic: /joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: header}
    select: [...]

  observation.images.top:           # At least 1 image required by most policies
    channel: {topic: /camera/image_raw/compressed,
              type: sensor_msgs/msg/CompressedImage}
    align: {strategy: hold, timeline: header}
    apply: [resize: [480, 640]]

actions:
  action:                           # Required by all action policies
    channel: {topic: /joint_commands, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: header}
    select: [...]

# For VLA policies, also provide a task prompt when recording:
# ros2 action send_goal ... "{prompt: 'pick up the red block'}"
```

**For VLA fine-tuning**: add a second camera and ensure your recording prompts are descriptive:

```yaml
observations:
  # ... state and first camera as above ...

  observation.images.wrist.right:
    channel: {topic: /wrist_camera/image_raw/compressed,
              type: sensor_msgs/msg/CompressedImage}
    align: {strategy: hold, timeline: header}
    apply: [resize: [512, 512]]
```


## License

Apache-2.0
