# Contract Reference

A contract is a YAML file mapping ROS2 topics to LeRobot's observation/action interface. The contract covers the full LeRobot `EnvTransition` interface:

| Contract Section | EnvTransition Slot | Status |
|-----------------|-------------------|--------|
| `observations` | `observation.*` | Supported |
| `actions` | `action*` | Supported |
| `tasks` | `complementary_data.task` | Supported |
| `rewards` | `next.reward` | Supported |
| `signals` | `next.done`, `next.truncated` | Supported |
| `complementary_data` | `complementary_data.*` | Supported |

A minimal contract only needs `observations` and `actions`. For which keys each policy requires, see [LeRobot data model](lerobot-data-model.md#policy-feature-compatibility).

## Top level

```yaml
robot_type: my_robot
robot_interface: ros2
fps: 30
```

`robot_type`, `robot_interface` (only `ros2` today), and `fps` are required. Top-level keys starting with `x-` are ignored. Use them to hold shared YAML anchors, e.g. an `x-qos:` block of reusable QoS profiles (see `contracts/stone.yaml`).

## Observations

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

Data on a channel carries several timestamps at once. The robot interface produces them as named **timelines**, and `align.timeline` selects one by name. Every ros2 channel provides `receive` (arrival time at the node). A message type carrying a std_msgs `Header` also provides `header` (sensor time, more accurate, but publishers must stamp correctly and hosts must be time-synced). Naming a timeline the channel does not provide is a load-time error. A header-timeline message arriving unstamped is dropped at ingest, never silently re-timed.

`align.strategy` picks how samples land on the frame clock. `hold` carries the last value forward. `asof` holds only within `tolerance_ms` (required, and only valid, with `asof`). `drop` gaps out anything older than one frame.

```yaml
  # Camera (resize is an operator in the apply pipeline; encoding hints live in the channel)
  observation.images.camera:
    channel: {topic: /camera/image_raw/compressed,
              type: sensor_msgs/msg/CompressedImage}
    align: {strategy: hold, timeline: header}
    apply: [resize: [224, 224]]  # [height, width]
```

A list value under one key declares ordered sources whose values are concatenated (see [Write a contract](../how-to/write-a-contract.md#multi-source-keys)).

## Actions

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

`channel.safety` is the stop behavior published by the watchdog and on deactivate. Values: `none` (default, publish nothing), `zeros` (zero action vector, run through the inverse `apply` pipeline before encoding, so a `clamp` still bounds the command), `hold` (re-send the last command). **Warning:** under position control, `zeros` commands a slam to the zero pose. Use `zeros` only for velocity-controlled channels. Rationale: [why safety defaults to none](../explanation/contract-design.md#why-safety-defaults-to-none).

Actions read the same pipeline right-to-left: recording decodes from the channel, serving encodes to it. A list value splits one action vector across channels in order (see `contracts/stone.yaml`), each with its own safety behavior and align.

## Field kinds (`kind`)

`kind` is an optional tag on an observation or action source naming the value's representation. LeRobot ignores the tag. The vla_foundry / starVLA adapters use the tag to pick per-group normalization and rotation handling. The default leaves every existing contract unchanged.

`kind` is a single token (default `continuous`):

| value | dims | meaning |
|---|---|---|
| `continuous` (default) | any | plain scalar/vector (joints, positions, velocities) |
| `quaternion` | 4 | rotation `[x, y, z, w]` |
| `euler_rpy` | 3 | roll / pitch / yaw |
| `axis_angle` | 3 | rotation vector |
| `rotation_6d` | 6 | 6-D continuous rotation |
| `binary` | any | discrete on/off (e.g. gripper) |

**Keys stay canonical.** Do not encode the type in the key. `action.binary` becomes a separate LeRobot feature and breaks policies reading `action`. Keep the canonical key and split a mixed vector into one source per kind. The sources share the key and concatenate into one flat feature, each with an own `kind`:

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

`kind` is per-source, so a mixed vector carries one tag per slice. Validation runs at contract load: the dim count must match the `kind` (`quaternion` is 4, etc.), and an untagged `x/y/z/w` run warns ("looks like a quaternion, set `kind: quaternion`") so a rotation is never silently min-max normalized.

## Operators

`apply` is an ordered operator pipeline run after `select` (field projection). On the **record/decode** path operators run front-to-back via their forward direction. On the **serve/encode** path (policy command → ROS) operators run back-to-front via their inverse. Select and apply are pure per-message transforms, so they commute with alignment. The runtime applies them once per message at ingest, observationally identical to the contract's channel → align → select → apply reading.

Each operator declares an **invertibility tier** governing where the operator runs:

| Tier | Meaning | On actions? | Round-trip gate |
|------|---------|:-----------:|:---------------:|
| `FORWARD_ONLY` | decode/build only, lossy and one-way | rejected at load | — |
| `BIDIRECTIONAL` | runs both ways but lossy (applies a bound) | allowed | — |
| `BIJECTIVE` | inverse exactly undoes forward | allowed | verified at load |

An action's `apply` accepts only serveable operators (`BIDIRECTIONAL` or `BIJECTIVE`). A `FORWARD_ONLY` operator on an action is rejected at contract load. A `BIJECTIVE` operator is round-trip verified at load: the contract refuses to load unless `inverse(forward(x)) == x`. Rationale: [why action operators must invert](../explanation/contract-design.md#why-action-operators-must-invert).

| Operator | Form | Tier | Notes |
|----|------|------|-------|
| `rad2deg` | `rad2deg` | `BIJECTIVE` | radians (ROS) ↔ degrees (dataset) |
| `clamp` | `clamp: {min: lo, max: hi}` | `BIDIRECTIONAL` | clip element-wise, preserving the input dtype. On actions, bounds the outgoing command (lossy, so not bijective) |
| `resize` | `resize: [h, w]` | `FORWARD_ONLY` | nearest-neighbor image resize (integers, ≤ 8192). Image observations only. Declares the stream's output geometry |

```yaml
apply: [rad2deg, clamp: {min: -180, max: 180}]   # convert then bound
apply: [resize: [224, 224]]                      # image resize (observations only)
```

`clamp` passes NaN through (`np.clip` semantics). The encode path refuses non-finite values: a NaN/Inf command is never published, the frame drops whole (no partial frame across a multi-channel action) with a throttled error, and if the condition persists the watchdog applies each channel's declared `safety` behavior. A recovered policy resumes seamlessly. Rationale: [why operators refuse to repair values](../explanation/contract-design.md#why-operators-refuse-to-repair-values).

To register your own operator, see [Add custom operators](../how-to/add-custom-operators.md).

## Teleop

Teleop uses fixed **role** sections (`input`, `events`, `feedback`) instead of frame keys. `input`/`feedback` are each a *list* of independently-targeted sources, not a single block. Every entry names a topic from the `actions`/`observations` sections (`target`/`origin`), validated at load, so a contract teleops one action, several, or none, without touching the `actions`/`observations` sections themselves:

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

  # Event vocabulary is closed: is_intervention, start_episode, success,
  # failure, end_success, end_failure. An unknown event name is a load error.

  feedback:
    - origin: /arm/joint_states      # names an existing observation channel's topic
      channel: {topic: /leader_arm/effort_feedback, type: sensor_msgs/msg/JointState}
      align: {strategy: hold, timeline: receive}
      select: [effort.j1, effort.j2]
```

Feedback channels never declare `safety`: a teleop device gets no fabricated commands, and declaring `safety` on feedback is a load error. See `contracts/stone.yaml` for a full worked example, and [Set up teleop and HIL](../how-to/teleop-and-hil.md) for runtime behavior.

On the LeRobot-native path (`lerobot_teleoperator_rosetta`), contract events map onto lerobot's `TeleopEvents`: `is_intervention`, `success`, and `failure` map one to one, `end_success`/`end_failure` assert the reward event plus `TERMINATE_EPISODE` together, and `start_episode` has no lerobot counterpart (ignored on that path, handled by `hil_manager_node`).

## Tasks, rewards, and signals

These sections are optional. Use them for task prompts from ROS2 topics, RL reward signals, or episode termination signals.

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

The extended sections (`rewards`, `signals`, `info`, `complementary_data`) are ordinary frame entries with three extra rules: `dtype` is mandatory, they are never images, and they are record-only (written to bags and datasets, never fed to a policy at inference).

Task labels are **per-frame**. During conversion, each frame's `task` is the latest string received on a task topic at or before the frame (hold semantics), so the task changes mid-episode and LeRobot stores a per-frame `task_index` accordingly. Frames before the first task message, or recordings with no task topic at all, fall back to the `prompt` argument passed when recording. Single-task episodes need no ROS2 topic. At inference, the task comes from the `RunPolicy` goal.

## Adjunct topics

Adjunct topics are recorded to the bag file with no LeRobot feature mapping. Unlike auto-discovered topics, adjunct topics are **required** at record time.

```yaml
adjunct:
  - channel: {topic: /tf, type: tf2_msgs/msg/TFMessage}
  - channel: {topic: /diagnostics, type: diagnostic_msgs/msg/DiagnosticArray}
  - channel: {topic: /imu/raw, type: sensor_msgs/msg/Imu}
```

The bag preserves this data, so add a contract mapping for these topics later and re-run the porter without re-recording.

## Select syntax

`select` is a flat list of dot-notation paths extracting nested fields from ROS2 messages:

```yaml
# JointState: {field}.{joint_name}
select: [position.shoulder, velocity.shoulder]

# Odometry: nested path
select: [twist.twist.linear.x, pose.pose.position.z]
```

## Alignment strategies

| Strategy | Behavior |
|----------|----------|
| `hold` | Use most recent message, no matter how old |
| `asof` | Use most recent message only if within the `tolerance_ms` window, otherwise a gap. Useful for rejecting stale data |
| `drop` | Use most recent message only if it arrived within the current step/frame window, otherwise a gap |

Every frame-clock entry declares one explicitly. There is no default.

**What a gap looks like.** Before **warmup**, no frames are emitted. Recording and inference start once every *observation* stream has produced at least one sample. Actions and extended sections legitimately start late or publish sparsely. After warmup, a stream with no sample at a tick **zero-fills** at its static dim (a zero vector for numerics, a zero image, `""` for strings), so every frame has the declared shape. Not configurable. Bag conversion and the live bridge share this behavior, so a gap looks identical in training data and at inference. With `hold`, post-warmup gaps rarely occur. Zero-fill shows up with `asof`/`drop` and on sparse extended streams (e.g. a reward published once per episode holds after the first message, zero-filled before). Rationale: [why gaps zero-fill](../explanation/contract-design.md#why-gaps-zero-fill-instead-of-skipping-frames).

## Supported message types

| Type | Extracted Fields |
|------|------------------|
| `sensor_msgs/msg/JointState` | position, velocity, effort by joint name |
| `sensor_msgs/msg/Image` | RGB uint8 array |
| `sensor_msgs/msg/CompressedImage` | Decoded to RGB uint8 |
| `geometry_msgs/msg/Twist` | linear.xyz, angular.xyz |
| `geometry_msgs/msg/TwistStamped` | twist.linear.xyz, twist.angular.xyz |
| `nav_msgs/msg/Odometry` | pose, twist fields |
| `sensor_msgs/msg/Joy` | axes, buttons arrays |
| `sensor_msgs/msg/Imu` | orientation, angular_velocity, linear_acceleration |
| `control_msgs/msg/MultiDOFCommand` | values, values_dot by DOF name |
| `trajectory_msgs/msg/JointTrajectory` | first-point position/velocity/acceleration/effort by joint name |
| `std_msgs/msg/Float32` | Scalar float32 |
| `std_msgs/msg/Float64` | Scalar float64 |
| `std_msgs/msg/Int32` | Scalar int32 |
| `std_msgs/msg/Int64` | Scalar int64 |
| `std_msgs/msg/String` | Text string |
| `std_msgs/msg/Bool` | Boolean |
| `std_msgs/msg/Float32MultiArray` | Vector float32 |
| `std_msgs/msg/Float64MultiArray` | Vector float64 |
| `std_msgs/msg/Int32MultiArray` | Vector int32 |

For other types, see [Add custom codecs](../how-to/add-custom-codecs.md).

## When `dtype` is required

A stream's dtype resolves by precedence (explicit `channel.dtype` > `video` for image keys > `float64` for custom decoders > the codec registry's native dtype), so most entries never declare one. An explicit `dtype` is required in exactly three situations:

1. **Custom decoder.** The registry has no knowledge of the function's return type. Without a declared dtype, custom-decoded streams are assumed `float64`.
2. **Multi-source keys.** All sources of one key must resolve to a single dtype, and different codecs have different natives (e.g. turtlebot3 pins `dtype: float64` on `/imu` and `/odom` to match the JointState codec).
3. **Extended sections** (`rewards`/`signals`/`info`/`complementary_data`). Mandatory: record-only columns have no other type source.

## Inline codec fields

`channel.decoder` / `channel.encoder` point one source at a codec function by `module:function` path, overriding the registry for that source only:

| Field | Used By | Purpose |
|-------|---------|---------|
| `decoder` on observations | Runtime, porter | Decode incoming sensor data |
| `decoder` on actions | porter | Read recorded actions from bags |
| `encoder` on actions | Runtime | Publish actions to ROS topics |

The module must be importable. Paths are validated at contract load time. See [Add custom codecs](../how-to/add-custom-codecs.md) for the trust model and function signatures.
