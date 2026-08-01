# Contract reference

A contract is a YAML file that maps ROS 2 topics to LeRobot's observation/action
interface. It covers the full LeRobot `EnvTransition` interface:

| Contract Section | EnvTransition Slot |
|-----------------|-------------------|
| `observations` | `observation.*` |
| `actions` | `action*` |
| `tasks` | `complementary_data.task` |
| `rewards` | `next.reward` |
| `signals` | `next.done`, `next.truncated` |
| `info`, `complementary_data` | record-only columns |

Not every section needs to be filled for every robot. A minimal contract only
needs `observations` and `actions`.

## Minimal example

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
Top-level keys starting with `x-` are ignored, so they can hold shared YAML
anchors such as an `x-qos:` block.

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
      strategy: hold            # hold | asof | drop (mandatory, no default)
      timeline: header          # a timeline the channel provides (mandatory)
    select: [position.j1, velocity.j1]
    apply: [rad2deg]            # optional operator pipeline

  # Camera
  observation.images.camera:
    channel: {topic: /camera/image_raw/compressed,
              type: sensor_msgs/msg/CompressedImage}
    align: {strategy: hold, timeline: header}
    apply: [resize: [224, 224]]  # [height, width]
```

`align.timeline` selects one of the timestamps the channel carries. Every ros2
channel provides `receive` (arrival time at the node). A message type carrying a
std_msgs `Header` also provides `header`. Naming a timeline the channel does not
provide is a load-time error, and a header-timeline message arriving unstamped
is dropped at ingest.

A list value under one key declares ordered sources whose values are
concatenated. Every source then needs a `select`, all sources must resolve to
the same dtype, and images never share a key.

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

Actions read the same pipeline right-to-left: recording decodes from the
channel, serving encodes to it. A list value splits one action vector across
channels in order, each with its own safety behavior.

`channel.safety` is the stop behavior published by the watchdog and on
deactivate: `none` publishes nothing, `zeros` publishes the zero action vector
run through the inverse `apply` pipeline, `hold` re-sends the last command.
**Under position control, `zeros` commands a slam to the zero pose.** `hold`
falls back to `zeros` when the channel has never published, so bound it with a
`clamp` in `apply`. With every channel on `none`, no watchdog runs at all.

## Key-count limit on the LeRobot live path

The porter writes one dataset feature per contract key. The LeRobot live path
cannot represent that: `hw_to_dataset_features` emits one hardcoded
`observation.state` for all numeric observations and one hardcoded `action` for
all actions. `lerobot_robot_rosetta` refuses such a contract at connect and at
`policy_runner_node`'s configure transition, so it ports and trains and then
fails to deploy.

Deployable through LeRobot means **at most one numeric observation key and at
most one action key**. Image keys are exempt, in any number. The limit counts
keys and not sources, so merging numeric streams under one key resolves it.
`load_contract` does not enforce this, since the contract layer stays
backend-neutral.

## Operators

`apply` is an ordered operator pipeline run after `select`. On the record path
operators run front-to-back through their forward direction; on the serve path
they run back-to-front through their inverse.

| Operator | Form | Tier | Notes |
|----|------|------|-------|
| `rad2deg` | `rad2deg` | `BIJECTIVE` | radians (ROS) ↔ degrees (dataset) |
| `clamp` | `clamp: {min: lo, max: hi}` | `BIDIRECTIONAL` | clip element-wise. On actions, bounds the outgoing command |
| `resize` | `resize: [h, w]` | `FORWARD_ONLY` | nearest-neighbor image resize. Image observations only |

An action's `apply` accepts only `BIDIRECTIONAL` or `BIJECTIVE` operators. A
`BIJECTIVE` operator is round-trip verified at load, so a wrong inverse fails
before deployment instead of corrupting actions silently. The encode path
refuses non-finite values: the frame drops whole and the watchdog applies the
declared `safety` if the condition persists.

## Field kinds (`kind`)

`kind` is an optional per-source tag naming the value's representation:
`continuous` (default), `quaternion` (4 dims), `euler_rpy` (3), `axis_angle`
(3), `rotation_6d` (6), `binary`. LeRobot ignores it; framework adapters use it
to pick normalization and rotation handling. Validation checks the dim count
against the kind at load. Do not encode the type in the key, since
`action.binary` becomes a separate LeRobot feature and breaks policies reading
`action`.

## Teleop

For human-in-the-loop recording with a leader arm or other input device:

```yaml
teleop:
  input:
    - target: /arm/joint_commands   # names an existing action channel's topic
      channel: {topic: /leader_arm/joint_states, type: sensor_msgs/msg/JointState}
      align: {strategy: hold, timeline: header}
      select: [position.j1, position.j2]

  events:                 # edge-triggered; no align, events are not resampled
    channel: {topic: /joy, type: sensor_msgs/msg/Joy}
    select:               # event_name -> button/axis path
      is_intervention: buttons.5
      success: buttons.0
      end_success: buttons.6
      end_failure: buttons.7
      failure: buttons.1

  feedback:
    - origin: /arm/joint_states     # names an existing observation channel's topic
      channel: {topic: /leader_arm/effort_feedback, type: sensor_msgs/msg/JointState}
      align: {strategy: hold, timeline: receive}
      select: [effort.j1, effort.j2]
```

The event vocabulary is closed: `is_intervention`, `start_episode`, `success`,
`failure`, `end_success`, `end_failure`. An unknown event name is a load error.
Feedback channels never declare `safety`.

## Tasks, rewards, and signals

These sections are optional. Use them when your workflow requires task prompts
from ROS 2 topics, RL reward signals, or episode termination signals.

```yaml
tasks:
  task:                       # not on the frame clock, so no align
    channel: {topic: /task_prompt, type: std_msgs/msg/String}

rewards:
  next.reward:                # extended sections: dtype is mandatory
    channel: {topic: /reward, type: std_msgs/msg/Float64, dtype: float64}
    align: {strategy: hold, timeline: receive}

signals:
  next.done:
    channel: {topic: /episode_done, type: std_msgs/msg/Bool, dtype: bool}
    align: {strategy: hold, timeline: receive}
```

The extended sections (`rewards`, `signals`, `info`, `complementary_data`) are
ordinary frame entries with three extra rules: `dtype` is mandatory, they are
never images, and they are record-only, never fed to a policy at inference.

Task labels are per-frame. For VLA policies the `task` string can also come from
the `prompt` argument when recording or running a policy, so no ROS 2 topic is
needed for it.

## Adjunct topics

Adjunct topics are recorded to the bag file but have no LeRobot feature mapping.
Unlike auto-discovered topics, adjunct topics are **required** to be present at
record time.

```yaml
adjunct:
  - channel: {topic: /tf, type: tf2_msgs/msg/TFMessage}
  - channel: {topic: /diagnostics, type: diagnostic_msgs/msg/DiagnosticArray}
```

## Select syntax

Dot notation extracts nested fields from ROS 2 messages:

```yaml
select: [position.shoulder, velocity.shoulder]       # JointState: {field}.{joint_name}
select: [twist.twist.linear.x, pose.pose.position.z] # Odometry: nested path
```

## Alignment strategies

| Strategy | Behavior |
|----------|----------|
| `hold` | Use most recent message, no matter how old |
| `asof` | Use most recent message only if within `tolerance_ms`, otherwise a gap |
| `drop` | Use most recent message only if it arrived within the current frame window |

Every frame-clock entry declares one explicitly. There is no default.

Before **warmup**, no frames are emitted: recording and inference start once
every observation stream has produced at least one sample. After warmup, a
stream with no sample at a tick **zero-fills** at its static dim, so every frame
has the declared shape. Bag conversion and the live bridge share this, so a gap
looks identical in training data and at inference.

## Supported message types

| Type | Auto dtype | Extracted fields |
|------|---|------------------|
| `sensor_msgs/msg/JointState` | `float64` | position, velocity, effort by joint name |
| `sensor_msgs/msg/Image` | `video` | RGB uint8 array |
| `sensor_msgs/msg/CompressedImage` | `video` | Decoded to RGB uint8 |
| `geometry_msgs/msg/Twist` | `float64` | linear.xyz, angular.xyz |
| `geometry_msgs/msg/TwistStamped` | `float64` | twist.linear.xyz, twist.angular.xyz |
| `nav_msgs/msg/Odometry` | `float64` | pose, twist fields |
| `sensor_msgs/msg/Joy` | `float32` | axes, buttons arrays |
| `sensor_msgs/msg/Imu` | `float64` | orientation, angular_velocity, linear_acceleration |
| `control_msgs/msg/MultiDOFCommand` | `float64` | values, values_dot by DOF name |
| `trajectory_msgs/msg/JointTrajectory` | `float64` | first-point position/velocity/effort |
| `std_msgs/msg/Float32`, `Float64`, `Int32`, `Int64` | matching | Scalar |
| `std_msgs/msg/String`, `Bool` | `string`, `bool` | Text, boolean |
| `std_msgs/msg/Float32MultiArray`, `Float64MultiArray`, `Int32MultiArray` | matching | Vector |

The dtype is auto-detected from the message type. Override it with the `dtype`
field, which is required for a custom decoder, for a multi-source key whose
sources have different natives, and in the extended sections. `video` is not a
selectable dtype; declaring anything else on an image key is a load error.

## Custom encoders and decoders

Add support for a message type beyond the built-ins by writing a decoder (ROS →
numpy) and, for actions, an encoder (numpy → ROS).

```python
from rosetta.frames.codecs import register_decoder, register_encoder

@register_decoder("my_msgs/msg/MyCustomSensor", dtype="float64")
def decode_my_sensor(msg, spec):        # spec.names holds the select list
    return np.array([msg.field1, msg.field2], dtype=np.float64)

@register_encoder("my_msgs/msg/MyCustomCommand")
def encode_my_command(values, spec, stamp_ns=None):   # values already ran spec.operators
    ...
```

Advertise the module under the `rosetta.codecs` entry-point group
(`rosetta.operators` for operators) and Rosetta imports it at contract load, so
the contract names the message type only. Registering a second codec for a
covered type is an error.

Alternatively point one source at a function by path:

```yaml
channel:
  topic: /my_command
  type: my_msgs/msg/MyCustomCommand
  decoder: my_package.codecs:decode_my_command   # module:function, for reading bags
  encoder: my_package.codecs:encode_my_command   # for publishing
```

The module must be importable. Paths are validated at contract load time.

> **A contract is code-equivalent.** Loading a contract *imports* every named
> `decoder:`/`encoder:` module and invokes those functions on robot message
> data. Only load contracts you trust. This matters most for the policy runner's
> sidecar fallback, which fetches `rosetta_contract.yaml` from a Hugging Face
> Hub repo.
