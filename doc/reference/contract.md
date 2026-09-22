# Contract

A contract is one YAML file. This page lists every key it accepts and every
rule the loader enforces. Message types, encoders and operators are on
[Message types and operators](message-types.md).

Loading a contract validates it. The loader imports every named codec
module, checks every operator, resolves every dtype and builds the frame
layout. With `rclpy` importable it also checks every type, timeline and QoS
against the installed ROS 2 interfaces. Without `rclpy` those three checks
wait until runtime. If it returns, the contract is valid in this
environment.

```bash
python -c "from rosetta.contract.schema import load_contract; load_contract('robot.yaml'); print('OK')"
```

Errors raise `rosetta.contract.errors.ContractValidationError`, a subclass of
`ValueError`.

## Top level

| Key | Required | Rule |
|---|---|---|
| `robot_type` | yes | Non-empty string. |
| `robot_interface` | yes | `ros2`. Case-insensitive. |
| `fps` | yes | Positive integer. An integral float such as `30.0` is accepted. |
| `observations` | no | Mapping of key to source or list of sources. |
| `actions` | no | Mapping of key to source or list of sources. |
| `rewards`, `signals`, `info`, `complementary_data` | no | Mapping of key to source or list. Record-only. |
| `tasks` | no | Mapping of key to `{channel}`. |
| `adjunct` | no | List of `{channel}`. |
| `teleop` | no | `input`, `events`, `feedback`. See [Teleop](#teleop). |
| `x-*` | no | Dropped before validation. Holds YAML anchors. |

An empty or null section is an error, so omit it instead. Unknown top-level
keys, duplicate keys anywhere in the file, and a key used in two sections are
errors too.

## Sections

| Section | Images | `dtype` | `align` | `safety`, `encoder` | Reaches the policy |
|---|---|---|---|---|---|
| `observations` | under `observation.images.*` only | optional | required | no | yes |
| `actions` | no | optional | required | yes | yes, as output |
| `rewards`, `signals`, `info`, `complementary_data` | no | required, never `video` | required | no | no |
| `tasks` | no | no | no | no | as the per-frame task string |
| `adjunct` | no | no | no | no | no, recorded only |
| `teleop.input` | no | optional | required | no | no, diagnostic column |
| `teleop.events` | no | no | no | no | no |
| `teleop.feedback` | no | optional | required | `encoder` only | no, diagnostic column |

An image key starts with `observation.images.`. An image key in any other
section is an error.

## Source

A source is a mapping with the keys `channel`, `align`, `select`, `apply`
and `kind`. Any other key is an error.

```yaml
observation.state:
  channel:
    topic: /joint_states
    type: sensor_msgs/msg/JointState
    qos: {reliability: reliable, history: keep_last, depth: 50}
    dtype: float64
  align: {strategy: hold, timeline: header}
  select: [position.shoulder_pan_joint, position.elbow_flex_joint]
  apply: [rad2deg]
  kind: continuous
```

### `channel`

| Key | Required | Rule |
|---|---|---|
| `topic` | yes | Non-empty string. |
| `type` | yes | ROS 2 type such as `sensor_msgs/msg/JointState`. Must import. |
| `qos` | no | See [QoS](#qos). |
| `dtype` | see Sections | One of `float32`, `float64`, `int32`, `int64`, `bool`, `string`, `video`. |
| `safety` | no | Actions only. `none` (default), `zeros`, `hold`. |
| `decoder` | no | `module.path:function`. Decoded sections only. See [Custom codecs](message-types.md#custom-codecs). |
| `encoder` | no | `module.path:function`. Actions and `teleop.feedback`. |

`dtype` resolves in this order: explicit value, `video` for an image key,
`float64` for a custom decoder, otherwise the decoder's native dtype. An
explicit non-video dtype on an image key is an error. `video` on a non-image
key is an error. A `tasks`, `adjunct` or `teleop.events` channel takes only
`topic`, `type` and `qos`.

A decoded source needs a built-in or custom decoder even with an explicit
`dtype`. An action or `teleop.feedback` source needs an encoder and a numeric
`dtype`.

`safety` is what the watchdog publishes when actions stop arriving for at
least two frame periods, and what deactivate publishes. `zeros` sends the zero vector
through the inverse `apply` pipeline. `hold` re-sends the last command and
sends zeros if nothing was sent yet. With every action channel on `none`, no
watchdog runs.

Don't put a position-controlled arm on `zeros`. Zero is a pose, and the arm
will go there.

### `qos`

| Key | Values | Default |
|---|---|---|
| `reliability` | `reliable`, `best_effort` | `reliable` |
| `durability` | `volatile`, `transient_local` | `volatile` |
| `history` | `keep_last`, `keep_all` | `keep_last` |
| `liveliness` | `automatic`, `manual_by_topic` | `automatic` |
| `depth` | integer | 10 |

Values are rclpy's short names, case-insensitive. rclpy's `system_default`,
`unknown` and `best_available` are accepted too. An unknown key or value is
an error. A `best_effort` publisher read with the default `reliable` delivers
nothing.

### `align`

Required on every frame-clock source. There is no default.

| Key | Rule |
|---|---|
| `strategy` | `hold`, `asof`, `drop`. |
| `timeline` | A timeline the channel provides. Case-sensitive. |
| `tolerance_ms` | Positive integer. Required with `asof`. An error with any other strategy. |

Timelines:

| Timeline | Provided by | Value live | Value offline |
|---|---|---|---|
| `receive` | every channel | node clock at arrival | the bag's message stamp |
| `header` | a type with a field `header` of type `std_msgs/Header` | `header.stamp` | `header.stamp` |

A `header` message whose stamp is `(0, 0)` is dropped and never re-stamped.
The drop is logged once, and again after the stream recovers and drops again.

Strategies, evaluated at each tick against the newest sample in the buffer:

| Strategy | Sample used |
|---|---|
| `hold` | The newest, at any age. |
| `asof` | The newest, if its age is at most `tolerance_ms`. Otherwise none. |
| `drop` | The newest, if it arrived within the last `1 / fps`. Otherwise none. |

A stamp ahead of the tick by up to `max(1 s, 2 ticks)` counts as age zero. A
stamp further ahead clears the buffer, as a clock reset.

No frame is produced until every observation source has a sample. That's
warmup, and action and record-only sources don't gate it. After warmup, a
source with no sample at a tick is zero-filled: numeric zeros at its width, a
black image at its size, or an empty string. On the live path the LeRobot
Robot plugin waits up to 5 s for warmup, then carries on with zero-filled
frames and a warning.

### `select`

A list of unique field paths. The order is the order in the frame. An empty
list is an error. Omit `select` to take the whole message.

`select` is required for `sensor_msgs/msg/JointState`, `sensor_msgs/msg/Imu`,
`nav_msgs/msg/Odometry`, `geometry_msgs/msg/Twist`,
`geometry_msgs/msg/TwistStamped`, `control_msgs/msg/MultiDOFCommand`,
`trajectory_msgs/msg/JointTrajectory` and `sensor_msgs/msg/Joy` unless the
channel names a custom `decoder` or `encoder`, and for every source of a
multi-source key. Field path syntax per type is on
[Message types and operators](message-types.md#decoders).

The width of a source is `len(select)`, or 1 without `select`.

### `apply`

An ordered list of operators. Each entry is a bare name or a one-key mapping.

```yaml
apply: [clamp: {min: -3.14159, max: 3.14159}, rad2deg]
```

Recording runs the list front to back through each operator's forward
direction. Serving runs it back to front through each inverse. In the example,
serving converts degrees to radians and then clamps radians.

An action or `teleop.feedback` source accepts only operators with an inverse.
A `resize` there is a load error. `apply` on a `string` source is an error.
Every image observation must carry an operator with a fixed output size,
which among the built-ins is `resize`. Operators are listed on
[Message types and operators](message-types.md#operators).

### `kind`

Optional. Names the value's representation and checks its width.

| `kind` | Width |
|---|---|
| `continuous` (default) | any |
| `quaternion` | 4 |
| `euler_rpy` | 3 |
| `axis_angle` | 3 |
| `rotation_6d` | 6 |
| `binary` | any |

A width that disagrees with `select` is an error. No shipped adapter reads
`kind`. A `continuous` select with an `x, y, z, w` run, or with `quat` in a
path, logs a warning.

## Multi-source keys

A list under one key declares ordered sources.

```yaml
observation.state:
  - channel: {topic: /arm/joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: header}
    select: [position.j1, position.j2]
  - channel: {topic: /gripper/state, type: std_msgs/msg/Float32}
    align: {strategy: hold, timeline: receive}
    select: [data]
```

Rules:

- Observations concatenate in order. Actions split in order.
- Every source needs `select`.
- Every source resolves to the same `dtype`. Set `dtype` explicitly when the
  natives differ.
- Images never share a key.
- Names get a per-topic prefix from the first topic segment that differs,
  for example `arm.position.j1` for `/arm/joint_states` and
  `/gripper/state`. A single-source key has no prefix. Two topics that
  normalize to the same name are an error.

## Tasks, rewards, signals, info, complementary_data

```yaml
tasks:
  task:
    channel: {topic: /task_prompt, type: std_msgs/msg/String}

rewards:
  next.reward:
    channel: {topic: /reward, type: std_msgs/msg/Float64, dtype: float64}
    align: {strategy: hold, timeline: receive}

signals:
  next.done:
    channel: {topic: /episode_done, type: std_msgs/msg/Bool, dtype: bool}
    align: {strategy: hold, timeline: receive}
```

A `tasks` channel has no `align`, and its type needs a string `data` field.
When porting, the frame's task at each tick is the newest string received at
or before the tick. With no `tasks` section, or before the first message, the
task is the prompt the episode was recorded with. Live, the task is the
`RunPolicy` goal's prompt.

`rewards`, `signals`, `info` and `complementary_data` are frame-clock sources
with `dtype` required and never `video`. They are recorded into the dataset
and never fed to a policy.

## Adjunct

```yaml
adjunct:
  - channel: {topic: /tf, type: tf2_msgs/msg/TFMessage}
  - channel: {topic: /tf_static, type: tf2_msgs/msg/TFMessage,
              qos: {durability: transient_local}}
```

Adjunct channels are recorded to the bag and never decoded. The recorder
subscribes to them as contract topics and reports any with zero messages at
episode end.

## Teleop

```yaml
teleop:
  input:
    - target: /forward_position_controller/commands   # an action channel's topic
      channel: {topic: /leader_arm/joint_states, type: sensor_msgs/msg/JointState}
      align: {strategy: hold, timeline: header}
      select: [position.shoulder_pan_joint, position.elbow_flex_joint]
  events:
    channel: {topic: /joy, type: sensor_msgs/msg/Joy}
    select:
      is_intervention: buttons.5
      success: buttons.0
      failure: buttons.1
      end_success: buttons.6
      end_failure: buttons.7
  feedback:
    - origin: /joint_states                           # an observation channel's topic
      channel: {topic: /leader_arm/effort_feedback, type: sensor_msgs/msg/JointState}
      align: {strategy: hold, timeline: receive}
      select: [effort.shoulder_pan_joint, effort.elbow_flex_joint]
```

| Key | Rule |
|---|---|
| `input[].target` | The topic of exactly one action entry. |
| `feedback[].origin` | The topic of exactly one observation entry. |
| `events.select` | Mapping of event name to a field path such as `buttons.N` or `axes.N`. The path is resolved per message, so a bad one is a runtime warning, not a load error. |

Event names are `is_intervention`, `start_episode`, `success`, `failure`,
`end_success`, `end_failure`. Any other name is an error. Events are
edge-triggered and never resampled. A `feedback` source must not declare
`safety`.

The porter writes `teleop.input` into the dataset as
`teleop.input.<action key>` and `teleop.feedback` as
`teleop.feedback.<observation key>`.

## Embedding

The recorder writes the contract text into the bag's `metadata.yaml` under
`rosbag2_bagfile_information.custom_data.rosetta.contract_yaml`, with the
prompt under `lerobot.operator_prompt`. The porter copies the contract it was
given to `meta/rosetta_contract.yaml` in the dataset. Neither copy is read for
decoding. The porter warns when the first bag's embedded contract differs from
`--contract` after parsing, so comments and whitespace don't count.

Loading a contract imports every `decoder:` and `encoder:` module it names,
so only load contracts you trust. A policy runner that resolves its contract
through a checkpoint warns about each such path before loading.

## Example contracts

The repository ships four under `contracts/`:

| File | Shows |
|---|---|
| `so_101.yaml` | Three compressed cameras, JointState state and action, `rad2deg`. |
| `so_101_hil.yaml` | The same plus `x-qos` anchors, teleop input, Joy events, a reward. |
| `stone.yaml` | Every section, annotated. Multi-source state, `asof`, custom codecs, split action, `clamp` with `zeros`. |
| `turtlebot3.yaml` | Two cameras, a 16-wide state from wheels, IMU and odometry, a TwistStamped action with `zeros`. |
