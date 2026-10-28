# Contract

A contract is one YAML file. This page lists every key it accepts and every
rule the loader enforces. Built-in message types, codecs and operators are on
[Message types and operators](message-types.md).

Loading a contract validates it. The loader parses the YAML, imports the robot
interface named by `robot_interface` and every codec module the file names,
checks every message type, timeline and QoS against that interface, builds
every operator, resolves every dtype and builds the frame layout. If it
returns, the contract is valid in this environment.

```bash
python3 -c "from rosetta.contract.schema import load_contract; load_contract('robot.yaml'); print('OK')"
```

For `robot_interface: ros2` the type check imports each message type, so a
contract loads only where ROS 2 and those message packages are importable.
Errors raise `rosetta.contract.errors.ContractValidationError`, a subclass of
`ValueError`. A missing file raises `FileNotFoundError`.

## Top level

| Key | Required | Rule |
|---|---|---|
| `robot_type` | yes | Non-empty string, no surrounding spaces. Written to the dataset as its robot type. |
| `robot_interface` | yes | The robot interface the contract is written for, by its registered name: `ros2` for `rosetta_ros`. A name that is not installed is an error that lists the installed ones. |
| `fps` | yes | Positive integer. Sets the tick rate. An integral float such as `30.0` is accepted, `29.97` is an error. |
| `observations` | no | Mapping of frame key to a source or a list of sources. |
| `actions` | no | Mapping of frame key to a source or a list of sources. |
| `rewards`, `signals`, `info`, `complementary_data` | no | Mapping of frame key to a source or a list of sources. Record-only. |
| `tasks` | no | Mapping of key to `{channel}`. |
| `adjunct` | no | List of `{channel}`. |
| `teleop` | no | Mapping with `input`, `events` and `feedback`. See [Teleop](#teleop). |
| `x-*` | no | Dropped before validation. Holds YAML anchors. |

Any other top-level key is an error. A section that is present but empty or
null is an error, so omit it instead. A duplicate key in any mapping of the
file is a YAML error. A frame key declared in two sections is an error.

## Sections

| Section | Image keys | `dtype` | `align` | `safety` | `encoder` | Reaches the policy |
|---|---|---|---|---|---|---|
| `observations` | yes | optional | required | no | no | yes, as input |
| `actions` | no | optional, numeric | required | yes | yes | yes, as output |
| `rewards`, `signals`, `info`, `complementary_data` | no | required, never `video` | required | no | no | no |
| `tasks` | no | no | no | no | no | no |
| `adjunct` | no | no | no | no | no | no |
| `teleop.input` | no | optional, numeric | required | no | no | no |
| `teleop.events` | no | no | no | no | no | no |
| `teleop.feedback` | no | optional, numeric | required | no | yes | no |

An image key starts with `observation.images.`. An image key in any section
other than `observations` is an error.

With the `policy_runner_node` parameter `is_classifier` set, the runner
publishes the policy output on the `rewards` channels instead of `actions`.
See [Nodes](nodes.md).

## Source

A source is a mapping with the keys `channel`, `align`, `select`, `apply` and
`kind`. `channel` and `align` are required. Any other key is an error.

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
| `type` | yes | Message type such as `sensor_msgs/msg/JointState`. Must import. |
| `qos` | no | See [`qos`](#qos). |
| `dtype` | per section | One of `float32`, `float64`, `int32`, `int64`, `bool`, `string`, `video`. |
| `decoder` | no | Custom decoder as `module.path:function`. Every section except `tasks`, `adjunct` and `teleop.events`. See [Custom codecs](message-types.md#custom-codecs). |
| `encoder` | no | Custom encoder as `module.path:function`. `actions` and `teleop.feedback` only. |
| `safety` | no | `actions` only. `none` (default), `zeros` or `hold`. |

A `tasks`, `adjunct` or `teleop.events` channel takes only `topic`, `type` and
`qos`. A key a section does not allow is an error.

A codec path is imported at load. A path without `:`, a module that does not
import, or a missing function is an error.

`dtype` resolves in this order: the explicit value, `video` for an image key,
`float64` for a channel with a custom `decoder`, otherwise the built-in
decoder's dtype. An explicit dtype other than `video` on an image key is an
error. `video` on any other key is an error.

An `observations`, record-only or `teleop.input` source needs a built-in or
custom decoder at load, even with an explicit `dtype`. An `actions` or
`teleop.feedback` source needs a built-in or custom encoder and a numeric
dtype at load, and a decoder when porting. A `teleop.input` source must
resolve to a numeric dtype.

`safety` is what the action watchdog publishes when no action frame has been
published for more than two frame periods, and what deactivating the node
publishes. The watchdog checks every two frame periods on the ROS clock. It
arms on the first published frame, fires once per stall, and does not run
when every action channel is `none`.

| `safety` | Publishes |
|---|---|
| `none` | Nothing. |
| `zeros` | The zero vector, sent through the inverse `apply` pipeline, so a `clamp` still bounds it. |
| `hold` | The last command sent. Zeros when nothing has been sent yet. |

Never put a position-controlled arm on `zeros`. Zero is a pose, and the arm
moves there. `hold` carries the same risk until the first command is sent: a
stop or deactivate before then has nothing to hold and sends zeros. The
LeRobot `Robot` plugin sends the safety action and clears the held command at
the end of every policy run, so this applies at the start of every episode.

### `qos`

| Key | Values | Default |
|---|---|---|
| `reliability` | `reliable`, `best_effort` | `reliable` |
| `durability` | `volatile`, `transient_local` | `volatile` |
| `history` | `keep_last`, `keep_all` | `keep_last` |
| `liveliness` | `automatic`, `manual_by_topic` | `system_default` |
| `depth` | integer | `10` |

Values are rclpy's short names, written exactly as shown, in lowercase. The
loader also accepts rclpy's `system_default` and `unknown` for all four policies, and
`best_available` for every policy except `history`. An unknown key or value
is an error. A channel without `qos` gets rclpy's default profile with depth
10. Duration-valued policies (deadline, lifespan, liveliness lease) are not
accepted.

A `best_effort` publisher read with the default `reliable` delivers nothing.

### `align`

Required on every source. There is no default.

| Key | Rule |
|---|---|
| `strategy` | `hold`, `asof` or `drop`. |
| `timeline` | A timeline the channel's type provides. Case-sensitive. |
| `tolerance_ms` | Positive integer, in milliseconds. Required with `asof`. An error with any other strategy. |

Timelines the `ros2` robot interface provides:

| Timeline | Provided by | Value live | Value when porting |
|---|---|---|---|
| `receive` | every type | node clock at arrival | the message's timestamp in the bag |
| `header` | a type with a field `header` of type `std_msgs/Header` | `header.stamp` | `header.stamp` |

A timeline the type does not provide is an error that lists the ones it does.
A `header` message whose stamp is `(0, 0)` is dropped, never re-stamped. The
drop is logged once per stream, and again after the stream recovers and drops
again.

Each stream keeps only its newest sample by timestamp. A message stamped older
than the buffered one is discarded. At each tick the strategy decides whether
that sample is used:

| Strategy | Sample used at a tick |
|---|---|
| `hold` | The newest, at any age. |
| `asof` | The newest, if its age is at most `tolerance_ms`. Otherwise none. |
| `drop` | The newest, if it is less than one tick period old. Otherwise none. |

A stamp ahead of the tick by up to `max(1 s, 2 tick periods)` counts as age
zero. A stamp further ahead clears the buffer, as a clock reset.

Warmup lasts until every observation source has a sample. Actions and
record-only sources do not gate it. When porting, ticks during warmup are
skipped. Live, the LeRobot `Robot` plugin waits up to 5 s for warmup, then
carries on with a warning.

After warmup, a source with no sample at a tick is zero-filled: numeric zeros
at its width, a black image at its size, or an empty string.

### `select`

A list of unique strings. The order is the order in the frame. An empty list
is an error. Without `select`, the source is one value wide.

What a `select` entry means depends on the codec. For dotted-path types it is
a field path. For the parallel-array types it is `<field>.<name>`. For
`std_msgs` scalars and multi-arrays it only names the values. The syntax per
type is on [Message types and operators](message-types.md#decoders).

`select` is required for these types, and for every source of a multi-source
key:

- `sensor_msgs/msg/JointState`
- `sensor_msgs/msg/Imu`
- `sensor_msgs/msg/Joy`
- `nav_msgs/msg/Odometry`
- `geometry_msgs/msg/Twist`
- `geometry_msgs/msg/TwistStamped`
- `control_msgs/msg/MultiDOFCommand`
- `trajectory_msgs/msg/JointTrajectory`

A channel with a custom `decoder` is exempt. An action of one of these types
is exempt only with both a custom `decoder` and a custom `encoder`, because
actions are decoded when porting.

The width of a source is `len(select)`, or 1 without `select`. A decoded or
encoded value of any other width fails for that message.

### `apply`

An ordered list of operators. Each entry is a bare name or a one-key mapping
of name to arguments.

```yaml
apply: [clamp: {min: -3.14159, max: 3.14159}, rad2deg]
```

Decoding runs the list front to back through each operator's forward
direction. Encoding runs it back to front through each inverse. In the
example, encoding converts degrees to radians and then clamps radians.

- An `actions` or `teleop.feedback` source accepts only operators with an
  inverse. `resize` there is an error.
- `apply` on a source that resolves to `string` is an error.
- Every image key needs an operator that sets a fixed output size. Among the
  built-ins that is `resize`. The last one in the list sets the stored size.
- An unknown operator name is an error that lists the registered ones.

Operators and their arguments are on
[Message types and operators](message-types.md#operators).

### `kind`

Optional. Names the value's representation and checks the `select` width.

| `kind` | Width |
|---|---|
| `continuous` (default) | any |
| `quaternion` | 4 |
| `euler_rpy` | 3 |
| `axis_angle` | 3 |
| `rotation_6d` | 6 |
| `binary` | any |

A width that disagrees with `select` is an error. A `continuous` source logs
a warning at load when four consecutive `select` entries end in `x`, `y`, `z`
and `w` in any order, or when an entry contains `quat`. `kind` is for framework adapters
that normalize or rotate by representation. The LeRobot adapter ignores it.

## Multi-source keys

A list under one key declares ordered sources.

```yaml
observation.state:
  - channel: {topic: /arm/joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: header}
    select: [position.j1, position.j2]
  - channel: {topic: /gripper/state, type: std_msgs/msg/Float32, dtype: float64}
    align: {strategy: hold, timeline: receive}
    select: [data]
```

- Observations concatenate in list order. Actions split in list order.
- Every source needs `select`.
- Every source resolves to the same dtype. Set `dtype` explicitly where the
  decoders differ, as on the `Float32` source in the example.
- An image key takes one source.
- Feature names get a per-topic prefix: the first topic segment position at
  which every topic differs, so `/arm/joint_states` and `/gripper/state` give
  `arm.position.j1` and `gripper.data`. Sources that share one topic get no
  prefix. Topics that normalize to the same path are an error, and so are two
  sources that render the same feature name.
- A single-source key has no prefix.
- An empty list is an error.

The LeRobot live path accepts one numeric observation key, one action key, and
no numeric source without `select`. The policy runner checks this at
configure and the LeRobot `Robot` plugin at connect, not at load. See
[About the contract](../explanation/design.md#one-numeric-key-on-the-live-path).

## Tasks

```yaml
tasks:
  task:
    channel: {topic: /task_prompt, type: std_msgs/msg/String}
```

A `tasks` channel is recorded and has no `align`. When porting, the frame's
task at each tick is the newest message received at or before the tick, from
any task channel. Before the first message, or with no `tasks` section, the
task is the prompt the episode was recorded with. A task message without a
string `data` field is ignored with one warning per bag. Live, the task is the
prompt of the `RunPolicy` goal or `start_policy` call, or the node's
`default_prompt`.

## Rewards, signals, info, complementary_data

```yaml
rewards:
  next.reward:
    channel: {topic: /reward, type: std_msgs/msg/Float64, dtype: float64}
    align: {strategy: hold, timeline: receive}

signals:
  next.done:
    channel: {topic: /episode_done, type: std_msgs/msg/Bool, dtype: bool}
    align: {strategy: hold, timeline: receive}
```

These four sections are frame-clock sources with `dtype` required and never
`video`. They are recorded into the dataset and never fed to a policy. Their
keys do not gate warmup.

## Adjunct

```yaml
adjunct:
  - channel: {topic: /tf, type: tf2_msgs/msg/TFMessage}
  - channel: {topic: /tf_static, type: tf2_msgs/msg/TFMessage,
              qos: {durability: transient_local}}
```

Adjunct channels are recorded to the bag and never decoded, so any type that
imports works. The recorder subscribes to them as contract topics, and its
end-of-episode summary marks any contract topic with zero messages.

## Teleop

```yaml
teleop:
  input:
    - target: /forward_position_controller/commands   # an action source's topic
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
    - origin: /joint_states                           # an observation source's topic
      channel: {topic: /leader_arm/joint_feedback, type: sensor_msgs/msg/JointState}
      align: {strategy: hold, timeline: receive}
      select: [position.shoulder_pan_joint, position.elbow_flex_joint]
      apply: [rad2deg]
```

| Key | Rule |
|---|---|
| `input` | List of sources, each with a `target`. |
| `input[].target` | The topic of exactly one `actions` entry. Each target appears once. |
| `events` | Mapping with `channel` and `select`, both required. No `align`. |
| `events.select` | Non-empty mapping of event name to a path such as `buttons.5` or `axes.2`. A numeric path segment indexes a sequence. A negative index fails. |
| `feedback` | List of sources, each with an `origin`. |
| `feedback[].origin` | The topic of exactly one `observations` entry. Each origin appears once. |

A teleop consumer decodes each `input` source with its own `select` and
`apply`, then encodes the value onto the target with the action source's
codec and inverse `apply`. The input must match the target source in width
and units. For `feedback`, the consumer decodes the origin with that
observation source's `select` and `apply`, then encodes it with the feedback
source's codec and inverse `apply`. The feedback `select` must match the
origin source in width. The loader checks neither width. A mismatch fails
every message at runtime with a logged error.

Event names are open. The loader accepts any name, and each consumer acts on
the names it knows and logs the ones it ignores, so one contract can serve
several consumers. The human-in-the-loop manager's names are in the
[`rosetta_ros_hil` README](https://github.com/iblnkn/rosetta/tree/main/robots/ros/rosetta_ros_hil),
and the LeRobot teleoperator maps `is_intervention`, `success`, `failure`,
`end_success` and `end_failure` onto LeRobot's `TeleopEvents`. Events are read
per message, never resampled. A bad event path is not a load error. The HIL
manager logs it once. The LeRobot teleoperator skips that event.

The porter writes each `input` source as `teleop.input.<action key>` and each
`feedback` source as `teleop.feedback.<observation key>`, with the feature
names of the source they target. A declared key with one of those names is an
error.

## Checked after load

These pass `load_contract` and fail later:

| Condition | Fails at |
|---|---|
| A teleop `input` or `feedback` source whose width differs from its target or origin. | Every forwarded message. |
| A `select` wider than 1 on a `std_msgs/msg/Float32` or `Float64` action. | Every encode. |
| A `JointState`, `JointTrajectory` or `MultiDOFCommand` action whose selected fields do not each cover every name. | Every encode. |
| One topic declared with two different types. | Recorder configure, and every episode when porting. |
| One topic declared with two different `qos`. | Recorder configure. |
| An action type with a custom encoder but no decoder. | Every episode when porting. |
| Two numeric observation keys, two action keys, or a numeric source without `select`, on the LeRobot live path. | Policy runner configure, or LeRobot `Robot` plugin connect. |
| `is_classifier` with no `rewards` section. | Policy runner configure. |

## Embedding

The recorder writes the contract text into the bag's `metadata.yaml` under
`rosbag2_bagfile_information.custom_data`, key `rosetta.contract_yaml`, unless
its `embed_contract` parameter is false. A non-empty prompt goes under
`lerobot.operator_prompt`, and the goal ID of an action-started recording
under `rosetta.goal_id`.

The porter decodes with `--contract` only. It warns when the first bag's
embedded contract differs from `--contract` after parsing, so comments and
whitespace do not count. It copies `--contract` to
`meta/rosetta_contract.yaml` in a LeRobot dataset unless
`--no-embed-contract` is given. `policy_runner_node` with an empty
`contract_path` follows the checkpoint's `train_config.json` to its dataset
and loads that copy.

Loading a contract imports every `decoder:` and `encoder:` module it names,
so load only trusted contracts. A policy runner that resolves its contract
through a checkpoint warns about each such path before loading it.

## Example contracts

`rosetta_ros` installs three under `share/rosetta_ros/contracts/`:

| File | Shows |
|---|---|
| `so_101.yaml` | Three `CompressedImage` cameras resized to 512x512, six-joint `JointState` state and action, `rad2deg`. |
| `stone.yaml` | Every section, annotated. Multi-source state and action, `asof`, a custom decoder and encoder, `clamp` with `zeros`, teleop. It has three numeric observation keys and two action keys, so the LeRobot live path refuses it. |
| `turtlebot3.yaml` | Two cameras, a 16-wide state from wheel joints, IMU and odometry, a `TwistStamped` action with `zeros`. It logs the quaternion warning at load. |
