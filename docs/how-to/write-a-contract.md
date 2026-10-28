# Write a contract

Write a contract that maps your robot's ROS 2 topics to the keys a policy
reads and writes. Every key and rule is on the [Contract](../reference/contract.md)
page. This page walks through the decisions in the order you make them.

You need the robot (or simulator) running so you can inspect its topics, and
a shell with ROS 2 and the Rosetta workspace sourced.

To start from a working file, copy one from `rosetta_ros`'s `contracts/`
directory (`robots/ros/rosetta_ros/contracts/` in the repository, or
`$(ros2 pkg prefix --share rosetta_ros)/contracts/` once installed).
`so_101.yaml` is a complete arm contract. `stone.yaml` is the annotated tour of
every feature, but it declares several numeric observation keys and two action
keys, so the LeRobot runner refuses it.

## Name the keys

Decide what the model sees and what it emits. LeRobot's built-in policies
expect `observation.images.<name>` per camera, `observation.state` for the
numeric state, and `action` for the command.

```yaml
robot_type: my_robot
robot_interface: ros2
fps: 30

observations:
  observation.images.wrist: ...
  observation.state: ...
actions:
  action: ...
```

`fps` is the rate frames are built at, when porting and at inference. It
has to be a positive integer. Your controller's update rate is a sensible
pick.

The LeRobot runner and the LeRobot `Robot` plugin accept one numeric observation
key and one action key, and every numeric source needs a `select`. They check
this when they start and refuse the contract otherwise. Image keys under
`observation.images.` are allowed only in `observations`.

## Point each key at a channel

A channel is the topic, its message type, and the QoS Rosetta subscribes or
publishes with. Find the topic and its type:

```bash
ros2 topic list -t
ros2 topic info -v /joint_states
```

Copy the type and the publisher's QoS into `channel`. Without `qos`, Rosetta
uses a reliable profile with depth 10, which receives nothing from a
`best_effort` publisher. Allowed `qos` keys are `reliability`, `durability`,
`history`, `liveliness` and `depth`.

```yaml
channel:
  topic: /wrist_camera/image_raw/compressed
  type: sensor_msgs/msg/CompressedImage
  qos: {reliability: best_effort, depth: 10}
```

Then check whether the type has a header and whether the driver fills it:

```bash
ros2 topic echo /joint_states --field header.stamp --once
```

A stamp of `sec: 0` and `nanosec: 0` means unstamped.

## When: `align`

Every source needs `align`, with a `timeline` and a `strategy`. There's no
default for either.

The `timeline` names which timestamp a message is placed at. The `ros2` robot
interface provides `receive` for every type, and `header` for types with a
`std_msgs/Header` field. Use `timeline: header` when the driver stamps its
messages. The stamp is part of the message, so porting a bag and running live
place a message at the same time. With `header`, a message stamped `0` is
dropped. Otherwise use `timeline: receive`, which is the bag's record time
when porting and the arrival time live, so it matches closely, not exactly. A
type with no header, like `std_msgs/msg/Float64MultiArray`, only provides
`receive`.

The `strategy` decides which sample each frame tick takes:

| Strategy | Takes |
|---|---|
| `hold` | The newest sample, however old. |
| `asof` | The newest sample if it is at most `tolerance_ms` older than the tick. `tolerance_ms` is required and must be a positive integer. |
| `drop` | The newest sample only if it is less than one frame period older than the tick. |

Use `hold` unless a stale value is worse than none. When a tick gets no
sample, the key is zero-filled for that tick.

```yaml
align: {strategy: hold, timeline: header}
```

## Which: `select`

List the fields in the order you want them in the vector. The syntax per type
is on [Message types](../reference/message-types.md#decoders).

```yaml
select: [position.shoulder_pan_joint, position.elbow_flex_joint, position.gripper_joint]
```

The `JointState` decoder looks joints up by name, so these have to match what
the driver publishes. Check with
`ros2 topic echo /joint_states --field name --once`. If a name doesn't match,
every message on that stream is dropped with a warning that lists the names
the message has, and the key is zero-filled.

For a scalar or array type such as `std_msgs/msg/Float64MultiArray`, `select`
gives one name per element, in order. The names label the dataset columns.

## How: `apply`

Add operators where the robot's units differ from what the policy should
learn.

```yaml
apply: [rad2deg]
```

On an action the list runs in reverse when the runner publishes. Put `clamp`
first to bound the outgoing command in robot units:

```yaml
apply: [clamp: {min: -3.14159, max: 3.14159}, rad2deg]
```

Every image observation needs `resize: [h, w]`, which fixes the stored image
size. `resize` has no inverse, so it can't go on an action.

## Merge streams into one key

To feed several topics into `observation.state`, list them as sources. Values
are concatenated in order. Every source under one key has to resolve to the
same `dtype`. A `std_msgs/msg/Float32` decodes as `float32` and a
`JointState` as `float64`, so set `dtype` on the channel that differs.

```yaml
observation.state:
  - channel: {topic: /arm/joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: header}
    select: [position.j1, position.j2]
  - channel: {topic: /gripper/state, type: std_msgs/msg/Float32, dtype: float64}
    align: {strategy: hold, timeline: receive}
    select: [gripper]
```

Feature names get a prefix from each topic so they stay unique. This key
becomes `arm.position.j1`, `arm.position.j2` and `gripper.gripper`.

## Set the action's safety

`safety` on an action channel is what the runner publishes when it stops
commanding the robot: when no action has gone out for two frame periods, when
a LeRobot run ends, and when the node deactivates.

| Value | Publishes | Use for |
|---|---|---|
| `none` (default) | Nothing. | Channels where nothing should be sent. |
| `zeros` | Zeros, run back through `apply`. | Velocity commands such as a `Twist`. |
| `hold` | The last command sent. | Position-controlled arms. |

Don't put a position-controlled arm on `zeros`. Zero is a pose, and the arm
will go there. `hold` sends zeros too if the runner hasn't published a
command yet in that run.

```yaml
channel: {topic: /arm_controller/commands, type: std_msgs/msg/Float64MultiArray, safety: hold}
```

## Validate

Put the pieces in one file, for example `robot.yaml`:

```yaml
robot_type: my_robot
robot_interface: ros2
fps: 30

observations:
  observation.images.wrist:
    channel:
      topic: /wrist_camera/image_raw/compressed
      type: sensor_msgs/msg/CompressedImage
      qos: {reliability: best_effort, depth: 10}
    align: {strategy: hold, timeline: header}
    apply: [resize: [224, 224]]

  observation.state:
    channel: {topic: /joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: header}
    select: [position.shoulder_pan_joint, position.elbow_flex_joint, position.gripper_joint]
    apply: [rad2deg]

actions:
  action:
    channel:
      topic: /arm_controller/commands
      type: std_msgs/msg/Float64MultiArray
      safety: hold
    align: {strategy: hold, timeline: receive}
    select: [shoulder_pan_joint, elbow_flex_joint, gripper_joint]
    apply: [clamp: {min: -3.14159, max: 3.14159}, rad2deg]
```

Load it in the same sourced shell:

```bash
python3 -c "from rosetta.contract.schema import load_contract; load_contract('robot.yaml'); print('OK')"
```

Loading checks every key, operator and codec, and asks the `ros2` robot
interface whether each message type exists, provides the chosen timeline, and
accepts the `qos`. It raises `ContractValidationError` naming the entry at
fault, for example:

```text
rosetta.contract.errors.ContractValidationError: actions.action.align: timeline 'header' is not provided by 'std_msgs/msg/Float64MultiArray'. This channel provides: ['receive']
```

Fix what it reports until it prints `OK`. Loading doesn't apply the LeRobot
limits from [Name the keys](#name-the-keys). Those fail when the runner
configures.
