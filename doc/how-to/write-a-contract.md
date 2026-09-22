# Write a contract

The [Contract](../reference/contract.md) page has every key and rule. This
page walks through writing one. If you'd rather start from an annotated file,
copy `contracts/stone.yaml` from the repository.

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

`fps` is the rate the policy runs at. Your controller's update rate is a
sensible pick.

## Point each key at a channel

Find the topic and its type:

```bash
ros2 topic list -t
ros2 topic info -v /joint_states
```

Copy the type and the publisher's QoS into `channel`. A QoS mismatch means no
messages arrive. Then check whether the type has a header and whether the
driver fills it:

```bash
ros2 topic echo /joint_states --field header.stamp --once
```

A `sec` of `0` means unstamped.

## When: `align`

Every source needs `align`. There's no default.

Use `timeline: header` when the type has a header and the driver stamps it.
Header-aligned streams replay from a bag exactly. Otherwise use
`timeline: receive`. A type with no header, like
`std_msgs/msg/Float64MultiArray`, only has `receive`.

Use `strategy: hold` unless a stale value is worse than none. `asof` with
`tolerance_ms` rejects samples older than the tolerance. `drop` keeps only
samples from the last frame period. When a sample is rejected, the key is
zero-filled for that tick.

```yaml
align: {strategy: hold, timeline: header}
```

## Which: `select`

List the fields in the order you want them in the vector. Syntax per type is
on [Message types](../reference/message-types.md#decoders).

```yaml
select: [position.shoulder_pan_joint, position.elbow_flex_joint, position.gripper_joint]
```

The decoder looks joints up by name, so these have to match what the driver
publishes. Check with `ros2 topic echo /joint_states --field name --once`.

## How: `apply`

Add operators where the robot's units differ from what the policy should
learn.

```yaml
apply: [rad2deg]
```

On an action the list runs in reverse when serving. Put `clamp` first to
bound the outgoing command in robot units:

```yaml
apply: [clamp: {min: -3.14159, max: 3.14159}, rad2deg]
```

Every image observation needs a `resize`, which fixes the stored image size.

## Merge streams into one key

LeRobot's live path takes one numeric observation key and one action key. To
feed several topics into `observation.state`, list them as sources. Values
are concatenated in order.

```yaml
observation.state:
  - channel: {topic: /arm/joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: header}
    select: [position.j1, position.j2]
  - channel: {topic: /gripper/state, type: std_msgs/msg/Float32}
    align: {strategy: hold, timeline: receive}
    select: [data]
```

## Set the action's safety

`safety` is what gets published when actions stop arriving. Use `hold` for a
position-controlled arm, `zeros` for a velocity command such as a Twist, and
`none` when nothing should be sent.

Don't put a position-controlled arm on `zeros`. Zero is a pose, and the arm
will go there.

```yaml
channel: {topic: /cmd, type: std_msgs/msg/Float64MultiArray, safety: hold}
```

## Validate

```bash
python -c "from rosetta.contract.schema import load_contract; load_contract('robot.yaml'); print('OK')"
```

Loading checks every operator and codec path, and, with `rclpy` importable,
every type, timeline and QoS key against the installed ROS 2 interfaces. Fix
what it reports until it prints `OK`.
