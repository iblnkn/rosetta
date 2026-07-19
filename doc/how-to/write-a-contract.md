# Write a Contract

This guide shows you how to write a contract for your robot: the mapping from topics to training frames. Full field documentation lives in the [contract reference](../reference/contract.md).

## Start minimal

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

Every frame-clock entry reads as one pipeline: `channel` provides, `align` chooses a timeline (mandatory), then `select`, then `apply`, then the mapping key. Actions read the same pipeline right-to-left. Full semantics: [contract reference](../reference/contract.md). Design rationale: [contract design decisions](../explanation/contract-design.md).

## Pick timelines and strategies

Use `timeline: header` for stamped sensors with synced clocks, `timeline: receive` otherwise. Use `strategy: hold` unless you need stale-data rejection (`asof` with `tolerance_ms`) or strict freshness (`drop`). See [alignment strategies](../reference/contract.md#alignment-strategies).

## Multi-source keys

A **list** value under one key declares ordered sources whose values concatenate into a single feature vector:

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

This matters because core policies expect specific key names (see [policy compatibility](../reference/lerobot-data-model.md#policy-feature-compatibility)). To feed several ROS2 topics into one observation, declare them as ordered sources of one key.

The same topic also works in several sources (e.g. position and orientation slices of one pose topic). Each source keeps its own selector and buffer.

Rules, validated at load (`ContractValidationError` otherwise):

- Every source of a multi-source key needs a `select`. Concatenation needs static dims to lay out the combined vector.
- All sources of a key must resolve to the **same** dtype. Set `dtype:` in the channel explicitly to align them.
- Images and strings never share a key. Give each an own key. Image features never concatenate.

Layout follows declaration order. Reorder sources by reordering the list. Rationale: [why layout follows declaration order](../explanation/contract-design.md#why-layout-follows-declaration-order).

## Set action safety

Declare a stop behavior per action channel. `none` (default) publishes nothing on watchdog or deactivate. `zeros` suits velocity control. `hold` re-sends the last command. Never leave a position-controlled arm on `zeros`. See [actions](../reference/contract.md#actions).

## Maximize policy compatibility

For a dataset working with the widest range of policies:

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

For VLA fine-tuning, add a second camera and keep recording prompts descriptive:

```yaml
observations:
  # ... state and first camera as above ...

  observation.images.wrist.right:
    channel: {topic: /wrist_camera/image_raw/compressed,
              type: sensor_msgs/msg/CompressedImage}
    align: {strategy: hold, timeline: header}
    apply: [resize: [512, 512]]
```

## Worked example

`contracts/stone.yaml` is the annotated tour: multi-source actions, teleop, QoS anchors, extended sections.
