# Write a contract

The contract defines the translation between ROS 2 topics and the keys LeRobot
expects. Full field documentation: [contract reference](../reference/contract.md).

| ROS 2 side | | LeRobot side |
|-----------|---|-------------|
| `/front_camera/image_raw/compressed` | &rarr; | `observation.images.front` |
| `/follower_arm/joint_states` (position fields) | &rarr; | `observation.state` |
| `/leader_arm/joint_states` (position fields) | &larr; | `action` |
| `/task_prompt` (String) | &rarr; | `task` |
| `/reward_signal` (Float64) | &rarr; | `next.reward` |

On the ROS 2 side, data lives in typed messages on named topics with rich
structure. On the LeRobot side, data lives in flat dictionaries with
dot-separated string keys and numpy values. The contract maps one to the other,
handling type conversion, field extraction, timestamp alignment, and resampling.

```yaml
observation.state:
  channel: {topic: /follower_arm/joint_states, type: sensor_msgs/msg/JointState}
  align: {strategy: hold, timeline: header}
  select: [position.shoulder_pan, position.shoulder_lift, position.elbow]
```

At each timestep, this **subscribes** to `/follower_arm/joint_states`,
**extracts** the named fields using dot notation
(`position.shoulder_pan` → `msg.position[msg.name.index("shoulder_pan")]`),
**assembles** a numpy array, and **stores** it under `observation.state`.

Use `timeline: header` for stamped sensors with synced clocks, `receive`
otherwise. Use `strategy: hold` unless a stale value is worse than a fabricated
zero, since a gap zero-fills rather than skipping the frame.

## Multi-source keys

A **list** value under one key declares ordered sources whose values are
concatenated in declaration order:

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

This matters because built-in policies look for specific key names by exact
match. Feeding several topics into one key is how a rich robot stays compatible
with them. Every source of a multi-source key needs a `select` and they must all
resolve to the same dtype. Images never share a key.

The live LeRobot path enforces the same shape harder: **at most one numeric
observation key and at most one action key**, or the contract is refused at
deploy time, after the dataset has already trained fine. See
[key-count limit](../reference/contract.md#key-count-limit-on-the-lerobot-live-path).

## Action safety

`channel.safety` declares what the watchdog publishes when actions stop
arriving: `none` (default), `zeros`, or `hold`. **Never put a
position-controlled arm on `zeros`:** zero is a pose, and the arm will slam to
it. `hold` lands in the same place before the channel has published once, since
there is no last command to re-send.

## Worked example

`contracts/stone.yaml` is the annotated tour: multi-source actions, teleop, QoS
anchors, extended sections. It declares three numeric observation keys, so read
it for the schema and do not copy its shape. `so_101.yaml` and `turtlebot3.yaml`
are the deployable examples.
