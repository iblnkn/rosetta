# Route commands through a mux

Put a node between the policy runner and the robot, such as a mux, a safety
filter or a recorder of the raw policy output. The runner publishes straight
to the contract's action topics, so move its output to other topics and let
the new node publish the real ones. Which tool you need depends on whether
both ends use the same message type.

## Move the runner's output

`output_prefix` moves every action and reward topic of the contract under a
prefix, so a contract with several action topics needs one argument:

```bash
ros2 launch rosetta_ros policy_runner_launch.py \
    contract_path:=robot.yaml output_prefix:=/policy
```

With the shipped `so_101.yaml`, the runner then publishes
`/policy/leader_arm/joint_states` instead of `/leader_arm/joint_states`. The
node you add subscribes to the prefixed topics and publishes the contract's
topics.

The launch file reads the topics to move from `contract_path`, so the prefix
can't follow a contract the runner loads from the checkpoint. When you start
the runner with `ros2 run` instead, remap each action topic yourself with a
standard ROS remap in its `--ros-args`, such as
`-r /leader_arm/joint_states:=/policy/leader_arm/joint_states`.

The runner publishes each action's `safety` command through the same
prefixed topics, so the safety command reaches the robot only if your node
forwards it.

## Connect topics of the same type

When the source and the destination use the same message type, connect them
without new code. Remap at launch, with `remappings=[("/teleop/cmd",
"/robot/cmd")]` on the node or `SetRemap` around an included launch file. To
connect two nodes you can't relaunch, run `relay` from the `topic_tools`
package:

```bash
ros2 run topic_tools relay /teleop/cmd /robot/cmd
```

To arbitrate between several velocity sources, use `twist_mux` and point one
of its inputs at the runner's prefixed topic.

## Translate between message types

When a device publishes a different type, field layout or unit than the robot
expects, a remap can't help. The contract's `teleop.input` section describes
that translation: each input is decoded with its own `select` and `apply` and
encoded as the action it targets. The rules are under
[Teleop](../reference/contract.md#teleop) in the contract reference.
`rosetta_ros_hil` applies the translation live while a person has control (see
[Run human-in-the-loop episodes](human-in-the-loop.md)).
