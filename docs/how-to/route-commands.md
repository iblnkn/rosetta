# Route commands through a mux

The policy runner publishes straight to the contract's action topics. To put
something between the policy and the robot, such as a mux, a safety filter or
a recorder of the raw policy output, move the runner's output to other topics
and let the extra node publish the real ones. Which tool you need depends on
whether the two ends use the same message type.

## Move the runner's output

`output_prefix` moves every action and reward topic of the contract under a
prefix, so a contract with several action topics needs one argument:

```bash
ros2 launch rosetta_ros policy_runner_launch.py \
    contract_path:=robot.yaml output_prefix:=/policy
```

With `stone.yaml`, the runner then publishes `/policy/arm/joint_commands`,
`/policy/gripper/command` and `/policy/cmd_vel`. The node you add subscribes
there and publishes the contract's topics.

## Connect topics of the same type

When the source and the destination use the same message type, connect them
without new code. Remap at launch, with `remappings=[("/teleop/cmd",
"/robot/cmd")]` on the node or `SetRemap` around an included launch file. To
connect two nodes you can't relaunch, run `topic_tools relay`:

```bash
ros2 run topic_tools relay /teleop/cmd /robot/cmd
```

A velocity mux between several sources is `twist_mux`'s job; point its inputs
at the runner's prefixed topics.

## Translate between message types

When a device publishes a different type, field layout or unit than the robot
expects, a remap can't help. The contract's `teleop.input` section describes
that translation: each input is decoded with its own `select` and `apply` and
encoded as the action it targets. `rosetta_ros_hil` does this while a person
has control, and the [Contract](../reference/contract.md#teleop) page lists the
rules.
