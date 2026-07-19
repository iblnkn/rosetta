# Set Up Teleop and Human-in-the-Loop

This guide shows you how to drive your robot with a leader device during recording, and how to mux human intervention into policy episodes. Contract schema: [teleop reference](../reference/contract.md#teleop).

## Declare teleop in the contract

```yaml
teleop:
  input:
    - target: /arm/joint_commands   # names an existing action channel's topic
      channel: {topic: /leader_arm/joint_states, type: sensor_msgs/msg/JointState}
      align: {strategy: hold, timeline: header}
      select: [position.j1, position.j2]

  events:
    channel: {topic: /joy, type: sensor_msgs/msg/Joy}
    select:
      is_intervention: buttons.5
      success: buttons.0

  feedback:
    - origin: /arm/joint_states
      channel: {topic: /leader_arm/effort_feedback, type: sensor_msgs/msg/JointState}
      align: {strategy: hold, timeline: receive}
      select: [effort.j1, effort.j2]
```

`input` entries name the action they drive (`target`). `feedback` entries name the observation they mirror (`origin`). A contract teleops one action, several, or none.

## What happens at runtime

`hil_manager_node` drives each `input` entry live: the node decodes the teleop message and encodes+publishes the value onto `target` with the same decode/encode machinery as everything else in the contract, not a raw byte-for-byte republish. A leader device with different fields, units, or timeline than the action still works. The action's own topic already carries the human's command during teleop. The porter additionally exposes `input`/`feedback` as their own `teleop.input.<action key>` / `teleop.feedback.<observation key>` dataset columns, for diagnostics. Feedback runs the mirror direction (observation → encode → publish to the human device) regardless of mux state.

## Launch the HIL stack

`hil_launch.py` wires four nodes: `hil_manager_node`, a policy runner (namespace `robot_policy`), an optional reward classifier (a second policy runner with `is_classifier: true`, namespace `reward_classifier`), and the episode recorder. Configuration lives in `params/hil_manager.yaml`, the "super YAML" covering all four nodes.

```bash
ros2 launch rosetta hil_launch.py contract_path:=/path/to/contract.yaml
```

The manager runs an episode end to end: start recording, run the policy, mux teleop intervention onto the action topics, apply reward overrides. Control:

```bash
# Run a managed episode
ros2 action send_goal /manage_episode \
    rosetta_interfaces/action/ManageEpisode "{prompt: 'task description'}"

# Take over manually mid-episode
ros2 service call /hil_manager/set_intervention std_srvs/srv/SetBool "{data: true}"

# Force the reward signal
ros2 service call /hil_manager/set_reward_override std_srvs/srv/SetBool "{data: true}"
```

Full service table: [nodes](../reference/nodes.md#hil_manager_node). Event vocabulary and load-time rules: [teleop reference](../reference/contract.md#teleop). See `contracts/stone.yaml` for a full worked example.
