# rosetta_interfaces

ROS 2 action and service definitions for [Rosetta](https://github.com/iblnkn/rosetta).

These form the control plane (start/stop/manage). The data plane (observations,
actions, rewards) is plain ROS 2 topics defined by the contract YAML.

| Interface | Kind | Server | Purpose |
|-----------|------|--------|---------|
| `RecordEpisode` | action | `episode_recorder` | Record a demonstration to a rosbag |
| `RunPolicy` | action | `policy_runner` | Run policy (or classifier) inference |
| `StartRecording` | service | `episode_recorder` | Lightweight start of recording |
| `StartPolicy` | service | `policy_runner` | Lightweight start of policy inference |
| `RecordEpisodeStatus`, `RecordEpisodeResult` | message | `episode_recorder` | `~/status` and `~/result` topics |
| `RunPolicyStatus`, `RunPolicyResult` | message | `policy_runner` | `~/status` and `~/result` topics |
| `ManageEpisode` | action | `hil_manager` | Run one human-in-the-loop episode |
| `StartHILEpisode` | service | `hil_manager` | Start an episode without the action protocol (returns once started) |
| `ManageEpisodeStatus`, `ManageEpisodeResult` | message | `hil_manager` | `~/status` and `~/result` topics |

The services exist because some clients (Foxglove, for one) can call services but
not actions. They are thin proxies for the action operations they are named
after, and they return immediately, per the ROS 2 design guidance that services
"should never be used for longer running processes."

Such clients also can't see action feedback or results, so each server
publishes the same fields on a `~/status` topic while work runs and on a
transient-local `~/result` topic when it ends, whatever started the work.

Every action result reports **how** it ended, not whether it "succeeded":

- The goal's terminal `GoalStatus` is the mechanics: `SUCCEEDED` when the work
  reached a defined end, `CANCELED` when a client took it away, `ABORTED` when
  the server stopped it (an error, or a lifecycle deactivate).
- `termination_reason` names the exact cause, with the legal values declared as
  constants in each `.action`.

`ManageEpisode` also reports whether the robot did the task, in `outcome`
(`success`, `failure`, `unlabeled`), independently of `termination_reason`.
Its servers live in [`rosetta_ros_hil`](../rosetta_ros_hil); the rest are served
by [`rosetta_ros`](../rosetta_ros). One package holds every Rosetta ROS 2
interface, as `nav2_msgs` and `moveit_msgs` do for their projects.

Field-by-field definitions with example calls:
**[Actions and services reference](https://iblnkn.github.io/rosetta/reference/nodes.html)**.
The `.action` and `.srv` files in this package are the source of truth: they are
commented, and short enough to read directly.

## Build

```bash
colcon build --packages-select rosetta_interfaces
source install/setup.bash
```

## Documentation

Full Rosetta documentation: **https://iblnkn.github.io/rosetta/**

## License

Apache-2.0
