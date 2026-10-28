# Interfaces

The `rosetta_interfaces` package holds every Rosetta ROS 2 interface: three
actions, three services and six messages. The nodes also use
`std_srvs/srv/Trigger`, `std_srvs/srv/SetBool` and `std_msgs/msg/Int8`. Which
node serves what is on [Nodes](nodes.md). The `ManageEpisode` family belongs to
the human-in-the-loop manager in
[`rosetta_ros_hil`](https://github.com/iblnkn/rosetta/tree/main/robots/ros/rosetta_ros_hil),
whose README describes its behavior.

```bash
ros2 interface show rosetta_interfaces/action/RecordEpisode
```

## Ending work

Every action result reports how the work ended in `termination_reason`. The
legal values are string constants in each `.action` file. The goal's terminal
state is the coarse view: `SUCCEEDED` when the work reached a defined end,
`CANCELED` when a client cancelled it, `ABORTED` when the server stopped it.
`ManageEpisode` also reports whether the robot did the task, in `outcome`,
independently of `termination_reason`.

A start service mirrors its action's goal and returns as soon as the work is
accepted. Work started this way has no goal, so it has no action feedback or
result. Each server therefore publishes the feedback fields on `~/status`
while work runs and the result fields on `~/result` when it ends, whatever
started the work. `~/result` is transient-local, so a subscriber that joins
later still gets the last one. Stop service-started work with the matching
cancel service, or let `default_max_duration_s` end it.

## Actions

Each action has a goal, a result and feedback. Defaults named in the goal
tables are parameters of the serving node.

### RecordEpisode

Server: `episode_recorder_node`, as `/record_episode` by default.

| Goal | Type | Meaning |
|---|---|---|
| `prompt` | string | Task label stored in the bag. Empty uses the node's `default_prompt`. |
| `max_duration_s` | float64 | Seconds. Zero or less uses the node's `default_max_duration_s`. |

| Result | Type | Meaning |
|---|---|---|
| `termination_reason` | string | `stopped`, `timeout`, `cancelled`, `node_deactivated`, `error` |
| `message` | string | Detail when something went wrong. Empty otherwise. |
| `bag_path` | string | Bag directory. Set on every path, including errors. |
| `messages_written` | int32 | Messages written to the bag. |

| Feedback | Type | Meaning |
|---|---|---|
| `elapsed_s` | float64 | Seconds since the recording started. |
| `messages_written` | int32 | Messages written so far. |

### RunPolicy

Server: `policy_runner_node`, as `/run_policy` by default.

| Goal | Type | Meaning |
|---|---|---|
| `prompt` | string | Task passed to the policy. Empty uses the node's `default_prompt`. |
| `max_duration_s` | float64 | Seconds. Zero or less uses the node's `default_max_duration_s`. |

| Result | Type | Meaning |
|---|---|---|
| `termination_reason` | string | `completed`, `timeout`, `cancelled`, `node_deactivated`, `error` |
| `message` | string | Detail when something went wrong. Empty otherwise. |

| Feedback | Type | Meaning |
|---|---|---|
| `published_actions` | uint32 | Actions published so far in this run. |
| `queue_depth` | uint32 | Actions waiting in the adapter's queue. |

### ManageEpisode

Server: `hil_manager_node`, as `/manage_episode` by default.

| Goal | Type | Meaning |
|---|---|---|
| `prompt` | string | Task for the episode. Empty uses the manager's `default_prompt`. |
| `max_duration_s` | float64 | Seconds. Zero or less uses the manager's `default_max_duration_s`. |
| `success_reward_threshold` | float64 | End the episode when the reward reaches this value. Zero or less disables it. |

| Result | Type | Meaning |
|---|---|---|
| `termination_reason` | string | `stopped`, `timeout`, `reward_threshold`, `cancelled`, `node_deactivated`, `error` |
| `outcome` | string | `success`, `failure`, `unlabeled` |
| `message` | string | Detail when something went wrong. Empty otherwise. |
| `bag_path` | string | Bag directory of the episode's recording. Empty when the manager does not record. |
| `final_reward` | float64 | Last reward value at episode end. |
| `messages_written` | int32 | Messages written to the bag. |

| Feedback | Type | Meaning |
|---|---|---|
| `elapsed_s` | float64 | Seconds since the episode started. |
| `current_reward` | float64 | Latest reward value. |
| `control_source` | string | `policy` or `teleop` |
| `outcome` | string | Verdict so far, one of the result's `outcome` values. |
| `messages_written` | int32 | Messages written so far. |

`unlabeled` means nobody claimed success or failure. It is a different answer
from a reward of `0.0`.

## Services

| Service | Request | Response |
|---|---|---|
| `StartRecording` | `prompt` string | `accepted` bool, `message` string |
| `StartPolicy` | `prompt` string | `accepted` bool, `message` string |
| `StartHILEpisode` | `prompt` string, `max_duration_s` float64, `success_reward_threshold` float64 | `accepted` bool, `message` string |

The request fields mean the same as the matching goal fields.
`StartRecording` and `StartPolicy` have no duration field, so work they start
always uses the node's `default_max_duration_s`. `accepted: false` comes with
the reason in `message`, such as `Rejected: already busy`.

## Messages

| Message | Fields | Published on |
|---|---|---|
| `RecordEpisodeStatus` | `RecordEpisode` feedback | recorder `~/status` |
| `RecordEpisodeResult` | `RecordEpisode` result | recorder `~/result` |
| `RunPolicyStatus` | `RunPolicy` feedback | runner `~/status` |
| `RunPolicyResult` | `RunPolicy` result | runner `~/result` |
| `ManageEpisodeStatus` | `ManageEpisode` feedback | manager `~/status` |
| `ManageEpisodeResult` | `ManageEpisode` result | manager `~/result` |

Each message has the same field names, types and order as the action part it
mirrors.
