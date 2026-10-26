# Interfaces

The `rosetta_interfaces` package holds every Rosetta ROS 2 interface. The nodes
also use `std_srvs/srv/Trigger`. This page covers the recorder's and the policy
runner's. The human-in-the-loop manager's (`ManageEpisode`, `StartHILEpisode`,
`ManageEpisodeStatus`, `ManageEpisodeResult`) are documented in the
[`rosetta_ros_hil` README](https://github.com/iblnkn/rosetta/tree/main/rosetta_ros_hil).

## Actions

### RecordEpisode

Server: `episode_recorder_node`, `/record_episode`.

| Goal | Type | Meaning |
|---|---|---|
| `prompt` | string | Task label stored in the bag. Empty uses `default_prompt`. |
| `max_duration_s` | float64 | Zero uses `default_max_duration_s`. |

| Result | Type | Meaning |
|---|---|---|
| `termination_reason` | string | `stopped`, `timeout`, `cancelled`, `node_deactivated`, `error` |
| `message` | string | |
| `bag_path` | string | |
| `messages_written` | int32 | |

Feedback: `elapsed_s` float64, `messages_written` int32.

### RunPolicy

Server: `policy_runner_node`, `/run_policy`.

| Goal | Type | Meaning |
|---|---|---|
| `prompt` | string | Task string passed to the policy. |
| `max_duration_s` | float64 | Zero uses `default_max_duration_s`. |

| Result | Type | Meaning |
|---|---|---|
| `termination_reason` | string | `completed`, `timeout`, `cancelled`, `node_deactivated`, `error` |
| `message` | string | |

Feedback: `published_actions` uint32, `queue_depth` uint32.

## Services

| Service | Request | Response |
|---|---|---|
| `StartRecording` | `prompt` string | `accepted` bool, `message` string |
| `StartPolicy` | `prompt` string | `accepted` bool, `message` string |

A start service returns as soon as the work is accepted. Work started this
way has no goal. Stop it with the matching cancel service, or let
`default_max_duration_s` end it. Neither `StartRecording` nor `StartPolicy`
has a duration field. Both use `default_max_duration_s`.

## Messages

Clients that start work through a service get no action feedback or result.
Each server therefore publishes the same fields on two topics, whatever
started the work. `~/status` carries the feedback fields at the feedback rate
while work runs. `~/result` carries the result fields once when it ends, and is
transient-local, so a client that subscribes later still gets the last one.

| Message | Fields | Topic |
|---|---|---|
| `RecordEpisodeStatus` | `RecordEpisode` feedback | `episode_recorder` `~/status` |
| `RecordEpisodeResult` | `RecordEpisode` result | `episode_recorder` `~/result` |
| `RunPolicyStatus` | `RunPolicy` feedback | `policy_runner` `~/status` |
| `RunPolicyResult` | `RunPolicy` result | `policy_runner` `~/result` |
