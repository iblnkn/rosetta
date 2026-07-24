# Actions and Services

The ROS 2 action and service definitions Rosetta's nodes expose, from the
[`rosetta_interfaces`](https://github.com/iblnkn/rosetta_interfaces) package.

These are the **control plane** — start, stop, and manage. The **data plane**
(observations, actions, rewards) is plain ROS 2 topics declared by the
[contract](contract.md); no custom message types are involved there.

Which node serves what, and the parameters behind each, is in the
[nodes reference](nodes.md).

| Interface | Kind | Server | Purpose |
|---|---|---|---|
| `RecordEpisode` | action | `episode_recorder` | Record a demonstration to a rosbag |
| `RunPolicy` | action | `policy_runner` | Run policy (or classifier) inference |
| `ManageEpisode` | action | `hil_manager` | Orchestrate a human-in-the-loop episode |
| `StartHILEpisode` | service | `hil_manager` | Start a HIL episode, returning once it has started |
| `StartRecording` | service | `episode_recorder` | Lightweight start of recording |

## Actions

### RecordEpisode

```bash
ros2 action send_goal /record_episode \
    rosetta_interfaces/action/RecordEpisode \
    "{prompt: 'pick up the red block'}"
```

```
# Goal
string prompt                    # Description of this demonstration episode
---
# Result
bool success                     # Whether recording completed successfully
string message                   # Error details or success info
string bag_path                  # Path to the recorded rosbag
int32 messages_written           # Total messages captured
---
# Feedback
int32 seconds_remaining          # Time until auto-stop (from max_duration param)
int32 messages_written           # Messages captured so far (confirms data flow)
string status                    # Current state: "recording", etc.
```

The `prompt` is stored in bag metadata and becomes the frame's task label at
port time, unless the contract declares a `tasks` section — see
[per-frame task labels](contract.md#tasks-rewards-and-signals).

### RunPolicy

```bash
ros2 action send_goal /run_policy \
    rosetta_interfaces/action/RunPolicy \
    "{prompt: 'pick up the red block'}"
```

```
# Goal
string prompt                    # Task description for policy execution
---
# Result
bool success                     # Whether policy execution completed successfully
string message                   # Error details or success info
---
# Feedback
uint32 published_actions         # Total actions published so far
uint32 queue_depth               # Current action queue depth
string status                    # Current state: "executing", etc.
```

Use the same prompt the episodes were recorded with. For language-conditioned
policies this is the instruction the policy is conditioned on.

### ManageEpisode

Orchestrates a human-in-the-loop episode: drives policy inference, the
recorder, and optionally a reward classifier, muxes policy against teleop
control, applies classifier and human reward overrides, and monitors
termination.

```bash
ros2 action send_goal /manage_episode \
    rosetta_interfaces/action/ManageEpisode \
    "{prompt: 'pick up the red block', max_duration_s: 60.0, success_reward_threshold: 1.0}"
```

```
# Goal
string prompt                       # Task description for the episode
float64 max_duration_s              # Max episode duration (0.0 = use contract default)
float64 success_reward_threshold    # Reward threshold for auto-success (0.0 = disabled)
---
# Result
bool success                        # Whether the episode completed successfully
string message                      # Error details or success info
string bag_path                     # Path to the recorded rosbag
float64 final_reward                # Last reward value at episode end
int32 messages_written              # Total messages recorded
string termination_reason           # "timeout" | "human_stop" | "reward_threshold" | "cancelled"
---
# Feedback
float64 elapsed_s                   # Seconds since episode start
float64 current_reward              # Latest reward value
string control_source               # "policy" | "teleop"
string status                       # "starting" | "running" | "stopping"
int32 messages_written              # Messages recorded so far
```

See [Set up teleop and HIL](../how-to/teleop-and-hil.md) for the launch
topology this action drives.

## Services

### StartHILEpisode

Starts a HIL episode and returns once it has started. The episode itself runs
asynchronously — a synchronous response would hold an executor thread for the
whole episode. Results (bag path, final reward, termination reason) come from
the `ManageEpisode` action result or the node log; stop via
`/hil_manager/stop_episode`.

```bash
ros2 service call /hil_manager/start_episode \
    rosetta_interfaces/srv/StartHILEpisode \
    "{prompt: 'pick up the red block', max_duration_s: 60.0, success_reward_threshold: 1.0}"
```

```
# Request — mirrors the ManageEpisode goal
string prompt                       # Task description for the episode
float64 max_duration_s              # Max episode duration (0.0 = use contract default)
float64 success_reward_threshold    # Reward threshold for auto-success (0.0 = disabled)
---
bool accepted                       # Episode started (false: node not active / already running)
string message                      # Rejection reason or confirmation
```

### StartRecording

Lightweight start of the recorder, for callers that cannot route to the hidden
`_action/*` services — the Foxglove extension and the keyboard controller both
use this.

```bash
ros2 service call /episode_recorder/start_recording \
    rosetta_interfaces/srv/StartRecording \
    "{prompt: 'pick up the red block'}"
```

```
# Request
string prompt                    # Description of this demonstration episode
---
# Response
bool accepted                    # Whether recording was started
string message                   # Error details or confirmation
```

## Plain-`std_srvs` services

Not every control-plane call needs a custom type. The stop, discard, and
override services are `std_srvs/srv/Trigger` and `std_srvs/srv/SetBool`; they
are listed per node in the [nodes reference](nodes.md).
