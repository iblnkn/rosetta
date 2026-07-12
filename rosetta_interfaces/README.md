# rosetta_interfaces

ROS2 action and service definitions for Rosetta.

These form the control plane (start/stop/manage). The data plane (observations,
actions, rewards) is plain ROS2 topics defined by the contract YAML.

| Interface | Kind | Server | Purpose |
|-----------|------|--------|---------|
| `RecordEpisode` | action | `episode_recorder` | Record a demonstration to a rosbag |
| `RunPolicy` | action | `rosetta_client` | Run policy (or classifier) inference |
| `ManageEpisode` | action | `rosetta_hil_manager` | Orchestrate a human-in-the-loop episode |
| `StartHILEpisode` | service | `hil_manager` | Start a HIL episode (returns once started) |
| `StartRecording` | service | `episode_recorder` | Lightweight start of recording |

## Actions

### RecordEpisode

Trigger demonstration recording:

```bash
ros2 action send_goal /record_episode \
    rosetta_interfaces/action/RecordEpisode \
    "{prompt: 'pick up the red block'}"
```

```
# Goal
string prompt           # Task description (stored in bag metadata)

# Result
bool success
string message
string bag_path         # Path to recorded bag
int32 messages_written

# Feedback
int32 seconds_remaining
int32 messages_written
string status           # "recording", "stopping", etc.
```

### RunPolicy

Trigger policy inference:

```bash
ros2 action send_goal /run_policy \
    rosetta_interfaces/action/RunPolicy \
    "{prompt: 'pick up the red block'}"
```

```
# Goal
string prompt           # Task description (for language-conditioned policies)

# Result
bool success
string message

# Feedback
uint32 published_actions
uint32 queue_depth
string status           # "executing", "waiting_for_policy", etc.
```

### ManageEpisode

Orchestrate a human-in-the-loop episode: drives policy inference, the recorder,
and (optionally) a reward classifier, muxing policy/teleop control and
classifier/human reward overrides, and monitors termination.

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

# Result
bool success                        # Whether the episode completed successfully
string message                      # Error details or success info
string bag_path                     # Path to the recorded rosbag
float64 final_reward                # Last reward value at episode end
int32 messages_written              # Total messages recorded
string termination_reason           # "timeout" | "human_stop" | "reward_threshold" | "cancelled"

# Feedback
float64 elapsed_s                   # Seconds since episode start
float64 current_reward              # Latest reward value
string control_source               # "policy" | "teleop"
string status                       # "starting" | "running" | "stopping"
int32 messages_written              # Messages recorded so far
```

## Services

### StartHILEpisode

Start a HIL episode and return once it has started (the episode itself runs
asynchronously — a synchronous response would hold an executor thread for the
whole episode). Results (bag path, final reward, termination reason) come from
the `ManageEpisode` action result or the node log; stop via the `stop_episode`
Trigger service.

```bash
ros2 service call /start_episode \
    rosetta_interfaces/srv/StartHILEpisode \
    "{prompt: 'pick up the red block', max_duration_s: 60.0, success_reward_threshold: 1.0}"
```

```
# Request — mirrors ManageEpisode goal
string prompt
float64 max_duration_s
float64 success_reward_threshold
---
bool accepted
string message
```

### StartRecording

Lightweight service to start the recorder. Primary interface for the Foxglove
extension and the keyboard controller, which cannot route to hidden
`_action/*` services.

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

## Installation

```bash
colcon build --packages-select rosetta_interfaces
source install/setup.bash
```

## License

Apache-2.0
