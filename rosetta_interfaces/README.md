# rosetta_interfaces

ROS2 action definitions for Rosetta.

## Actions

### RecordEpisode

Trigger demonstration recording:

```bash
ros2 action send_goal /episode_recorder/record_episode \
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
ros2 action send_goal /rosetta_client/run_policy \
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

## Installation

```bash
colcon build --packages-select rosetta_interfaces
source install/setup.bash
```

## License

Apache-2.0
