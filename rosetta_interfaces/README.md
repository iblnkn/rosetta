# rosetta_interfaces

ROS 2 action and service definitions for [Rosetta](https://github.com/iblnkn/rosetta).

These form the control plane (start/stop/manage). The data plane (observations,
actions, rewards) is plain ROS 2 topics defined by the contract YAML.

| Interface | Kind | Server | Purpose |
|-----------|------|--------|---------|
| `RecordEpisode` | action | `episode_recorder` | Record a demonstration to a rosbag |
| `RunPolicy` | action | `policy_runner` | Run policy (or classifier) inference |
| `ManageEpisode` | action | `hil_manager` | Orchestrate a human-in-the-loop episode |
| `StartHILEpisode` | service | `hil_manager` | Start a HIL episode (returns once started) |
| `StartRecording` | service | `episode_recorder` | Lightweight start of recording |

Field-by-field definitions with example calls:
**[Actions and services reference](https://iblnkn.github.io/rosetta/reference/interfaces.html)**.
The `.action` and `.srv` files in this repo are the source of truth — they are
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
