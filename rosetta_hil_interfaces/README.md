# rosetta_hil_interfaces

ROS 2 action and service definitions for [`rosetta_ros_hil`](../rosetta_ros_hil), the human-in-the-loop manager built on [Rosetta](https://github.com/iblnkn/rosetta).

| Interface | Kind | Server | Purpose |
|-----------|------|--------|---------|
| `ManageEpisode` | action | `hil_manager` | Run one human-in-the-loop episode |
| `StartHILEpisode` | service | `hil_manager` | Start an episode without the action protocol (returns once started) |
| `ManageEpisodeStatus`, `ManageEpisodeResult` | message | `hil_manager` | `~/status` and `~/result` topics, for clients that can't use the action |

`ManageEpisode` reports how an episode ended in `termination_reason` (the
shared reasons from `rosetta_interfaces`, plus `reward_threshold`) and whether
the robot did the task in `outcome` (`success`, `failure`, `unlabeled`). The
two are independent.
