# rosetta_ros_hil

A human-in-the-loop episode manager built on
[Rosetta](https://github.com/iblnkn/rosetta). It runs a policy with a teleop
device ready to take over, an optional reward classifier, and the episode
recorder, and reports how each episode ended and whether the robot did the
task.

`rosetta_ros_hil` uses only the public Python API of `rosetta` and
`rosetta_ros` and their actions (`RecordEpisode`, `RunPolicy`). Its
interfaces (`ManageEpisode`, `StartHILEpisode`, `ManageEpisodeStatus`,
`ManageEpisodeResult`) live with the rest of Rosetta's in
[`rosetta_interfaces`](../rosetta_interfaces).

## Usage

Add a `teleop` section to your contract (this package's
`contracts/so_101_hil.yaml` is a complete example), then launch:

```bash
ros2 launch rosetta_ros_hil hil_launch.py \
    contract_path:=robot_hil.yaml \
    pretrained_name_or_path:=<checkpoint> \
    policy_type:=act
```

`hil_launch.py` includes `rosetta_ros`'s `policy_runner_launch.py` (in
namespace `robot_policy`, its output under `/hil/policy`) and
`episode_recorder_launch.py`, and starts the manager next to them. With
`enable_reward_classifier:=true` it includes the policy runner a second time,
as a classifier in namespace `reward_classifier`. Run `--show-args` for every
argument.

Start an episode, take over with the `is_intervention` button, and end it
with `end_success` or `end_failure`:

```bash
ros2 action send_goal /manage_episode \
    rosetta_interfaces/action/ManageEpisode "{prompt: 'place cubes on tray'}"
```

The full walkthrough is
[Run human-in-the-loop episodes](https://iblnkn.github.io/rosetta/how-to/human-in-the-loop.html).

Things to know before a run:

- Nothing watches the person's commands. If the teleop device stops
  publishing while it has control, the robot gets no further commands. The
  policy runner's watchdog covers only the policy's output.
- An episode starts with the policy in control, unless the `is_intervention`
  button is still held.
- Nothing here deletes a bag. Call the recorder's `delete_last_bag` service.
- There is no environment reset between episodes. Put the scene back
  yourself, by hand or with teleop between episodes.

## Reference

`hil_manager_node` is a lifecycle node named `hil_manager`, configured and
activated by `hil_launch.py`. Its defaults come from `params/hil_manager.yaml`.
It runs one episode at a time. A start while an episode runs, or while the
node is not active, is rejected. On deactivate it waits up to 10 s for a
running episode to stop.

### Parameters

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `contract_path` | string | `""` | Contract YAML. Required, configure fails when empty. `hil_launch.py` defaults it to this package's `contracts/so_101_hil.yaml`. Read-only. |
| `enable_recording` | bool | `true` | Send a `RecordEpisode` goal per episode. When false, nothing is written to a bag and everything else runs the same. Read-only. |
| `manage_policy_lifecycle` | bool | `true` | Send a `RunPolicy` goal at episode start and cancel it at the end. When false, no policy goal is ever sent or cancelled. Read-only. |
| `enable_reward_classifier` | bool | `false` | Send a `RunPolicy` goal to the classifier per episode and forward its rewards from `reward_remap_prefix`. Read-only. |
| `policy_action_name` | string | `/robot_policy/run_policy` | The policy runner's `RunPolicy` action. Read-only. |
| `reward_classifier_action_name` | string | `/reward_classifier/run_policy` | The classifier's `RunPolicy` action. Read-only. |
| `recorder_action_name` | string | `/record_episode` | The episode recorder's `RecordEpisode` action. Read-only. |
| `policy_remap_prefix` | string | `/hil/policy` | The policy publishes each action topic as `<prefix><topic>`. The manager forwards it to `<topic>` while the policy has control. Read-only. |
| `reward_remap_prefix` | string | `/hil/reward` | The classifier publishes each reward topic as `<prefix><topic>`. The manager forwards it to `<topic>` while no human label holds. Read-only. |
| `human_reward_positive` | double | `1.0` | Reward published for a success label. Read at each label. |
| `human_reward_negative` | double | `-1.0` | Reward published for a failure label. Distinct from the `0.0` of an unlabeled episode. Read at each label. |
| `default_prompt` | string | `""` | Prompt for a goal, service call or `start_episode` press that leaves it empty. Read at each episode start. |
| `default_max_duration_s` | double | `0.0` | Episode limit in seconds when the request leaves `max_duration_s` at 0. 0 or negative means no limit. Read at each episode start. |
| `feedback_rate_hz` | double | `30.0` | Rate of feedback, `~/status` and `hil_intervention` during an episode, and of the termination checks. 0.1 to 1000. Read at each episode start. |

### Interfaces

| Interface | Type | Effect |
|---|---|---|
| `manage_episode` | action `rosetta_interfaces/action/ManageEpisode` | Runs one episode. Goal `prompt`, `max_duration_s` (0 uses `default_max_duration_s`), `success_reward_threshold` (0 disables). |
| `~/start_episode` | service `rosetta_interfaces/srv/StartHILEpisode` | Starts an episode with the goal's fields and returns at once. |
| `~/end_episode` | service `std_srvs/srv/SetBool` | Ends the episode with a label, `true` success and `false` failure. `termination_reason` is `stopped` and the goal succeeds. Refused with no active episode. |
| `~/cancel_episode` | service `std_srvs/srv/Trigger` | Abandons the episode. An action goal ends `CANCELED`. The label stays as it was. Refused with no active episode. |
| `~/set_intervention` | service `std_srvs/srv/SetBool` | `true` hands control to teleop, `false` to the policy. Works between episodes. |
| `~/set_reward_override` | service `std_srvs/srv/SetBool` | Labels without ending, like the `success` and `failure` buttons. Refused with no active episode. |
| `~/clear_reward_override` | service `std_srvs/srv/Trigger` | Retracts the label, so classifier rewards flow again. Refused with no active episode. |
| `hil_intervention` | topic `std_msgs/msg/Int8` | `0` policy, `1` teleop. Reliable, transient-local, depth 1. Published on every change, between episodes too, and at `feedback_rate_hz` during an episode. |
| `~/status` | topic `rosetta_interfaces/msg/ManageEpisodeStatus` | The feedback fields at `feedback_rate_hz` during an episode, however it started. |
| `~/result` | topic `rosetta_interfaces/msg/ManageEpisodeResult` | The result fields, once per episode, including one that failed to start. Transient-local, so a late subscriber gets the last one. |

Topics from the contract:

- Subscribes to `<policy_remap_prefix><topic>` and publishes `<topic>` for
  each action topic.
- Subscribes to each `teleop.input` channel and each `teleop.feedback` origin,
  and publishes each `teleop.feedback` channel. Feedback reaches the device
  whoever has control.
- Publishes every reward topic. With the classifier enabled, subscribes to
  `<reward_remap_prefix><topic>` for each.
- Subscribes to the `teleop.events` channel.

A teleop input target or feedback origin topic shared by two contract sources
fails configure. Reward topics need a message type with a numeric `data`
field, such as `std_msgs/msg/Float64`.

### Teleop events

Events are edge-triggered: the manager acts when a button's value changes.

| Event | Acts on | Effect |
|---|---|---|
| `is_intervention` | press and release | Teleop has control while held. |
| `start_episode` | press | Starts an episode with `default_prompt`. Ignored while one runs. |
| `success`, `failure` | press | Labels the episode without ending it. |
| `end_success`, `end_failure` | press | Labels the episode and ends it with `stopped`. |

The four label events are ignored when no episode is running. A label holds
until another label, `~/clear_reward_override`, or the next episode. Event
names are open in the contract, so the manager ignores names outside these
six and logs them as a warning at configure.

### Result

`termination_reason` says how the episode ended, and `outcome` says whether
the robot did the task.

| `termination_reason` | Cause | Goal status |
|---|---|---|
| `stopped` | `~/end_episode`, or an `end_success` or `end_failure` press | `SUCCEEDED` |
| `timeout` | The episode reached its maximum duration | `SUCCEEDED` |
| `reward_threshold` | The reward reached `success_reward_threshold`. An unlabeled outcome becomes `success`. | `SUCCEEDED` |
| `cancelled` | An action cancel or `~/cancel_episode` | `CANCELED` |
| `node_deactivated` | The manager was deactivated during the episode | `ABORTED` |
| `error` | The recorder or policy goal failed to start, or the episode raised | `ABORTED` |

`outcome` is `success`, `failure` or `unlabeled`.

## License

Apache-2.0
