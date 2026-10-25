# rosetta_hil

A human-in-the-loop episode manager built on
[Rosetta](https://github.com/iblnkn/rosetta). It runs a policy with a teleop
device ready to take over, an optional reward classifier, and the episode
recorder, and reports how each episode ended and whether the robot did the
task.

`rosetta_hil` uses only Rosetta's public Python API and its actions
(`RecordEpisode`, `RunPolicy`). Its own interfaces live in
[`rosetta_hil_interfaces`](https://github.com/iblnkn/rosetta_hil_interfaces).

## Declare the teleop device

Add a `teleop` section to the contract: an `input` whose `target` is the
action topic, and `events` mapping event names to buttons. Rosetta's
`contracts/so_101_hil.yaml` is a complete example, and the schema is on
Rosetta's Contract reference page.

## Launch

```bash
ros2 launch rosetta_hil hil_launch.py \
    contract_path:=robot_hil.yaml \
    action_remap_from:=/forward_position_controller/commands \
    pretrained_name_or_path:=<checkpoint> \
    policy_type:=act
```

`action_remap_from` is the contract's action topic. The launch fails if it
isn't one. With a reward classifier, `reward_remap_from` names the reward
topic the same way. Run `--show-args` for every argument; `contract_path`
defaults to Rosetta's `so_101_hil.yaml`.

This starts the manager, Rosetta's recorder, and a Rosetta policy runner in
namespace `robot_policy`. With `enable_reward_classifier:=true` it also starts
a classifier in namespace `reward_classifier`. The runner publishes to
`/hil/policy<topic>`, and the manager forwards to `<topic>` while the policy
has control. One `feedback_rate_hz`, default `30.0`, goes to the manager, the
recorder and both runners.

## Run an episode

```bash
ros2 action send_goal /manage_episode \
    rosetta_hil_interfaces/action/ManageEpisode "{prompt: 'place cubes on tray'}"
```

Hold the `is_intervention` button to take over and release it to hand back.
Press `end_success` or `end_failure` to finish. The services do the same:

```bash
ros2 service call /hil_manager/set_intervention std_srvs/srv/SetBool "{data: true}"
ros2 service call /hil_manager/end_episode std_srvs/srv/SetBool "{data: true}"
```

The result reports `termination_reason` and `outcome` separately. Nothing
here deletes a bag. For that, call the recorder's `delete_last_bag` service.

To end on a reward, set `success_reward_threshold` on the goal. When the
reward reaches it, the episode ends with `termination_reason:
reward_threshold`, and an unlabeled outcome is recorded as `success`.

## Reference

`hil_manager_node` is a lifecycle node named `hil_manager`. Defaults come from
`params/hil_manager.yaml`. On deactivate it waits up to 10 s for a running
episode to stop.

| Parameter | Default | Meaning |
|---|---|---|
| `contract_path` | `""` | Required. The launch file defaults to Rosetta's `contracts/so_101_hil.yaml`. |
| `enable_recording` | `true` | Send a `RecordEpisode` goal per episode. |
| `manage_policy_lifecycle` | `true` | Send and cancel a `RunPolicy` goal per episode. |
| `enable_reward_classifier` | `false` | |
| `policy_action_name` | `/robot_policy/run_policy` | |
| `reward_classifier_action_name` | `/reward_classifier/run_policy` | |
| `recorder_action_name` | `/record_episode` | |
| `policy_remap_prefix` | `/hil/policy` | The runner publishes actions to `<prefix><action topic>`. The manager forwards to `<action topic>` while the policy is in control. |
| `reward_remap_prefix` | `/hil/reward` | Same for the classifier. |
| `human_reward_positive` | `1.0` | |
| `human_reward_negative` | `-1.0` | |
| `default_prompt` | `""` | |
| `default_max_duration_s` | `0.0` | |
| `feedback_rate_hz` | `30.0` | |

| Interface | Type | Effect |
|---|---|---|
| `manage_episode` | action `rosetta_hil_interfaces/action/ManageEpisode` | |
| `~/start_episode` | service `rosetta_hil_interfaces/srv/StartHILEpisode` | |
| `~/end_episode` | service `std_srvs/srv/SetBool` | Ends the episode. `true` labels success, `false` failure. The goal succeeds. |
| `~/cancel_episode` | service `std_srvs/srv/Trigger` | Abandons the episode. The goal ends `CANCELED`. |
| `~/set_intervention` | service `std_srvs/srv/SetBool` | `true` hands control to teleop, `false` to the policy. |
| `~/set_reward_override` | service `std_srvs/srv/SetBool` | Labels without ending. |
| `~/clear_reward_override` | service `std_srvs/srv/Trigger` | |
| `hil_intervention` | topic `std_msgs/msg/Int8` | `0` policy, `1` human. Published at the feedback rate while an episode runs. |

Teleop events from the contract are edge-triggered: an `is_intervention`
press hands control to teleop and its release hands it back, `start_episode`
starts an episode with `default_prompt`, `success` and `failure` set the
label, and `end_success` and `end_failure` end the episode. A label holds
until the other button, `~/clear_reward_override`, or the next episode.

Result `termination_reason`: `stopped`, `timeout`, `reward_threshold`,
`cancelled`, `node_deactivated`, `error`. `outcome`: `success`, `failure`,
`unlabeled`.
