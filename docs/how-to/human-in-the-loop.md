# Run human-in-the-loop episodes

Run episodes in which a policy drives the robot, a person can take over from
a teleop device at any time, and each episode is recorded and labeled
success or failure. The
[`rosetta_ros_hil`](https://github.com/iblnkn/rosetta/tree/main/robots/ros/rosetta_ros_hil)
package runs them. Its README lists every parameter, service and topic of the
HIL manager.

You need:

- `rosetta_ros_hil` built in your workspace, next to `rosetta_ros`.
- A trained policy for your contract (see [Deploy a policy](deploy-a-policy.md)).
- A teleop device the robot's commands can be translated from, such as a
  leader arm, and buttons for the episode events, such as a gamepad on `/joy`.
- Optionally, a trained reward classifier.

## Add a teleop section to the contract

The manager reads the teleop device from the contract's `teleop` section. The
package's `contracts/so_101_hil.yaml` is a complete example:

```yaml
teleop:
  input:
    - target: /leader_arm/joint_states
      channel: {topic: /human/leader_arm/joint_states,
                type: sensor_msgs/msg/JointState, qos: *state_qos}
      align: {strategy: hold, timeline: header}
      select: [position.shoulder_pan, position.shoulder_lift, position.elbow,
               position.wrist_pitch, position.wrist_roll, position.wrist_yaw]
      apply: [rad2deg]

  events:
    channel: {topic: /joy, type: sensor_msgs/msg/Joy, qos: *event_qos}
    select: {is_intervention: buttons.1, start_episode: buttons.3, end_success: buttons.2, end_failure: buttons.0}
```

- `input`: each entry's `target` is the topic of one action source. The
  device's message is decoded with the entry's `select` and `apply` and
  encoded as that action, so the device and the robot may use different
  message types and units. If two sources share the target topic, configure
  fails with `forwarding needs exactly one`.
- `events`: maps event names to buttons. The manager acts on
  `is_intervention`, `start_episode`, `success`, `failure`, `end_success` and
  `end_failure`, and logs a warning at configure for any other name. Check
  which index each button has with `ros2 topic echo /joy` before you rely on
  it.

The full rules are under [Teleop](../reference/contract.md#teleop) in the
contract reference.

A human label is published on every reward topic of the contract by setting
the message's `data` field, so declare rewards with a type like
`std_msgs/msg/Float64`.

## Launch

```bash
ros2 launch rosetta_ros_hil hil_launch.py \
    contract_path:=robot_hil.yaml \
    pretrained_name_or_path:=<checkpoint> \
    policy_type:=act
```

`robot_hil.yaml` is your contract. It defaults to the package's
`so_101_hil.yaml`. `<checkpoint>` is the trained policy, a local path or a
Hugging Face Hub repo ID, and `policy_type` must match it. Run
`ros2 launch rosetta_ros_hil hil_launch.py --show-args` for every argument.

The launch file starts these nodes and configures and activates each:

| Node | Role |
|---|---|
| `/robot_policy/policy_runner` | The policy, publishing every action topic under `/hil/policy` (`policy_remap_prefix`). |
| `/reward_classifier/policy_runner` | The reward classifier, only with `enable_reward_classifier:=true`. Publishes every reward topic under `/hil/reward` (`reward_remap_prefix`). |
| `/episode_recorder` | The episode recorder. It records the contract topics, so the bag holds the commands the robot got, whoever sent them. |
| `/hil_manager` | Forwards the policy's or the person's commands onto the action topics, and runs the episodes. |

The manager logs each route at configure. With `so_101_hil.yaml`:

```console
[hil_manager_node-3] [INFO] [1793144303.129239501] [hil_manager]: Policy output: /hil/policy/leader_arm/joint_states -> /leader_arm/joint_states
[hil_manager_node-3] [INFO] [1793144303.130129308] [hil_manager]: Teleop input: /human/leader_arm/joint_states -> /leader_arm/joint_states
[hil_manager_node-3] [INFO] [1793144303.130864114] [hil_manager]: Reward publisher: /reward
[hil_manager_node-3] [INFO] [1793144303.131638163] [hil_manager]: Teleop events: /joy
```

The runners and the recorder take their settings from params files in the
package's `params/`, layered over `rosetta_ros`'s defaults. They hold only
what HIL changes, such as a 30 Hz feedback rate. To tune a node, for example
`actions_per_chunk` for the policy, pass your own file with
`robot_policy_params_file:=`, `reward_classifier_params_file:=` or
`episode_recorder_params_file:=`. It needs only the keys you change, under a
`policy_runner` or `episode_recorder` section. Deployment values
(`pretrained_name_or_path`, `policy_type`, `policy_device`, `server_address`,
`bag_base_dir`, `storage_id`) are launch arguments, where empty keeps the
params-file value. `policy_device` applies to both runners.

To add the reward classifier, also pass:

```bash
    enable_reward_classifier:=true \
    reward_classifier_pretrained_name_or_path:=<classifier>
```

`<classifier>` is the trained reward classifier. The classifier uses
`contract_path` unless you set `reward_classifier_contract_path`. Its params
file sets `policy_type` to `reward_classifier` and `server_address` to
`127.0.0.1:8081`, next to the policy's `127.0.0.1:8080`.

## Run an episode

Send a `ManageEpisode` goal:

```bash
ros2 action send_goal /manage_episode \
    rosetta_interfaces/action/ManageEpisode \
    "{prompt: 'place cubes on tray', max_duration_s: 90.0}" --feedback
```

The manager starts a recording, then the policy, then the classifier if it is
enabled. If the recorder or the policy isn't up within 10 s or rejects the
goal, the episode ends at once with `termination_reason: error` and a
message such as `Failed to start robot policy`. A classifier that fails to
start is logged and the episode goes on without it.

Feedback carries `elapsed_s`, `current_reward`, `control_source` (`policy` or
`teleop`), the `outcome` so far, and `messages_written`.

You can also start an episode with the `start_episode` button or the
`/hil_manager/start_episode` service (`rosetta_interfaces/srv/StartHILEpisode`).
Those have no action client, so follow the episode on `/hil_manager/status` and
read how it ended on `/hil_manager/result`. The manager runs one episode at a
time. A second start is rejected: the service answers `Rejected: already busy`,
and a goal is rejected.

## Take over

Hold the `is_intervention` button to take control, and release it to hand
control back to the policy. From a script:

```bash
ros2 service call /hil_manager/set_intervention std_srvs/srv/SetBool "{data: true}"
```

While the person has control, the manager drops the policy's commands and
forwards the teleop input instead. `/hil_intervention` (`std_msgs/msg/Int8`)
says who has control: `0` the policy, `1` the person.

Take-over works between episodes too, so you can drive the robot to a start
pose. An episode starts with the person in control only if the
`is_intervention` button is still held. Otherwise it starts with the policy
in control, even after `set_intervention` set the person.

Nothing watches the person's commands. If the device stops publishing while
the person has control, the robot gets no further commands from the manager.
The policy runner's safety command goes through `/hil/policy` like
the policy's other output, so it reaches the robot only while the policy has
control.

## End and label the episode

| To | Button | Service | `termination_reason` | Goal status |
|---|---|---|---|---|
| End as a success | `end_success` | `/hil_manager/end_episode` with `data: true` | `stopped` | `SUCCEEDED` |
| End as a failure | `end_failure` | `/hil_manager/end_episode` with `data: false` | `stopped` | `SUCCEEDED` |
| Abandon it | none | `/hil_manager/cancel_episode` | `cancelled` | `CANCELED` |
| Label without ending | `success`, `failure` | `/hil_manager/set_reward_override` | not ended | not ended |
| Retract the label | none | `/hil_manager/clear_reward_override` | not ended | not ended |

The episode also ends on its own:

- After `max_duration_s` (or the manager's `default_max_duration_s` when the
  goal leaves it at 0), with `termination_reason: timeout`. Both default to
  no limit.
- When the reward reaches the goal's `success_reward_threshold` (0, the
  default, turns this off), with `termination_reason: reward_threshold`. An
  unlabeled episode is then reported as a success.

A label sets the reward to `human_reward_positive` (default `1.0`) or
`human_reward_negative` (default `-1.0`), publishes it on the reward topics,
and holds until another label replaces it, it is retracted, or the next
episode starts. While it holds, the classifier's rewards are ignored. A
success label with a `success_reward_threshold` at or below
`human_reward_positive` therefore also ends the episode, with
`reward_threshold`.

The result reports `termination_reason` (how the episode ended) and `outcome`
(`success`, `failure` or `unlabeled`) separately, so a timed-out episode can
still be a success.

## Find the bag

The result's `bag_path` is the episode's bag. Nothing in the manager deletes
a bag, even for a failed or abandoned episode. To drop the last one, call the
recorder:

```bash
ros2 service call /episode_recorder/delete_last_bag std_srvs/srv/Trigger
```

The recorder's own result for an episode always says `cancelled`, because the
manager stops it with a cancel. The real reason is on the `ManageEpisode`
result. See [Record episodes](record-episodes.md) for checking a bag.

## Run without recording or with your own policy

- To run episodes without writing bags, pass `enable_recording:=false`.
  Everything else works the same.
- To run around a policy started some other way, pass
  `manage_policy_lifecycle:=false`. The manager then never sends or cancels a
  `RunPolicy` goal and only forwards, labels and records. Your policy must
  publish under `policy_remap_prefix`. `hil_launch.py` still starts
  `/robot_policy/policy_runner`, which then sits idle.
