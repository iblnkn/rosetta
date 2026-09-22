# Run human-in-the-loop episodes

Run a policy with a teleop device ready to take over, and record what
happens. Parameters are under
[hil_manager_node](../reference/nodes.md#hil_manager_node).

## Declare the teleop device

Add a `teleop` section to the contract: an `input` whose `target` is the
action topic, and `events` mapping event names to buttons.
`contracts/so_101_hil.yaml` is a complete example. The schema is on the
[Contract](../reference/contract.md#teleop) page.

## Launch

```bash
ros2 launch rosetta hil_launch.py \
    contract_path:=robot_hil.yaml \
    action_remap_from:=/forward_position_controller/commands \
    pretrained_name_or_path:=<checkpoint> \
    policy_type:=act
```

`action_remap_from` is the contract's action topic. The launch fails if it
isn't one. With a reward classifier, `reward_remap_from` names the reward
topic the same way.

This starts the manager, the recorder, and a policy runner in namespace
`robot_policy`. With `enable_reward_classifier:=true` it also starts a
classifier in namespace `reward_classifier`. The runner publishes to
`/hil/policy<topic>`, and the manager forwards to `<topic>` while the policy
has control.

## Run an episode

```bash
ros2 action send_goal /manage_episode \
    rosetta_interfaces/action/ManageEpisode "{prompt: 'place cubes on tray'}"
```

Hold the `is_intervention` button to take over. Release it to hand back.
Press `end_success` or `end_failure` to finish. The services do the same:

```bash
ros2 service call /hil_manager/set_intervention std_srvs/srv/SetBool "{data: true}"
ros2 service call /hil_manager/end_episode std_srvs/srv/SetBool "{data: true}"
```

The result reports `termination_reason` and `outcome` separately. Nothing
here deletes a bag. For that, call the recorder's `delete_last_bag` service.

## End on a reward

Set `success_reward_threshold` on the goal. When the reward reaches it, the
episode ends with `termination_reason: reward_threshold`, and an unlabeled
outcome is recorded as `success`.
