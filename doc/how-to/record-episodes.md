# Record episodes

Record demonstrations as bags with the episode recorder. Every parameter
is listed under [episode_recorder_node](../reference/nodes.md#episode_recorder_node).

## Start the recorder

```bash
ros2 launch rosetta episode_recorder_launch.py \
    contract_path:=robot.yaml \
    bag_base_dir:=datasets/bags
```

Add `use_sim_time:=true` if the robot is a simulator publishing `/clock`.

The recorder records every topic on the graph, named in the contract or
not. The contract is a checklist: at the end of each episode the recorder
logs a message count per contract topic and marks any that got nothing with
`(!)`. To record contract topics only, set `record_all: false` in a params
file. To skip specific topics, list regexes in `exclude_topics`.

## Drive it from the keyboard

In a second terminal:

```bash
ros2 run rosetta episode_keyboard_node
```

`t` types the task prompt, `r` starts, `s` stops and keeps, `d` stops and
deletes. Each episode is one bag directory under `bag_base_dir`.

## Drive it from a script

Send the action goal. Stop with Ctrl+C on the client or call the cancel
service. Either way the bag is kept.

```bash
ros2 action send_goal /record_episode \
    rosetta_interfaces/action/RecordEpisode "{prompt: 'place cubes on tray'}"

ros2 service call /episode_recorder/cancel_recording std_srvs/srv/Trigger
```

An untimed recording ends `CANCELED`. That's the normal end, not an error.
To bound the episode instead, set `max_duration_s` on the goal.

To throw away the last bag:

```bash
ros2 service call /episode_recorder/delete_last_bag std_srvs/srv/Trigger
```

## Use bags you already have

Any rosbag2 bag with the contract's observation topics ports. `ros2 bag
record -a` works, and so do bags recorded before the contract existed. Put
each bag in its own directory under one parent and pass the parent to
`rosetta_port`.

## Check a bag

```bash
ros2 bag info datasets/bags/<episode>
```

Every observation topic in the contract needs a message count above zero, or
the episode won't port.
