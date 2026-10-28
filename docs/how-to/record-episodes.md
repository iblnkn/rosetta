# Record episodes

Record demonstrations as rosbag2 bags with the episode recorder, one bag
directory per episode. Every parameter is listed under
[episode_recorder_node](../reference/nodes.md#episode_recorder_node).

You need a built workspace, a contract for your robot (see
[Write a contract](write-a-contract.md)), and the robot or simulator
publishing its topics.

## Start the recorder

```bash
ros2 launch rosetta_ros episode_recorder_launch.py \
    contract_path:=robot.yaml \
    bag_base_dir:=datasets/bags
```

`robot.yaml` is your contract. `bag_base_dir` is where the bags go. A relative
path resolves against the directory you launch from, and it defaults to
`datasets/bags`. Add `use_sim_time:=true` if the robot is a simulator
publishing `/clock`. The recorder then records `/clock` too.

The recorder configures and activates on its own. It logs
`Configured: robot_type=<robot_type>, topics=<n>`, where `n` counts the
contract topics it records.

## Choose what gets recorded

By default the recorder records every topic on the graph, named in the
contract or not, like `ros2 bag record -a`. It looks for topics when an
episode starts, so a topic that first appears mid-episode is not in that bag.
`/rosout` and `/parameter_events` are always left out. For each camera it
keeps one image stream: the one the contract names, or else the first that
exists of `compressed`, `zstd`, `theora`, `compressedDepth` and raw.

The contract is a checklist. At the end of each episode the recorder logs a
message count per contract topic and marks any that got nothing with `(!)`.

The topic settings are not launch arguments, so set them in a params file:

```yaml
episode_recorder:
  ros__parameters:
    exclude_topics: ["^/episode_recorder/", "/_action/"]
```

```bash
ros2 launch rosetta_ros episode_recorder_launch.py \
    contract_path:=robot.yaml params_file:=my_recorder.yaml
```

The file layers over the package defaults, so it needs only the keys you
change. It must have an `episode_recorder` section (or `/**`), or the launch
fails with `no section applies to node 'episode_recorder'`.

- To skip topics, list regexes in `exclude_topics`. The example keeps the
  recorder's own topics, and the status and feedback topics of every
  action, out of the bag.
- To record contract topics only, set `record_all: false`.
- To keep an image stream the one-per-camera rule would drop, list a regex
  for it in `include_topics`.

## Drive it from the keyboard

In a second terminal:

```bash
ros2 run rosetta_ros episode_keyboard_node
```

It needs an interactive terminal. Without one it exits with
`Failed to enable raw terminal mode. Are you running in an interactive terminal?`

| Key | Action |
|---|---|
| `t` | Edit the task prompt for the next episode. Enter applies, Esc cancels. |
| `r` or Right arrow | Start recording. |
| `s` or Left arrow | Stop and keep the bag. |
| `d` or Backspace | Stop and delete the bag. After a stop, delete the bag that stop kept. |
| `h` or `?` | Show the keys. |
| `q` | Quit. |

If the recorder runs under a namespace, pass it with
`--ros-args -p recorder_ns:=/<namespace>/episode_recorder`.

## Drive it from a script

Send a `RecordEpisode` goal. To bound the episode, set `max_duration_s`:

```console
$ ros2 action send_goal /record_episode \
    rosetta_interfaces/action/RecordEpisode \
    "{prompt: place cubes on tray, max_duration_s: 2.0}"
Waiting for an action server to become available...
Sending goal:
     prompt: place cubes on tray
max_duration_s: 2.0

Goal accepted with ID: 76de3b830bb748b0abd42bad7cd633e3

Result:
    termination_reason: timeout
message: ''
bag_path: datasets/bags/1793143725_326991248
messages_written: 67

Goal finished with status: SUCCEEDED
```

`bag_path` is built from `bag_base_dir` as given, so a relative
`bag_base_dir` gives a path relative to where the recorder was launched.

Without `max_duration_s` the recording runs until you stop it. Stop it with
Ctrl+C on the client or with the cancel service. Either way the bag is kept,
and the goal ends `CANCELED` with `termination_reason: cancelled`. That's the
normal end of an untimed recording, not an error.

```bash
ros2 service call /episode_recorder/cancel_recording std_srvs/srv/Trigger
```

After Ctrl+C, `ros2 action send_goal` prints `Executor is already spinning`
instead of the result. Read the result on `/episode_recorder/result`, which
keeps the last one for late subscribers:

```bash
ros2 topic echo --once /episode_recorder/result
```

A client that can call services but not actions starts a recording with
`/episode_recorder/start_recording` (`rosetta_interfaces/srv/StartRecording`).
The call returns as soon as the recording starts. Follow it on
`/episode_recorder/status` and read how it ended on `/episode_recorder/result`.

The recorder runs one recording at a time. A second start is rejected: the
service answers `Rejected: already busy`, and `ros2 action send_goal` prints
`Goal was rejected.`

## Throw away a bag

```bash
ros2 service call /episode_recorder/delete_last_bag std_srvs/srv/Trigger
```

Called during a recording, it deletes that recording once it ends
(`Will delete <bag> when the recording ends`). Called between recordings, it
deletes the bag the last recording wrote. Only that one bag can be deleted:
once a new recording starts, the older bags stay. With nothing to delete it
answers `success: false` and `No bag to delete`.

## Check a bag

The end-of-episode log is the quickest check. Here, nothing published the
cameras or the leader arm:

```console
[episode_recorder_node-1] [INFO] [1793143727.367834703] [episode_recorder]: Recorded 67 messages to datasets/bags/1793143725_326991248 (2.0s)
[episode_recorder_node-1]   /front_camera/image_raw/compressed: 0 (!)
[episode_recorder_node-1]   /top_camera/image_raw/compressed: 0 (!)
[episode_recorder_node-1]   /wrist_camera/image_raw/compressed: 0 (!)
[episode_recorder_node-1]   /follower_arm/joint_states: 60
[episode_recorder_node-1]   /leader_arm/joint_states: 0 (!)
[episode_recorder_node-1]   /episode_recorder/status: 3
[episode_recorder_node-1]   /record_episode/_action/feedback: 3
[episode_recorder_node-1]   /record_episode/_action/status: 1
```

Contract topics come first and are always listed. The other topics are listed
only when they got messages.

For a bag already on disk:

```bash
ros2 bag info datasets/bags/<episode>
```

`<episode>` is the bag directory, named `<seconds>_<nanoseconds>` from the
time the recording started. Every contract topic is listed, with a count of 0
when nothing arrived. Every observation topic needs a count above zero, or
`rosetta_port` fails that episode and goes on with the others.

The bag's `metadata.yaml` holds the prompt under `custom_data` as
`lerobot.operator_prompt` and, unless `embed_contract` is off, the contract
text as `rosetta.contract_yaml`.

## Use bags you already have

Any rosbag2 bag that has the contract's observation topics ports. Bags from
`ros2 bag record -a` work, and so do bags recorded before the contract
existed. Put each bag in its own directory under one parent and pass the
parent to `rosetta_port` as `--raw-dir` (see
[Prepare a dataset](prepare-a-dataset.md)).
