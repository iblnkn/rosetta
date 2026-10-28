# rosetta_port

`rosetta_port` turns a directory of bags into a training dataset through a
contract. It is a console script in `rosetta_ros`.

```bash
ros2 run rosetta_ros rosetta_port --raw-dir <bags> --contract <robot.yaml> [options]
```

`<bags>` is the directory to search for bags, and `<robot.yaml>` is the
contract to decode them with. `python3 -m rosetta_ros.offline.port` runs the
same entry point.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--raw-dir` | required | Directory searched recursively for bag directories. |
| `--contract` | required | Contract YAML. It decodes every bag and is the copy embedded in the dataset. |
| `--repo-id` | the `--raw-dir` directory name | Dataset repo ID, such as `my_org/my_dataset`. |
| `--root` | the writer's default | Parent directory. The LeRobot writer puts the dataset at `<root>/<repo-id>`, or at `$HF_LEROBOT_HOME/<repo-id>` without `--root`. |
| `--framework` | `lerobot` | Dataset writer, by its name in the `rosetta.dataset_writers` entry-point group. `--help` lists the installed ones. |
| `--num-shards` | none | Total shards for a parallel run. |
| `--shard-index` | none | This run's shard, from 0 to `--num-shards` minus 1. Required with `--num-shards`, ignored without it. |
| `--no-embed-contract` | embed | Skip copying `--contract` into the dataset. |

LeRobot writer options:

| Option | Default | Meaning |
|---|---|---|
| `--push-to-hub` | off | Upload to the Hugging Face Hub after the last episode. |
| `--hub-public` | private | Make the uploaded dataset public. |
| `--hub-tags` | `rosetta,rosbag` | Comma-separated Hub tags. |
| `--vcodec` | `libsvtav1` | Video codec for image keys. |
| `--streaming-encoding` | off | Encode video frames as they arrive instead of through intermediate PNG files. |

Writer-specific options, passed to the writer only when given:
`--past-steps`, `--future-steps`, `--image-indices` (a comma list such as
`-1,0`) and `--samples-per-shard`. The LeRobot writer ignores all four.

## Behavior

- A bag directory is any directory under `--raw-dir` holding a
  `metadata.yaml`. One bag is one episode. Episodes are ported in sorted path
  order.
- No bag directory under `--raw-dir` is an error:
  `RuntimeError: No bag directories found in <raw-dir>`.
- The bag's `storage_identifier` picks the rosbag2 storage plugin, `mcap`
  when the metadata names none.
- Shard `i` of `n` takes every nth bag in sorted order, starting at bag `i`.
  An empty shard logs `No bags to process in this shard`, writes nothing and
  exits 0.
- Every shard writes its own dataset, so shards that share `--root` and
  `--repo-id` collide. The later shard fails with `FileExistsError`.
- Ticks start at the bag's start time and are `1 / fps` apart. Ticks before
  every observation source has a sample are skipped.
- When the first bag's embedded contract differs from `--contract` after
  parsing, the porter logs this warning and carries on:

  ```text
  Bag was recorded with a different contract than --contract; using --contract for decoding and embedding.
  ```
- A bag missing an observation topic fails that episode with `observation
  topics missing from bag`. A present observation topic that never yields a
  sample fails it with `no frames emitted`.
- A bag missing an action, reward, signal, info or teleop topic ports with
  that key zero-filled and a warning: `Topic <topic> not in bag, <key> will be
  zero-filled`.
- A source whose type has no decoder fails every episode at open. A topic the
  contract declares with two different types fails every episode too.
- A failed episode is discarded, logged, and the run continues. The run ends
  with a `Completed: <ok>/<total> episodes (<failed> failed) in <seconds>s` line and one
  warning per failed episode.
- If every episode fails, the run exits non-zero with `RuntimeError: All <n>
  bags failed to convert`. The dataset directory and its `meta/` already exist
  by then.
- The LeRobot writer refuses a dataset directory that already exists, with
  `FileExistsError`. Delete it, or pick another `--root` or `--repo-id`,
  before running again.
- The task string per frame is the newest `tasks` message at or before the
  tick, or else the prompt stored in the bag's metadata. See
  [Tasks](contract.md#tasks).

## Output

With the LeRobot writer, a LeRobot v3.0 dataset at `<root>/<repo-id>`.

- Features are the contract's recorded keys (observations, actions, the four
  record-only sections, and any `teleop.input.*` and `teleop.feedback.*`
  keys) plus `is_first`, `is_last` and `is_terminal`, and LeRobot's own
  `timestamp`, `frame_index`, `episode_index`, `index` and `task_index`. `is_first` marks the
  first frame after warmup. `is_terminal` equals `is_last`.
- `robot_type` and `fps` come from the contract.
- `meta/rosetta_contract.yaml` holds a copy of `--contract` unless
  `--no-embed-contract` is given. `policy_runner_node` can find its contract
  there. See [Nodes](nodes.md).
