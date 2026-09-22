# rosetta_port

Turns a directory of bags into a training dataset through a contract.

```bash
ros2 run rosetta rosetta_port --raw-dir <bags> --contract <robot.yaml> [options]
```

## Options

| Option | Default | Meaning |
|---|---|---|
| `--raw-dir` | required | Directory searched recursively for bag directories. |
| `--contract` | required | Contract YAML. |
| `--repo-id` | the `--raw-dir` name | Dataset id. |
| `--root` | the framework's cache | Parent directory. The dataset lands at `<root>/<repo-id>`. |
| `--framework` | `lerobot` | Dataset writer, by entry-point name. `--help` lists the installed ones. |
| `--num-shards` | none | Total shards for a parallel run. |
| `--shard-index` | none | This shard. Required with `--num-shards`. |
| `--no-embed-contract` | embed | Skip the `meta/rosetta_contract.yaml` copy. |

LeRobot writer options:

| Option | Default | Meaning |
|---|---|---|
| `--push-to-hub` | off | Upload after writing. Private unless `--hub-public`. |
| `--hub-public` | off | |
| `--hub-tags` | `rosetta,rosbag` | Comma list. |
| `--vcodec` | `libsvtav1` | Video codec. |
| `--streaming-encoding` | off | Encode video while writing. |

Writer-specific options the LeRobot writer ignores: `--past-steps`,
`--future-steps`, `--image-indices`, `--samples-per-shard`.

## Behavior

- A bag directory is any directory holding `metadata.yaml`. One bag is one
  episode. Episodes are ported in sorted path order. No bag directory under
  `--raw-dir` is an error.
- The bag's `storage_identifier` picks the rosbag2 reader. Any installed
  storage plugin works.
- Shard `i` of `n` takes every nth bag starting at `i`. An empty shard
  writes nothing and exits normally.
- The first bag's embedded contract is compared with `--contract`. If they
  differ, the porter logs `Bag was recorded with a different contract than
  --contract; using --contract for decoding and embedding.`
- A bag missing an observation topic, or holding one with no messages, fails
  that episode. A bag missing an action, reward or signal topic ports with
  that key zero-filled and a warning.
- A contract source whose type has no decoder, built-in or custom, fails
  every episode at open.
- A failed episode is discarded and the run continues. If every bag fails,
  the run exits with an error. The dataset directory and its `meta/` are
  created before the first episode, so they'll be on disk. Delete them before
  retrying: the LeRobot writer refuses to open an existing dataset directory.
- The task string per frame is the newest `tasks` message at or before the
  tick, or else the prompt stored in the bag's metadata.

## Output

With the LeRobot writer, a LeRobot v3 dataset at `<root>/<repo-id>`. Features
are the contract keys plus `is_first`, `is_last` and `is_terminal`.
`robot_type` and `fps` come from the contract. `meta/rosetta_contract.yaml`
holds the contract text unless you passed `--no-embed-contract`.
