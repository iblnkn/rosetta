# Port Existing Bags

This guide shows you how to turn bags into a training dataset, whether Rosetta recorded them or not. Full CLI documentation: [porter CLI](../reference/porter-cli.md).

## Basic conversion

```bash
rosetta_port \
    --raw-dir ./datasets/bags \
    --contract ./contract.yaml \
    --repo-id my-org/my-dataset \
    --root ./datasets/lerobot
```

## Bring your own bags

The porter consumes standard rosbag2. Bags from `ros2 bag record`, `rosbag2_py` scripts, third-party tools, or an old archive all work. The recorder, keyboard, and HIL nodes are conveniences, not requirements. Write a contract for the topics your bags contain and port them.

What the porter expects:

- **One bag directory = one episode.** The porter finds bags by `metadata.yaml` and searches `--raw-dir` recursively. Split long multi-episode recordings into per-episode bags first.
- **Observation topics must be present.** A bag missing one fails the warmup gate. Missing action, reward, and signal topics zero-fill, same as the live bridge.
- **`align.timeline: header` needs stamped messages.** Unstamped messages on a header timeline drop at ingest, same as live.
- **Task labels need a source.** The porter reads per-frame tasks from a `tasks:` topic, falling back to the `lerobot.operator_prompt` field in bag metadata (the episode recorder writes this). Foreign bags have neither and produce an empty task string. Declare a task topic or write the metadata field before training VLA policies.
- **No embedded contract, no problem.** The mismatch warning only runs when a bag carries one. `--contract` defines the translation, and the output dataset still embeds the contract for deployment-time resolution.

## Re-port with a changed contract

Bags preserve raw data, so a contract change (keys, `fps`, alignment, operators, new topics) only needs a re-run:

```bash
rosetta_port --raw-dir ./datasets/bags --contract ./contract_v2.yaml \
    --repo-id my-org/my-dataset-v2 --root ./datasets/lerobot
```

## Parallelize large conversions

Shard a directory of bags across invocations:

```bash
rosetta_port --raw-dir ./bags --contract c.yaml --num-shards 8 --shard-index 0 &
rosetta_port --raw-dir ./bags --contract c.yaml --num-shards 8 --shard-index 1 &
# ...
```

For SLURM workflows, see the [LeRobot Porting Datasets Guide](https://huggingface.co/docs/lerobot/en/porting_datasets_v3) and substitute `rosetta_port` for `port_droid.py`.
