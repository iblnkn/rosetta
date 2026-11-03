# Train without porting

Train a LeRobot policy straight from your recordings with
`rosetta-lerobot-train`. It runs LeRobot's `lerobot-train` with a Rosetta
reader in place of a ported dataset. The reader applies the contract while
training reads the data, so a contract change takes effect on the next run
with nothing to port again.

The reader that ships with Rosetta is `rerun`, from `rerun_rosetta`. It reads
`.rrd` recordings registered to a Rerun catalog server.

You need recordings made with the `rrd` storage plugin, a contract, and
LeRobot installed with its `training` extra (see
[Installation](../installation.md)). `rosetta-lerobot-train` comes with
`lerobot_dataset_rosetta`. In the repository's pixi environment it is on the
`PATH`.

## Record and register

Record with the `rrd` storage plugin
([`rosbag2_storage_rrd`](https://github.com/rerun-io/rosbag2_storage_rrd)):

```bash
ros2 launch rosetta_ros episode_recorder_launch.py contract_path:=robot.yaml storage_id:=rrd
```

Start a catalog server with `rerun server` in its own terminal. It prints the
address it listens on. Register the recordings, and compute the
normalization stats once per contract:

```python
from rerun.catalog import CatalogClient
from rerun_rosetta import write_stats_layer

dataset = CatalogClient("rerun+http://127.0.0.1:51234").create_dataset("my_task", exist_ok=True)
dataset.register_prefix("file:///abs/path/datasets/bags").wait()
write_stats_layer(dataset, "robot.yaml", "datasets/rosetta_stats")
```

`file://` needs the server on the same filesystem as the recordings.
`write_stats_layer` computes only the episodes that don't have stats yet, so
run it again after registering new recordings. Stats belong to one contract,
so run it again after changing the contract too.

## Train

```bash
rosetta-lerobot-train \
    --rosetta.reader=rerun \
    --rosetta.contract=robot.yaml \
    --rosetta.catalog_url=rerun+http://127.0.0.1:51234 \
    --rosetta.dataset=my_task \
    --policy.type=act \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir=outputs/train/my_policy
```

`--rosetta.token` authenticates against a hosted catalog.
`--rosetta.manifest=<path>.parquet` keeps Rerun's sample index between runs.
Delete that file after registering new recordings, or after changing the
contract, the policy's windows or `--num_workers`. A run with a file that no
longer matches stops and names what changed.

`lerobot-train` flags work the same way as in [Train a policy](train.md),
`--dataset.episodes`, `--dataset.eval_split` and `--resume` included.
`--dataset.episodes` trains on those episodes only. `--dataset.eval_split`
holds out the episodes it would hold out of a ported dataset: for each task,
the last `ceil(n × eval_split)` of its `n` episodes, among
`--dataset.episodes` when you pass it. `rosetta-lerobot-train` sets a
placeholder `--dataset.repo_id=rosetta-lerobot-train/<reader>` unless you pass one. Two
flags differ:

- `--dataset.streaming=true` is refused. LeRobot would read the samples in
  episode order. Rerun's block shuffle is the one way to stream (see
  [Read in blocks](#read-in-blocks)).
- `--max_eval_samples` has no effect. LeRobot applies it only to datasets it
  loads itself, so each eval reads every held-out sample.

## Read in blocks

By default the reader serves samples by index under LeRobot's sampler,
fetching each batch from the server on its own. Each fetch has a fixed cost,
so this is the slower way to read a catalog. `--rosetta.shuffle=block`
streams instead: it fetches contiguous blocks of samples and mixes them
through a buffer of `--rosetta.shuffle_buffer` samples (256 by default).
Contiguous blocks are far less work for the server, and throughput grows
with `--num_workers`. Batches are then correlated: about
128 / `shuffle_buffer` of each batch comes from one stretch of one episode.

Streaming has no eval split and no episode filter, so
`--rosetta.shuffle=block` refuses `--dataset.eval_split` above 0 and
`--dataset.episodes`, and `--resume` doesn't continue the data order where
the run stopped.

## How samples differ from a ported dataset

Samples hold what a ported dataset gives LeRobot: the same frames, the
windows the policy asks for, the episode's task, and the same episode and
frame numbering. Any LeRobot policy trains on them. A few things differ:

- Rerun's dataloader trims windows instead of padding them. With ACT's
  `--policy.chunk_size=100`, the last 99 ticks of each episode are never
  sample starts. With an observation history, the first ticks are skipped
  too. `<key>_is_pad` is always false.
- Every stream must use `align: {strategy: hold, timeline: receive}`.
- Each episode's task is the prompt the recorder stored in the recording. A
  contract with a `tasks` section is refused.
- `--rosetta.shuffle=block` runs in one training process only. Rerun builds
  its streaming sample index for a single rank. Reading by index across
  several processes hasn't been tested.
- Quantiles in the normalization stats are exact over every frame, while a
  ported dataset's are LeRobot's frame-weighted average of each episode's
  quantiles. A policy that normalizes with quantiles, such as π0.5,
  normalizes differently than it would on the port.

The `rerun_rosetta`
[README](https://github.com/iblnkn/rosetta/tree/main/data/rerun/rerun_rosetta)
lists the rest of its limits.

## Deploy the checkpoint

`rosetta-lerobot-train` copies the contract into each checkpoint as
`pretrained_model/rosetta_contract.yaml`. `policy_runner_node` reads it
there when you give it only the checkpoint:

```bash
ros2 launch rosetta_ros policy_runner_launch.py \
    pretrained_name_or_path:=outputs/train/my_policy/checkpoints/last/pretrained_model
```

`--policy.push_to_hub` uploads the model without that file. To deploy a
model from the Hub, pass `contract_path:=robot.yaml` as well. Without it,
configure fails with a warning that the model was trained from a frame reader
and to pass `contract_path`.
