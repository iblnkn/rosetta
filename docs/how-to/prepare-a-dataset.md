# Prepare a dataset

Port recorded bags into a LeRobot dataset with `rosetta_port`. Every option
is on the [rosetta_port](../reference/rosetta-port.md) page.

You need a directory of bags, one bag directory per episode (the layout
[Record episodes](record-episodes.md) produces), and a contract that loads
(see [Write a contract](write-a-contract.md)).

## Port

```bash
ros2 run rosetta_ros rosetta_port \
    --raw-dir datasets/bags \
    --contract robot.yaml \
    --repo-id my_org/my_dataset \
    --root datasets/lerobot
```

`--raw-dir` is searched recursively, and every directory with a
`metadata.yaml` in it becomes one episode, in sorted order. `--repo-id`
defaults to the name of `--raw-dir`. The dataset lands in `--root` joined with
`--repo-id`, here `datasets/lerobot/my_org/my_dataset`. That directory must
not exist yet, or the port fails with `FileExistsError`.

The porter runs the same alignment code as the live runner, so a dataset
frame is what the policy would see at runtime. It skips the ticks before every
observation stream has a sample, and the LeRobot runner waits up to 5 s for
the same condition before its first frame. The last log line counts the episodes:

```text
Completed: 4/4 episodes (0 failed) in 0.6s
```

`--contract` decides how every bag is decoded. If a bag was recorded with a
different contract, the porter logs a warning and uses `--contract` anyway.

## Check the result

```bash
ls datasets/lerobot/my_org/my_dataset/meta/
```

```text
episodes  info.json  rosetta_contract.yaml  stats.json  tasks.parquet
```

`info.json` lists the features with their dtypes, shapes and names.
`rosetta_contract.yaml` is a copy of the contract the porter used. Leave it
out with `--no-embed-contract`. To print the features:

```bash
python3 -c "
import json
info = json.load(open('datasets/lerobot/my_org/my_dataset/meta/info.json'))
for key, f in info['features'].items():
    print(key, f['dtype'], f['shape'])
"
```

For a contract with only `observation.state` and `action`, both from six
joints, it prints:

```text
observation.state float64 [6]
action float64 [6]
is_first bool [1]
is_last bool [1]
is_terminal bool [1]
timestamp float32 [1]
frame_index int64 [1]
episode_index int64 [1]
index int64 [1]
task_index int64 [1]
```

## Revise and port again

Edit the contract and run the same command with a new `--repo-id`, or delete
the old dataset directory first. A new key, a different `fps`, a different
alignment or a different image size doesn't need a new recording, as long as
the topics the contract reads are in the bags. By default the recorder keeps
every topic except `/rosout` and `/parameter_events`, and one stream per
camera.

## Port in parallel

Split the bags into shards with `--num-shards` and `--shard-index`. Shard `i`
takes every `n`th bag starting at bag `i`. Each shard has to write its own
dataset, because two processes can't create the same directory. Merge them
afterwards with LeRobot's `lerobot-edit-dataset`. Use no more shards than
bags: a shard with no bags writes no dataset, and the merge then can't find
it.

1. Port each shard to its own repo ID:

   ```bash
   for i in 0 1 2 3; do
     ros2 run rosetta_ros rosetta_port --raw-dir datasets/bags --contract robot.yaml \
         --repo-id my_org/my_dataset_shard_$i --root datasets/shards \
         --num-shards 4 --shard-index $i &
   done
   wait
   ```

2. Merge the shards:

   ```bash
   lerobot-edit-dataset \
       --new_repo_id my_org/my_dataset \
       --new_root datasets/lerobot/my_org/my_dataset \
       --operation.type merge \
       --operation.repo_ids "[my_org/my_dataset_shard_0, my_org/my_dataset_shard_1, my_org/my_dataset_shard_2, my_org/my_dataset_shard_3]" \
       --operation.roots "[datasets/shards/my_org/my_dataset_shard_0, datasets/shards/my_org/my_dataset_shard_1, datasets/shards/my_org/my_dataset_shard_2, datasets/shards/my_org/my_dataset_shard_3]"
   ```

   It ends with `Episodes: <n>, Frames: <n>` for the merged dataset.

3. Copy the contract into the merged dataset. The merge doesn't carry
   `rosetta_contract.yaml` over.

   ```bash
   cp datasets/shards/my_org/my_dataset_shard_0/meta/rosetta_contract.yaml \
       datasets/lerobot/my_org/my_dataset/meta/
   ```

## Push to the Hugging Face Hub

Log in with `hf auth login`, then add `--push-to-hub` with a repo ID under
your Hub user or organization:

```bash
ros2 run rosetta_ros rosetta_port \
    --raw-dir datasets/bags \
    --contract robot.yaml \
    --repo-id <hf_user>/<name> \
    --root datasets/lerobot \
    --push-to-hub
```

The upload runs after every episode is written. It's private unless you add
`--hub-public`. `--hub-tags` replaces the default tags `rosetta,rosbag`.

## If an episode fails

The porter logs `-> FAILED <bag>: <error>` for that bag, drops its frames and
moves on. These fail an episode:

- An observation topic missing from the bag.
- An observation topic that never produces a sample, which leaves the episode
  with no frames.
- A contract type with no decoder.

A missing action, reward or signal topic doesn't fail the episode. The key is
zero-filled and the porter logs `Topic <topic> not in bag, <key> will be
zero-filled`. Check the bag with `ros2 bag info <bag_dir>` against the topic
names in the contract.

If every episode fails, the port exits non-zero with
`RuntimeError: All <n> bags failed to convert`. The dataset directory it
created is left behind, so delete it before you port again.
