# Train on several backends

Train one policy on episodes from more than one backend, such as Gazebo and
MuJoCo, or a simulator and the real arm. Record every backend with the same
contract, then port all the bags into one dataset.

You need a contract every backend can satisfy (see
[Write a contract](write-a-contract.md)) and a way to drive each backend
while you record (see [Record episodes](record-episodes.md)).

## Record each backend with one contract

Run the recorder with the same `contract_path` on every backend, and give each
backend its own `bag_base_dir`:

```bash
ros2 launch rosetta_ros episode_recorder_launch.py \
    contract_path:=robot.yaml \
    bag_base_dir:=datasets/two_backends/gazebo \
    use_sim_time:=true
```

Change `bag_base_dir` to `datasets/two_backends/mujoco` for the next
backend. Drop `use_sim_time:=true` on the real arm. Keep these directories
for this dataset only: the port takes every bag under them.

The contract reads the same topics on every backend, so each backend has to
publish them:

- Topic names and message types.
- The joint names each `select` reads.

An observation stream a backend doesn't match fails that backend's episodes
when you port them. An action stream that doesn't match (a missing topic, a
wrong type or wrong joint names) ports with zeros and only a warning, so the
`Completed` line still counts the episode as ported. Read the porter's
warnings for the action key before you train.

Image resolutions can differ. Every image stream's `apply` needs an operator
that sets a fixed size, which gives every backend's images the size the
contract declares. The built-in one is `resize`, and it stretches a camera
whose aspect ratio differs from that size.

Rates can differ too. A stream aligned with `strategy: hold` repeats its last
sample, so a camera slower than the contract's `fps` repeats each image over
several ticks. With `asof` or `drop`, a sample older than the strategy
allows is not used, and after warmup the tick is zero-filled: a black image
for a camera, zeros for a numeric stream (see
[`align`](../reference/contract.md#align)).

## Port into one dataset

Point `--raw-dir` at the directory above the backends' directories:

```bash
ros2 run rosetta_ros rosetta_port \
    --raw-dir datasets/two_backends \
    --contract robot.yaml \
    --repo-id my_org/two_backends \
    --root datasets/lerobot
```

`--raw-dir` is searched recursively, so every bag under `gazebo/` and
`mujoco/` becomes an episode. Episodes are numbered in sorted path order, so
the Gazebo episodes come first. A bag that fails to port gets no number, and
the bags after it move up one. With three Gazebo bags and two MuJoCo bags,
the port ends with:

```text
Completed: 5/5 episodes (0 failed) in 57.1s
```

The dataset has one set of features for both backends, the ones the contract
declares.

## Train

Train on the dataset as on any ported dataset (see
[Train a policy](train.md)). To train on some backends only, pass their
episode numbers to `--dataset.episodes`. For the port in
[Port into one dataset](#port-into-one-dataset),
`--dataset.episodes='[3,4]'` trains on the two MuJoCo episodes. Check that
the port's `Completed` line shows no failures before you count.
