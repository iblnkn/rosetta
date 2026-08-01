# Design

Rosetta is built around one philosophy: the translation between a robot
and a policy is defined once and enforced as late as possible. Defined
once, so training and deployment cannot drift apart. Enforced late, so
recordings stay reusable as models change. This page explains both
halves, then shows how the packages divide the work.

## Record everything, decide later

Your robot's data outlives any single model. Demonstrations recorded today
can retrain next year's architecture, but only if the recordings still
contain what that architecture wants. Preprocessing at record time throws
that option away. So Rosetta records as much as possible, as raw as
possible, and defers every decision about what a model sees.

Rosetta records to [rosbag2](https://github.com/ros2/rosbag2) files and
converts them to training datasets in a separate step. Bags store every
message at its original rate and timestamp, with no alignment,
downsampling, or lossy transformation, and they hold topics that map to no
training feature: diagnostics, TF trees, debug streams, extra sensors.
They are the standard recording format in ROS 2, so playback and
inspection tooling already exists. They are also cheap to write live,
where training formats built on Parquet and MP4 are read-optimized and
need in-memory buffering and post-episode video encoding. Rosetta defaults
to [MCAP](https://mcap.dev/) storage, which adds random-access reads and
compression.

Any tool that writes bags works, and recordings made before or without
Rosetta convert the same way. On top of plain capture, the episode recorder
Rosetta provides adds episode scoping through a ROS 2 action, provenance
(the operator prompt and the contract text embedded in each bag's
metadata), and a capture audit. It records every topic on the graph by
default, with regex include and exclude lists for the exceptions, and at
episode end it reports per-topic message counts and flags contract topics
that received nothing. At record time the contract acts as a manifest
rather than a filter. It never narrows what is recorded; it checks that
everything you declared you will train on is arriving. The audit is only a
log summary; there is no mid-episode warning yet and no rate check against
the contract's `fps`.

Revising a contract never requires re-recording. Conversion reads the
contract you pass it, treats the copy embedded in a bag as provenance only,
and warns on mismatch. The same bags can produce different datasets as your
schema evolves.

## The translation problem

Raw data is not what a model eats. A robot speaks topics: typed messages,
named fields, many rates, many timelines, most of it irrelevant to any one
policy. A policy speaks frames: one flat value per key per tick, fixed
shape, fixed rate, meaning carried by position. Getting from one to the
other means deciding how asynchronous streams land on the frame clock,
which fields appear in which order, and how values map between robot units
and the ranges a policy expects.

These decisions must hold twice: once when recorded data becomes a training
dataset, and again live, when observations feed the model and its output
becomes messages. Write them twice and nothing checks the two agree. That
failure mode is train/serve skew, and it is quiet: the policy produces
plausible-looking actions, performs worse than it should, and no error
points at the cause.

## The contract

Rosetta puts the translation in one place. One YAML file per robot, the
contract, declares the frames and the streams behind them, in both
directions. Bag conversion and live inference run the same code from the
same contract, so there is no second implementation to drift. Bags and
datasets embed the contract text, and a checkpoint resolves its own
translation at deploy time through the dataset it was trained on.

The contract's territory ends at the dataset boundary: frames in robot
units, named and shaped exactly as the dataset stores them. Model-side
preparation (normalization from dataset statistics, batching, tokenization)
belongs to the framework and travels with the checkpoint; LeRobot saves its
pre- and post-processors next to the weights. The two mechanisms mirror
each other. Rosetta keeps the robot side of the boundary identical between
training and deployment, and the checkpoint's processors keep the model
side identical.

The [contract reference](../reference/contract.md) documents the schema;
[write a contract](../how-to/write-a-contract.md) builds one up from
scratch.

## LeRobot

[LeRobot](https://github.com/huggingface/lerobot) is Hugging Face's
open-source framework for
[robot learning](https://huggingface.co/spaces/lerobot/robot-learning-tutorial).
It provides tools for recording demonstrations, training policies (ACT,
Diffusion Policy, VLAs like SmolVLA and Pi0), and deploying them on
hardware. LeRobot defines a standard dataset format (v3) built on Parquet
files and MP4 videos, with community datasets and models shared on the
[Hugging Face Hub](https://huggingface.co/datasets?other=LeRobot).

## Architecture

| Package | Purpose |
|---------|---------|
| `rosetta` | Core library, nodes, bag conversion |
| [`rosetta_interfaces`](https://github.com/iblnkn/rosetta_interfaces) | ROS 2 action and service definitions |
| [`lerobot_rosetta`](https://github.com/iblnkn/lerobot-rosetta) | LeRobot framework adapter: dataset writer, policy runner, inference servers |
| [`lerobot_robot_rosetta`](https://github.com/iblnkn/lerobot-robot-rosetta) | LeRobot Robot plugin |
| [`lerobot_teleoperator_rosetta`](https://github.com/iblnkn/lerobot-teleoperator-rosetta) | LeRobot Teleoperator plugin (experimental) |

No framework appears in a contract. Framework adapters register through
Python entry points (`rosetta.dataset_writers`, `rosetta.policy_runners`),
and `rosetta_port --framework` or the policy runner's `framework` parameter
selects one by name. `lerobot_rosetta` registers both under the name
`lerobot`.

The `lerobot_robot_rosetta` and `lerobot_teleoperator_rosetta` packages
implement LeRobot's [Robot](https://huggingface.co/docs/lerobot/integrate_hardware)
and Teleoperator interfaces, following LeRobot's plugin naming convention
(`lerobot_robot_*`, `lerobot_teleoperator_*`) for auto-discovery.

A typical LeRobot robot (like `so101_follower`) talks to hardware directly:
motors over serial, cameras over USB, the `Robot` class is the driver. A
Rosetta robot is a ROS 2 lifecycle node: observations arrive on topics,
actions leave on topics, and the drivers live elsewhere in the graph. Any
ROS 2 robot can use LeRobot's native tools this way: define a contract and
pass `--robot.type=rosetta`. Because the plugin creates a ROS 2 node
internally, ROS 2 must be installed even when you start it through
LeRobot's CLI. LeRobot's `connect()` maps to the lifecycle `activate`
transition, and `disconnect()` to `deactivate` then `cleanup`, which
publishes the declared safety action on the way down.
