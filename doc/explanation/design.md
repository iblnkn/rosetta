# Design

## What is LeRobot?

[LeRobot](https://github.com/huggingface/lerobot) is Hugging Face's open-source
framework for [robot learning](https://huggingface.co/spaces/lerobot/robot-learning-tutorial).
It provides tools for recording demonstrations, training policies (ACT,
Diffusion Policy, VLAs like SmolVLA and Pi0), and deploying them on hardware.
LeRobot defines a standard dataset format (v3) built on Parquet files and MP4
videos, with a growing ecosystem of community-contributed datasets and models on
the [Hugging Face Hub](https://huggingface.co/datasets?other=LeRobot).

## What is Rosetta?

Every robot-learning pipeline translates robot messages into training frames.
Timing, units, field order, image geometry. Write the translation twice, once for
your dataset script and again for your deployment node, and nothing checks the
two agree. Your policy trains on one translation and runs on another.

Rosetta puts the translation in one place: the contract. One YAML per robot
defines the frames and the streams behind them, in both directions. Bag
conversion and live inference run the same code from the same contract, so there
is no second implementation to drift. Bags and datasets embed the contract text,
so a checkpoint resolves its own translation at deploy time.

## Architecture

| Package | Purpose |
|---------|---------|
| `rosetta` | Core library, nodes, bag conversion |
| [`rosetta_interfaces`](https://github.com/iblnkn/rosetta_interfaces) | ROS2 action/service definitions |
| [`lerobot_rosetta`](https://github.com/iblnkn/lerobot-rosetta) | LeRobot backend adapter: dataset writer, policy runner, inference servers |
| [`lerobot_robot_rosetta`](https://github.com/iblnkn/lerobot-robot-rosetta) | LeRobot Robot plugin |
| [`lerobot_teleoperator_rosetta`](https://github.com/iblnkn/lerobot-teleoperator-rosetta) | LeRobot Teleoperator plugin (experimental) |

The `lerobot_robot_rosetta` and `lerobot_teleoperator_rosetta` packages
implement LeRobot's [Robot](https://huggingface.co/docs/lerobot/integrate_hardware)
and Teleoperator interfaces, and follow LeRobot's plugin naming conventions
(`lerobot_robot_*`, `lerobot_teleoperator_*`) for auto-discovery when installed.

**Typical LeRobot robots** (like `so101_follower`) communicate directly with
hardware: motors via serial/CAN, cameras via USB, and the `Robot` class IS the
hardware interface. **Rosetta robots** are ROS2 lifecycle nodes: they subscribe
to topics for observations, publish to topics for actions, and the hardware
drivers exist elsewhere in the ROS2 graph. Any ROS2 robot can use LeRobot's
tools this way. Define a contract and use `--robot.type=rosetta`.

**Important:** because `lerobot_robot_rosetta` creates a ROS2 lifecycle node
internally, **your system needs ROS2 installed** to use it, even when invoking it
through LeRobot's standard CLI tools.

LeRobot's `connect()` maps to the lifecycle `activate` transition and
`disconnect()` to `deactivate` then `cleanup`, which publishes the declared
safety action on the way down.

## Why record to bag files?

Rosetta records demonstrations to [rosbag2](https://github.com/ros2/rosbag2)
files first, then converts them to LeRobot datasets in a separate step. This is a
deliberate design choice with several benefits:

- **Preserves raw data.** Bag files store every message at its original rate and
  timestamp, with no alignment, downsampling, or lossy transformation. You can
  reprocess the same recordings later with a different contract without
  re-recording.
- **Familiar to ROS2 users.** Bag files are the standard data format in the ROS2
  ecosystem, with mature tooling for recording, playback, inspection, and
  analysis.
- **Stores data beyond what LeRobot needs.** Bags can include topics that map to
  no LeRobot feature: diagnostics, TF trees, debug streams, extra sensors.
- **Leverages MCAP.** Rosetta defaults to [MCAP](https://mcap.dev/) storage,
  which provides high-performance random-access reads, efficient compression,
  and broad ecosystem support beyond ROS2.
- **Write-optimized for live recording.** LeRobot datasets (Parquet + MP4) are
  read-optimized for training and involve more overhead when writing live,
  including in-memory buffering and post-episode video encoding.
