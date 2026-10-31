# lerobot_dataset_rosetta

Rosetta's dataset writer for [LeRobot](https://github.com/huggingface/lerobot).
It turns contract frames into a `LeRobotDataset` (Parquet and MP4), one
episode at a time. `rosetta_port` loads it by the name `lerobot`, which is
the default.

The writer copies the contract into the dataset as
`meta/rosetta_contract.yaml`, so a policy trained on the dataset can find its
contract at deployment.

This package is the data half of Rosetta's LeRobot support. The policy runner
and inference servers are in
[`lerobot_rosetta`](../../../policies/lerobot/lerobot_rosetta). The package name
matches none of LeRobot's plugin discovery prefixes, so LeRobot never imports
it.

## Entry points

| Group | Name | Target |
|---|---|---|
| `rosetta.dataset_writers` | `lerobot` | `lerobot_dataset_rosetta.dataset_writer:LeRobotDatasetWriter` |

## Install

LeRobot has no rosdep key, so `rosdep install` neither installs it nor
reports it missing. Install it with pip first. The package is not released to
the ROS buildfarm.

```bash
pip install 'lerobot[dataset]<0.6.2'
colcon build --packages-up-to lerobot_dataset_rosetta
```

## Documentation

- [Prepare a dataset](https://iblnkn.github.io/rosetta/how-to/prepare-a-dataset.html)
- [rosetta_port](https://iblnkn.github.io/rosetta/reference/rosetta-port.html)
- Full Rosetta documentation: [iblnkn.github.io/rosetta](https://iblnkn.github.io/rosetta/)

## License

Apache-2.0
