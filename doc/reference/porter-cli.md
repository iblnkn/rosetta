# Porter CLI

`rosetta_port` (equivalently `python -m rosetta.robots.ros2.offline.port`) converts rosbag2 files to training datasets using the contract for key mapping, timestamp alignment, resampling, and dtype conversion. The porter applies the same `StreamBuffer` resampling logic as live inference, so your offline dataset matches what the robot sees at runtime.

## Usage

```bash
rosetta_port \
    --raw-dir ./datasets/bags \
    --contract ./contract.yaml \
    --repo-id my_dataset \
    --root ./datasets/lerobot
```

## Arguments

| Argument | Description |
|----------|-------------|
| `--raw-dir` | **(Required)** Directory of bags, searched recursively |
| `--contract` | **(Required)** Rosetta contract YAML defining the topic → feature mapping |
| `--framework` | Dataset writer to use: `lerobot` (default). Other writers resolve by entry-point name |
| `--root` | Override output directory (LeRobot defaults to `~/.cache/huggingface/lerobot`) |
| `--repo-id` | Dataset repo ID. Defaults to the `--raw-dir` directory name |
| `--vcodec` | Video codec selection (default `libsvtav1`, not in base LeRobot porters) |
| `--num-shards` / `--shard-index` | Split a directory of bags across parallel porter invocations |
| `--no-embed-contract` | Skip writing the contract into the dataset as `meta/rosetta_contract.yaml` |
| `--push-to-hub` | Push the dataset to the Hugging Face Hub (`lerobot`) |
| `--hub-public` / `--hub-tags` | With `--push-to-hub`: make the repo public (private by default) / set tags (default `rosetta,rosbag`) |
| `--past-steps`, `--future-steps`, `--image-indices`, `--samples-per-shard` | Writer-specific windowing and sharding options |

## Contract embedding

The porter embeds the contract into the dataset (`meta/rosetta_contract.yaml`) by default, so the dataset carries the exact translation behind the data. Inference resolves the contract from there when no `contract_path` is given (see [Nodes: contract resolution](nodes.md#contract-resolution)). When a bag carries an embedded contract (recorder default), the porter compares the bag's contract against `--contract` and warns on a semantic mismatch. `--contract` stays authoritative.

## Relationship to LeRobot

The porter mirrors the interface of LeRobot's example porters (like `port_droid.py`):

```bash
# LeRobot's port_droid.py
python examples/port_datasets/port_droid.py \
    --raw-dir /data/droid/1.0.1 \
    --repo-id my_org/droid \
    --push-to-hub

# Rosetta's porter (same pattern + contract)
rosetta_port \
    --raw-dir ./datasets/bags \
    --contract contract.yaml \
    --repo-id my_org/my_dataset \
    --root ./datasets/lerobot
```

For large-scale conversions, parallel processing, and SLURM cluster workflows, see the [LeRobot Porting Datasets Guide](https://huggingface.co/docs/lerobot/en/porting_datasets_v3) and substitute `rosetta_port` for `port_droid.py` in the examples.
