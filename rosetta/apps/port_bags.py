#!/usr/bin/env python3
# Copyright 2025 Isaac Blankenau
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
ROS2 bag → LeRobot dataset porting (composition root).

Wires the ROS2 bag reader (:mod:`rosetta.ros2.bag_reader`) to the LeRobot
dataset writer (:mod:`rosetta.lerobot.dataset_writer`). Decoding + resampling
match live inference for train/inference parity.

Usage:
    # Port all bags
    python -m rosetta.apps.port_bags \\
        --raw-dir /path/to/bags \\
        --repo-id my_dataset \\
        --contract /path/to/contract.yaml

    # Port a single shard (for SLURM parallel processing)
    python -m rosetta.apps.port_bags \\
        --raw-dir /path/to/bags \\
        --repo-id my_dataset \\
        --contract /path/to/contract.yaml \\
        --num-shards 100 \\
        --shard-index 0

    # Push to HuggingFace Hub
    python -m rosetta.apps.port_bags \\
        --raw-dir /path/to/bags \\
        --repo-id my_org/my_dataset \\
        --contract /path/to/contract.yaml \\
        --push-to-hub
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time

from lerobot.utils.utils import get_elapsed_time_in_days_hours_minutes_seconds

from rosetta.core.contract import load_contract
from rosetta.core.contract_utils import iter_specs
from rosetta.lerobot.dataset_writer import build_features, create_dataset
from rosetta.ros2.bag_reader import find_bag_dirs, stream_frames_from_bag


# ---------- Main porting function ----------


def port_bags(
    raw_dir: Path,
    repo_id: str,
    contract_path: Path,
    root: Path | None = None,
    push_to_hub: bool = False,
    num_shards: int | None = None,
    shard_index: int | None = None,
    vcodec: str = 'libsvtav1',
):
    """
    Port ROS2 bags to LeRobot dataset format.

    Args:
    ----
    raw_dir: Directory containing bag subdirectories.
    repo_id: HuggingFace repository ID (e.g., "my_org/my_dataset").
    contract_path: Path to Rosetta contract YAML.
    root: Output directory for dataset.
    push_to_hub: Whether to upload to HuggingFace Hub after porting.
    num_shards: Total number of shards for parallel processing.
    shard_index: Index of this shard (0 to num_shards-1).
    vcodec: Video codec for encoding (libsvtav1, libx264, hevc, etc.).

    """
    contract = load_contract(contract_path)
    specs = list(iter_specs(contract))
    features = build_features(specs)

    all_bag_dirs = find_bag_dirs(raw_dir)
    total_bags = len(all_bag_dirs)
    logging.info('Found %d bags in %s', total_bags, raw_dir)

    # Select shard subset if sharding
    if num_shards is not None:
        if shard_index is None:
            raise ValueError('shard_index required when num_shards is specified')
        if shard_index >= num_shards:
            raise ValueError(f'shard_index ({shard_index}) >= num_shards ({num_shards})')

        bag_dirs = all_bag_dirs[shard_index::num_shards]
        logging.info('Shard %d/%d: processing %d bags', shard_index, num_shards, len(bag_dirs))
    else:
        bag_dirs = all_bag_dirs

    if not bag_dirs:
        logging.warning('No bags to process in this shard')
        return

    # LeRobot uses root directly as dataset path, so append repo_id
    dataset_root = root / repo_id if root else None
    dataset = create_dataset(
        repo_id=repo_id,
        root=dataset_root,
        robot_type=contract.robot_type,
        fps=contract.fps,
        features=features,
        vcodec=vcodec,
    )

    start_time = time.time()
    num_episodes = len(bag_dirs)
    successful = 0
    failed: list[tuple[Path, str]] = []

    for episode_index, bag_dir in enumerate(bag_dirs):
        elapsed_time = time.time() - start_time
        d, h, m, s = get_elapsed_time_in_days_hours_minutes_seconds(elapsed_time)

        logging.info(
            f'{episode_index} / {num_episodes} episodes processed '
            f'(after {d} days, {h} hours, {m} minutes, {s:.3f} seconds)'
        )

        try:
            frame_count = 0
            for frame in stream_frames_from_bag(bag_dir, specs):
                dataset.add_frame(frame)
                frame_count += 1

            dataset.save_episode()
            successful += 1
            logging.info('  -> %d frames from %s', frame_count, bag_dir.name)

        except Exception as e:
            failed.append((bag_dir, str(e)))
            logging.error('  -> FAILED %s: %s', bag_dir.name, e)
            continue

    elapsed_time = time.time() - start_time
    d, h, m, s = get_elapsed_time_in_days_hours_minutes_seconds(elapsed_time)
    logging.info(
        f'\nCompleted: {successful}/{num_episodes} episodes '
        f'({len(failed)} failed) in {d}d {h}h {m}m {s:.1f}s'
    )

    if failed:
        logging.warning('Failed bags:')
        for bag_dir, error in failed:
            logging.warning('  - %s: %s', bag_dir.name, error)

    if successful == 0:
        raise RuntimeError(f'All {num_episodes} bags failed to convert')

    dataset.finalize()

    if push_to_hub:
        dataset.push_to_hub(
            tags=['rosetta', 'rosbag'],
            private=False,
        )


# ---------- CLI ----------


def main():
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Port ROS2 bags to LeRobot dataset')

    parser.add_argument(
        '--raw-dir', type=Path, required=True, help='Directory containing bag subdirectories'
    )
    parser.add_argument(
        '--repo-id',
        type=str,
        default=None,
        help='HuggingFace repository ID (e.g., my_org/my_dataset). Defaults to raw-dir name.',
    )
    parser.add_argument('--contract', type=Path, required=True, help='Rosetta contract YAML path')
    parser.add_argument(
        '--root',
        type=Path,
        default=None,
        help=(
            'Parent directory for datasets. Dataset saved to '
            'root/repo-id. (default: ~/.cache/huggingface/lerobot)'
        ),
    )
    parser.add_argument(
        '--push-to-hub', action='store_true', help='Upload to HuggingFace Hub after porting'
    )
    parser.add_argument(
        '--num-shards',
        type=int,
        default=None,
        help='Total number of shards for parallel processing',
    )
    parser.add_argument(
        '--shard-index', type=int, default=None, help='Index of this shard (0 to num-shards-1)'
    )
    parser.add_argument(
        '--vcodec',
        type=str,
        default='libsvtav1',
        choices=['libsvtav1', 'libx264', 'h264', 'hevc', 'h264_nvenc'],
        help=(
            'Video codec for encoding (default: libsvtav1). '
            'Use libx264/h264 for faster encoding.'
        ),
    )

    args = parser.parse_args()

    repo_id = args.repo_id or args.raw_dir.name

    try:
        port_bags(
            raw_dir=args.raw_dir,
            repo_id=repo_id,
            contract_path=args.contract,
            root=args.root,
            push_to_hub=args.push_to_hub,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            vcodec=args.vcodec,
        )
    except KeyboardInterrupt:
        logging.info('\nInterrupted by user')
    except Exception as e:
        logging.error('Error: %s', e)
        raise


if __name__ == '__main__':
    main()
