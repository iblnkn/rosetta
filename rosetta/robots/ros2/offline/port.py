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
Rosbag to dataset porter.

Reads rosbag2 recordings, decodes and resamples them through the contract using
the same ``StreamBuffer`` used live (so train and inference match), then feeds
the frame dicts to a :class:`~rosetta.policies.DatasetWriter` chosen
by ``--framework``.

Usage:
    # LeRobot dataset (default framework)
    python -m rosetta.robots.ros2.offline.port \
        --raw-dir ./datasets/bags --contract contract.yaml \
        --repo-id my_org/my_dataset --root ./datasets/lerobot

    # vla_foundry tar shards
    python -m rosetta.robots.ros2.offline.port --framework vla_foundry \
        --raw-dir ./datasets/bags --contract contract.yaml \
        --repo-id my_dataset --root ./datasets/vla
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

from rosetta.contract.schema import load_contract
from rosetta.contract.specs import iter_observation_specs, iter_specs
from rosetta.policies import DATASET_WRITER_GROUP, available_frameworks, load_dataset_writer
from rosetta.robots.ros2.offline.bag_frames import BagFrameSource, iter_bag_frames


def port(
    raw_dir: Path,
    repo_id: str,
    contract_path: Path,
    *,
    framework: str = "lerobot",
    root: Path | None = None,
    num_shards: int | None = None,
    shard_index: int | None = None,
    writer_opts: dict[str, Any] | None = None,
) -> None:
    """Port rosbag2 recordings to a policy-framework dataset."""
    contract = load_contract(contract_path)
    specs = list(iter_specs(contract))
    # Only observation streams gate warmup: actions and extended sections
    # (rewards/signals/info) may legitimately start late or publish sparsely.
    warmup_keys = {s.key for s in iter_observation_specs(contract)}
    # Per-frame task labels: task topics win over the recorded prompt.
    task_topics = {t.channel.topic: t.channel.type for t in contract.tasks}

    source = BagFrameSource(
        raw_dir,
        specs,
        num_shards=num_shards,
        shard_index=shard_index,
        warmup_keys=warmup_keys,
        task_topics=task_topics,
    )
    bag_dirs = source.bag_dirs()
    if not bag_dirs:
        logging.warning("No bags to process in this shard")
        return

    writer = load_dataset_writer(framework)
    writer.open(
        contract=contract,
        specs=specs,
        repo_id=repo_id,
        root=root,
        **(writer_opts or {}),
    )

    start_time = time.time()
    num_episodes = len(bag_dirs)
    successful = 0
    failed: list[tuple[Path, str]] = []

    for episode_index, bag_dir in enumerate(bag_dirs):
        logging.info("%d / %d episodes processed", episode_index, num_episodes)
        try:
            frame_count = 0
            for frame in iter_bag_frames(bag_dir, specs, warmup_keys=warmup_keys):
                writer.add_frame(frame)
                frame_count += 1
            writer.save_episode()
            successful += 1
            logging.info("  -> %d frames from %s", frame_count, bag_dir.name)
        except Exception as e:
            failed.append((bag_dir, str(e)))
            logging.error("  -> FAILED %s: %s", bag_dir.name, e)
            # Drop any frames already buffered for this episode so they
            # don't leak into the next successful one.
            writer.discard_episode()
            continue

    elapsed = time.time() - start_time
    logging.info(
        "Completed: %d/%d episodes (%d failed) in %.1fs",
        successful,
        num_episodes,
        len(failed),
        elapsed,
    )
    for bag_dir, error in failed:
        logging.warning("  - %s: %s", bag_dir.name, error)

    if successful == 0:
        raise RuntimeError(f"All {num_episodes} bags failed to convert")

    writer.finalize()


def main():
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Port ROS2 bags to a policy-framework dataset")
    parser.add_argument(
        "--framework",
        default="lerobot",
        help="Policy framework to write for (default: lerobot). Installed: "
        + (", ".join(available_frameworks(DATASET_WRITER_GROUP)) or "(none)"),
    )
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory of bag subdirs")
    parser.add_argument("--repo-id", type=str, default=None, help="Dataset repo ID (default: raw-dir name)")
    parser.add_argument("--contract", type=Path, required=True, help="Contract YAML path")
    parser.add_argument("--root", type=Path, default=None, help="Output parent directory")
    parser.add_argument("--num-shards", type=int, default=None, help="Total shards (parallel port)")
    parser.add_argument("--shard-index", type=int, default=None, help="This shard index")

    # Common and framework-specific writer options. Writers ignore options they
    # don't use.
    parser.add_argument("--push-to-hub", action="store_true", help="[lerobot] Upload to HF Hub")
    parser.add_argument("--vcodec", type=str, default="libsvtav1", help="[lerobot] Video codec")
    parser.add_argument("--past-steps", type=int, default=None, help="[vla_foundry] past lowdim steps")
    parser.add_argument("--future-steps", type=int, default=None, help="[vla_foundry] future lowdim steps")
    parser.add_argument("--image-indices", type=str, default=None, help='[vla_foundry] comma list, e.g. "-1,0"')
    parser.add_argument("--samples-per-shard", type=int, default=None, help="[vla_foundry] samples per tar shard")

    args = parser.parse_args()
    repo_id = args.repo_id or args.raw_dir.name

    writer_opts: dict[str, Any] = {
        "push_to_hub": args.push_to_hub,
        "vcodec": args.vcodec,
    }
    if args.past_steps is not None:
        writer_opts["past_lowdim_steps"] = args.past_steps
    if args.future_steps is not None:
        writer_opts["future_lowdim_steps"] = args.future_steps
    if args.image_indices is not None:
        writer_opts["image_indices"] = [int(x) for x in args.image_indices.split(",")]
    if args.samples_per_shard is not None:
        writer_opts["samples_per_shard"] = args.samples_per_shard

    try:
        port(
            raw_dir=args.raw_dir,
            repo_id=repo_id,
            contract_path=args.contract,
            framework=args.framework,
            root=args.root,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            writer_opts=writer_opts,
        )
    except KeyboardInterrupt:
        logging.info("\nInterrupted by user")
    except Exception as e:
        logging.error("Error: %s", e)
        raise


if __name__ == "__main__":
    main()
