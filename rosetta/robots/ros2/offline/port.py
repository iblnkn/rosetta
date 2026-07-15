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
from typing import TYPE_CHECKING, Any, Iterable

from rosetta.contract.schema import load_contract
from rosetta.contract.sidecar import LEROBOT_CONTRACT_SIDECAR_PATH
from rosetta.contract.specs import iter_observation_specs, iter_specs
from rosetta.policies import available_dataset_writers, load_dataset_writer
from rosetta.robots.ros2.bag_metadata import contract_hash
from rosetta.robots.ros2.offline.bag_frames import find_bag_dirs, iter_bag_frames, read_bag_contract_hash

if TYPE_CHECKING:
    from rosetta.contract.schema import Contract
    from rosetta.policies import DatasetWriter


def warn_if_contract_mismatch(contract_path: Path, bag_hash: str) -> None:
    """Heads-up when a bag was recorded with a different contract than --contract.

    --contract is always the source of truth for decoding; a mismatch only
    warns, and this check itself must never be why a port fails — an
    unreadable --contract here (it already loaded fine upstream) or a bag
    with no recorded hash silently skips the comparison.
    """
    try:
        local_hash = contract_hash(contract_path)
    except OSError:
        return
    if bag_hash and bag_hash != local_hash:
        logging.warning(
            "Bag was recorded with a different contract (hash=%s) than "
            "--contract (hash=%s); using --contract for decoding and embedding.",
            bag_hash,
            local_hash,
        )


def write_dataset(
    writer: DatasetWriter,
    episodes: Iterable[tuple[str, Iterable[dict[str, Any]]]],
    *,
    contract: Contract,
    repo_id: str,
    root: Path | None = None,
    contract_path: Path | None = None,
    embed_contract: bool = True,
    writer_opts: dict[str, Any] | None = None,
) -> None:
    """Drive a DatasetWriter over ``(episode_name, frames)`` pairs.

    The orchestration core, separated from bag/entry-point wiring (see
    :func:`port`) so its logic tests against a plain fake writer: open once,
    pump each episode with per-episode failure isolation (a failed episode is
    discarded so buffered frames never leak into the next one), then
    finalize. Raises ``RuntimeError`` — without finalizing — when every
    episode failed.
    """
    writer.open(
        contract=contract,
        repo_id=repo_id,
        root=root,
        contract_path=contract_path,
        embed_contract=embed_contract,
        **(writer_opts or {}),
    )

    start_time = time.time()
    episodes = list(episodes)  # (name, frames) pairs are cheap; frame iterables stay lazy
    num_episodes = len(episodes)
    successful = 0
    failed: list[tuple[str, str]] = []

    for episode_index, (name, frames) in enumerate(episodes):
        logging.info("%d / %d episodes processed", episode_index, num_episodes)
        try:
            frame_count = 0
            for frame in frames:
                writer.add_frame(frame)
                frame_count += 1
            writer.save_episode()
            successful += 1
            logging.info("  -> %d frames from %s", frame_count, name)
        except Exception as e:
            failed.append((name, str(e)))
            logging.error("  -> FAILED %s: %s", name, e)
            # Drop any frames already buffered for this episode so they
            # don't leak into the next successful one, and close the
            # abandoned frame iterator so its open bag reader is released
            # now instead of at end-of-run (episodes stays referenced).
            writer.discard_episode()
            close = getattr(frames, "close", None)  # Iterable contract: close is optional
            if close is not None:
                close()
            continue

    elapsed = time.time() - start_time
    logging.info(
        "Completed: %d/%d episodes (%d failed) in %.1fs",
        successful,
        num_episodes,
        len(failed),
        elapsed,
    )
    for name, error in failed:
        logging.warning("  - %s: %s", name, error)

    if successful == 0:
        raise RuntimeError(f"All {num_episodes} bags failed to convert")

    writer.finalize()


def port(
    raw_dir: Path,
    repo_id: str,
    contract_path: Path,
    *,
    framework: str = "lerobot",
    root: Path | None = None,
    num_shards: int | None = None,
    shard_index: int | None = None,
    embed_contract: bool = True,
    writer_opts: dict[str, Any] | None = None,
) -> None:
    """Port rosbag2 recordings to a policy-framework dataset.

    The wiring shell: loads the contract, discovers bags, resolves the
    ``--framework`` writer from the entry-point registry, and delegates the
    actual writing to :func:`write_dataset`.
    """
    contract = load_contract(contract_path)
    specs = list(iter_specs(contract))
    # Only observation streams gate warmup: actions and extended sections
    # (rewards/signals/info) may legitimately start late or publish sparsely.
    warmup_keys = {s.key for s in iter_observation_specs(contract)}
    # Per-frame task labels: task topics win over the recorded prompt.
    task_topics = {t.channel.topic: t.channel.type for t in contract.tasks}

    bag_dirs = find_bag_dirs(raw_dir, num_shards=num_shards, shard_index=shard_index)
    if not bag_dirs:
        logging.warning("No bags to process in this shard")
        return

    # First bag as a representative sample: all bags in a run are normally
    # recorded by one recorder config, so one hash comparison suffices.
    warn_if_contract_mismatch(contract_path, read_bag_contract_hash(bag_dirs[0]))

    episodes = (
        (bag_dir.name, iter_bag_frames(bag_dir, specs, warmup_keys=warmup_keys, task_topics=task_topics))
        for bag_dir in bag_dirs
    )
    write_dataset(
        load_dataset_writer(framework),
        episodes,
        contract=contract,
        repo_id=repo_id,
        root=root,
        contract_path=Path(contract_path),
        embed_contract=embed_contract,
        writer_opts=writer_opts,
    )


def main():
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Port ROS2 bags to a policy-framework dataset")
    parser.add_argument(
        "--framework",
        default="lerobot",
        help="Policy framework to write for (default: lerobot). Installed: "
        + (", ".join(available_dataset_writers()) or "(none)"),
    )
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory of bag subdirs")
    parser.add_argument("--repo-id", type=str, default=None, help="Dataset repo ID (default: raw-dir name)")
    parser.add_argument("--contract", type=Path, required=True, help="Contract YAML path")
    parser.add_argument("--root", type=Path, default=None, help="Output parent directory")
    parser.add_argument("--num-shards", type=int, default=None, help="Total shards (parallel port)")
    parser.add_argument("--shard-index", type=int, default=None, help="This shard index")
    parser.add_argument(
        "--no-embed-contract",
        dest="embed_contract",
        action="store_false",
        default=True,
        help=f"Don't copy the contract into the dataset's {LEROBOT_CONTRACT_SIDECAR_PATH} sidecar",
    )

    # Common and framework-specific writer options. Writers ignore options they
    # don't use.
    parser.add_argument("--push-to-hub", action="store_true", help="[lerobot/starvla] Upload to HF Hub")
    parser.add_argument(
        "--hub-public",
        dest="hub_private",
        action="store_false",
        default=True,
        help="[lerobot/starvla] Push publicly instead of privately (default: private)",
    )
    parser.add_argument(
        "--hub-tags",
        type=str,
        default=None,
        help='[lerobot/starvla] Comma-separated HF Hub tags (default: "rosetta,rosbag")',
    )
    parser.add_argument("--vcodec", type=str, default="libsvtav1", help="[lerobot/starvla] Video codec")
    parser.add_argument("--past-steps", type=int, default=None, help="[vla_foundry] past lowdim steps")
    parser.add_argument("--future-steps", type=int, default=None, help="[vla_foundry] future lowdim steps")
    parser.add_argument("--image-indices", type=str, default=None, help='[vla_foundry] comma list, e.g. "-1,0"')
    parser.add_argument("--samples-per-shard", type=int, default=None, help="[vla_foundry] samples per tar shard")

    args = parser.parse_args()
    repo_id = args.repo_id or args.raw_dir.name

    writer_opts: dict[str, Any] = {
        "push_to_hub": args.push_to_hub,
        "hub_private": args.hub_private,
        "vcodec": args.vcodec,
    }
    if args.hub_tags is not None:
        writer_opts["hub_tags"] = [t.strip() for t in args.hub_tags.split(",") if t.strip()]
    if args.past_steps is not None:
        writer_opts["past_lowdim_steps"] = args.past_steps
    if args.future_steps is not None:
        writer_opts["future_lowdim_steps"] = args.future_steps
    if args.image_indices is not None:
        writer_opts["image_indices"] = [int(x) for x in args.image_indices.split(",")]
    if args.samples_per_shard is not None:
        writer_opts["samples_per_shard"] = args.samples_per_shard

    # No exception dressing: a failure (or Ctrl-C) must exit non-zero with its
    # traceback — convert_bags_parallel.sh reads shard exit codes under
    # pipefail, and a swallowed interrupt would report a half-ported shard as
    # success.
    port(
        raw_dir=args.raw_dir,
        repo_id=repo_id,
        contract_path=args.contract,
        framework=args.framework,
        root=args.root,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        embed_contract=args.embed_contract,
        writer_opts=writer_opts,
    )


if __name__ == "__main__":
    main()
