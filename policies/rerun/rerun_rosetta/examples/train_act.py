#!/usr/bin/env python3
# Copyright 2026 Isaac Blankenau
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

r"""Train a LeRobot ACT policy directly from rrd bags on a Rerun server.

    rerun server                                   # terminal 1
    python examples/train_act.py \\
        --raw-dir ./datasets/bags --contract contract.yaml \\
        --dataset-name my_task --steps 500         # terminal 2

Frames come straight off the catalog server through the compiled contract
query: the same frames the porter would have written to a LeRobot dataset
(test/test_query_bag_parity.py checks this).

Against a hub the command is the same with the recordings uploaded first:
`--catalog-url rerun+https://<hub>` `--prefix s3://bucket/run42` `--token ...`
(or `RERUN_TOKEN`). A `file://` prefix needs the server on the same filesystem.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from rerun.catalog import CatalogClient, OnDuplicateSegmentLayer
from rerun.experimental.dataloader import BlockShuffle, SampleShuffle

from rerun_rosetta import contract_dataset, load_stats, write_stats_layer


def lerobot_stats(stats: dict, image_keys: set[str]) -> dict:
    """rerun_rosetta's stats in LeRobot's form: images per channel ``(3, 1, 1)`` in [0, 1]."""
    out: dict = {}
    for key, key_stats in stats.items():
        out[key] = {}
        for name, value in key_stats.items():
            if key in image_keys and name != "count":
                value = value.reshape(-1, 1, 1) / 255.0
            out[key][name] = torch.from_numpy(np.asarray(value, dtype=np.float64 if name == "count" else np.float32))
    return out


def collate(samples: list[dict], image_keys: set[str]) -> dict[str, torch.Tensor]:
    """Frame dicts -> ACT batch: images HWC uint8 -> CHW float [0, 1], vectors float32.

    ``action`` arrives as a ``[chunk, D]`` window. The dataloader never lets a
    window cross the episode edge, so no chunk position is padded.
    """
    batch: dict[str, torch.Tensor] = {}
    for key in samples[0]:
        stacked = torch.from_numpy(np.stack([s[key] for s in samples]))
        batch[key] = stacked.permute(0, 3, 1, 2).float() / 255.0 if key in image_keys else stacked.float()
    batch["action_is_pad"] = torch.zeros(batch["action"].shape[:2], dtype=torch.bool)
    return batch


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory of rrd bag subdirs")
    parser.add_argument("--contract", type=Path, required=True, help="Contract YAML path")
    parser.add_argument("--catalog-url", default="rerun+http://127.0.0.1:51234")
    parser.add_argument("--token", default=os.environ.get("RERUN_TOKEN"), help="Catalog auth token (hub)")
    parser.add_argument("--prefix", default=None, help="Recordings prefix to register (default: file://<raw-dir>)")
    parser.add_argument("--dataset-name", default=None, help="Catalog dataset name (default: raw-dir name)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/train/act_from_rrd"))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=50, help="ACT action chunk length (grid steps)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--block-shuffle-buffer",
        type=int,
        default=0,
        help="Use BlockShuffle with this emission buffer (decoded samples held per worker); 0 = SampleShuffle",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Save the sample manifest here and reuse it (delete after adding recordings)",
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=Path("datasets/rosetta_stats"),
        help="Where the stats layer's .rrd files are written (on the catalog server's filesystem for file://)",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    name = args.dataset_name or args.raw_dir.name
    prefix = args.prefix or f"file://{args.raw_dir.resolve()}"
    dataset = CatalogClient(args.catalog_url, token=args.token).create_dataset(name, exist_ok=True)
    # REPLACE: re-running after new recordings is the normal workflow.
    dataset.register_prefix(prefix, on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
    if not dataset.segment_ids():
        sys.exit(f"No recordings found under {prefix}")

    frames = contract_dataset(
        args.catalog_url,
        name,
        args.contract,
        token=args.token,
        windows={"action": (0, args.chunk_size - 1)},
        shuffle_strategy=(
            BlockShuffle(buffer_size=args.block_shuffle_buffer) if args.block_shuffle_buffer else SampleShuffle()
        ),
        num_workers=args.num_workers,
        manifest=args.manifest,
    )
    logging.info("Dataset: %d samples from %d episodes", len(frames), len(dataset.segment_ids()))

    from lerobot.configs.types import FeatureType
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.feature_utils import dataset_to_policy_features

    image_keys = {k for k, f in frames.features.items() if f["dtype"] == "video"}
    features = dataset_to_policy_features({k: f for k, f in frames.features.items() if f["dtype"] != "string"})
    config = ACTConfig(
        input_features={k: f for k, f in features.items() if f.type != FeatureType.ACTION},
        output_features={k: f for k, f in features.items() if f.type == FeatureType.ACTION},
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        device=args.device,
    )
    loader = torch.utils.data.DataLoader(
        frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        multiprocessing_context="spawn" if args.num_workers else None,
        persistent_workers=args.num_workers > 0,  # workers read each epoch set by set_epoch
        collate_fn=partial(collate, image_keys=image_keys),
    )
    # Exact stats over every frame, kept as a layer beside the recordings: only
    # episodes without one are computed. Normalization lives in lerobot's processor
    # pipelines, which are saved with the weights so inference applies the same stats.
    computed = write_stats_layer(dataset, args.contract, args.stats_dir, token=args.token)
    logging.info("Stats computed for %d new episodes", len(computed))
    stats = lerobot_stats(load_stats(dataset, args.contract), image_keys)
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    policy = ACTPolicy(config)
    policy.train().to(args.device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    step = 0
    for epoch in itertools.count():
        frames.set_epoch(epoch)
        for batch in loader:
            loss, _ = policy.forward(preprocessor(batch))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % 50 == 0 or step == 1:
                logging.info("epoch %d  step %d  loss %.4f", epoch, step, loss.item())
            if step >= args.steps:
                policy.save_pretrained(args.output_dir)
                preprocessor.save_pretrained(args.output_dir)
                postprocessor.save_pretrained(args.output_dir)
                logging.info("Saved checkpoint and processors to %s", args.output_dir)
                return


if __name__ == "__main__":
    main()
