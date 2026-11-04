#!/usr/bin/env python
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

"""``rosetta-lerobot-train``: ``lerobot-train`` on recordings, through a Rosetta frame reader.

Every ``lerobot-train`` flag works. The ``--rosetta.*`` flags pick the data
instead of ``--dataset.repo_id``::

    rosetta-lerobot-train --rosetta.reader=rerun --rosetta.contract=robot.yaml \
        --rosetta.catalog_url=rerun+http://127.0.0.1:51234 --rosetta.dataset=my_task \
        --policy.type=act --output_dir=outputs/act

- ``--rosetta.reader``: a reader registered under ``rosetta.frame_readers``,
  such as ``rerun``.
- ``--rosetta.contract``: the contract's path.
- Any other ``--rosetta.<option>=<value>`` goes to the reader as a keyword
  argument. Rerun's are ``catalog_url``, ``dataset``, ``token``, ``segments``
  (comma-separated), ``manifest``, ``shuffle`` and ``shuffle_buffer``.

LeRobot 0.6.1 has no hook for another dataset, so this runs LeRobot's own
``lerobot_train.main()`` with two of its functions replaced:
``make_train_eval_datasets`` returns a :class:`~.view.LeRobotView` over the
reader, and ``save_checkpoint`` also copies the contract into each
checkpoint's ``pretrained_model/``, where
:func:`~.sidecar.find_contract_for_pretrained` finds it at deployment.

The view is read by index, as a ported dataset is, so LeRobot's own sampler
shuffles, resumes exactly and shards across ranks, and
``--dataset.eval_split`` holds out episodes as it does for a ported dataset.
``--rosetta.shuffle=block`` (Rerun) streams instead
(``--dataset.streaming=true`` is set): Rerun reads contiguous blocks far
faster, but streaming has no eval split.
"""

from __future__ import annotations

import inspect
import shutil
import sys
from pathlib import Path
from typing import Any

from rosetta.contract.sidecar import CONTRACT_SIDECAR_FILENAME
from rosetta.data import frame_reader_class, load_frame_reader

from .sidecar import READER_DATASET_NAMESPACE

__all__ = ["main", "split_rosetta_args"]

PATCHED = ("make_train_eval_datasets", "save_checkpoint")
"""The names in ``lerobot.scripts.lerobot_train`` this wrapper replaces."""


def split_rosetta_args(argv: list[str]) -> tuple[dict[str, str], list[str]]:
    """``--rosetta.<name>=<value>`` flags as a dict, and every other argument unchanged."""
    options: dict[str, str] = {}
    rest: list[str] = []
    for arg in argv:
        if arg.startswith("--rosetta."):
            name, sep, value = arg[len("--rosetta.") :].partition("=")
            if not sep or not name:
                raise SystemExit(f"Expected --rosetta.<name>=<value>, got {arg!r}")
            options[name.replace("-", "_")] = value
        else:
            rest.append(arg)
    return options, rest


def _has_flag(argv: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in argv)


def _reader_options(cls: type, options: dict[str, str], cfg: Any) -> dict[str, Any]:
    """The reader's keyword options, plus ``num_workers`` when it takes one (Rerun's manifest must match)."""
    accepted = inspect.signature(cls).parameters
    if "num_workers" in accepted and "num_workers" not in options:
        options = {**options, "num_workers": cfg.num_workers}
    return options


def main(argv: list[str] | None = None) -> None:
    from lerobot.scripts import lerobot_train
    from lerobot.utils.constants import PRETRAINED_MODEL_DIR

    argv = list(sys.argv[1:] if argv is None else argv)
    options, rest = split_rosetta_args(argv)
    try:
        reader_name = options.pop("reader")
        contract = Path(options.pop("contract"))
    except KeyError as e:
        raise SystemExit(f"rosetta-lerobot-train needs --rosetta.{e.args[0]}=...") from None
    if not contract.is_file():
        raise SystemExit(f"Contract not found: {contract}")
    missing = [name for name in PATCHED if not hasattr(lerobot_train, name)]
    if missing:
        raise SystemExit(f"This LeRobot's lerobot_train has no {missing}; rosetta-lerobot-train needs lerobot<0.6.2.")

    if not _has_flag(rest, "--dataset.repo_id"):
        rest.append(f"--dataset.repo_id={READER_DATASET_NAMESPACE}/{reader_name}")
    # Block shuffle streams. Anything else is read by index, under LeRobot's
    # sampler: streaming an indexed view would train in episode order.
    streams = options.get("shuffle") == "block"
    if streams:
        rest = [a for a in rest if not a.startswith("--dataset.streaming")]
        rest.append("--dataset.streaming=true")
    elif any(a.startswith("--dataset.streaming") and not a.endswith("=false") for a in rest):
        raise SystemExit(
            "rosetta-lerobot-train reads by index, with LeRobot's own sampler; --dataset.streaming is only for "
            "--rosetta.shuffle=block."
        )

    def make_train_eval_datasets(cfg):
        from lerobot.transforms import ImageTransforms

        from .view import LeRobotStreamView, LeRobotView, train_eval_views

        if streams and cfg.dataset.eval_split > 0:
            raise ValueError(
                "--rosetta.shuffle=block streams, and streaming has no eval split. Pass --dataset.eval_split=0, "
                "or drop --rosetta.shuffle=block to read by index."
            )
        if streams and cfg.dataset.episodes is not None:
            raise ValueError(
                "--rosetta.shuffle=block streams every episode; drop --dataset.episodes, or drop "
                "--rosetta.shuffle=block to read by index."
            )
        reader_cls = frame_reader_class(reader_name)
        reader = load_frame_reader(reader_name, contract=contract, **_reader_options(reader_cls, options, cfg))
        transforms = ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
        common = {"repo_id": cfg.dataset.repo_id, "use_imagenet_stats": cfg.dataset.use_imagenet_stats}
        if streams:
            stream = LeRobotStreamView(
                reader, cfg.trainable_config, image_transforms=transforms, seed=cfg.seed or 0, **common
            )
            return stream, None
        view = LeRobotView(reader, cfg.trainable_config, image_transforms=transforms, **common)
        return train_eval_views(
            view, episodes=cfg.dataset.episodes, eval_split=cfg.dataset.eval_split, image_transforms=transforms
        )

    original_save = lerobot_train.save_checkpoint

    def save_checkpoint(*args, **kwargs):
        original_save(*args, **kwargs)
        checkpoint_dir = Path(kwargs["checkpoint_dir"] if "checkpoint_dir" in kwargs else args[0])
        shutil.copy(contract, checkpoint_dir / PRETRAINED_MODEL_DIR / CONTRACT_SIDECAR_FILENAME)

    originals = {name: getattr(lerobot_train, name) for name in PATCHED}
    lerobot_train.make_train_eval_datasets = make_train_eval_datasets
    lerobot_train.save_checkpoint = save_checkpoint
    sys.argv = [sys.argv[0] if sys.argv else "rosetta-lerobot-train", *rest]
    try:
        lerobot_train.main()
    finally:
        for name, original in originals.items():
            setattr(lerobot_train, name, original)
