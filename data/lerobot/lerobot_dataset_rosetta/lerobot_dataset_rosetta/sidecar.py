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

"""Where a LeRobot dataset keeps its contract, and finding it from a checkpoint.

:class:`~lerobot_dataset_rosetta.dataset_writer.LeRobotDatasetWriter` copies
the contract to :data:`LEROBOT_CONTRACT_SIDECAR_PATH` in each dataset.
:func:`find_contract_for_pretrained` finds the contract a LeRobot checkpoint was
trained with: in the checkpoint itself (``rosetta-lerobot-train`` puts it
there), or by following the checkpoint back to that file. These helpers find
files in a local directory or an HF Hub repo (``huggingface_hub`` is
optional). Failures go to the ``warn`` callback so ROS callers can route them
to their own logger.

``HF_HUB_OFFLINE=1`` keeps Hub probes off the network; cached files still
resolve.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Literal

from rosetta.contract.sidecar import CONTRACT_SIDECAR_FILENAME

__all__ = [
    "LEROBOT_CONTRACT_SIDECAR_PATH",
    "READER_DATASET_NAMESPACE",
    "find_contract_for_pretrained",
    "resolve_repo_file",
]

LEROBOT_CONTRACT_SIDECAR_PATH = f"meta/{CONTRACT_SIDECAR_FILENAME}"
"""Repo-relative sidecar path in the LeRobot dataset layout (extra file under ``meta/``)."""

READER_DATASET_NAMESPACE = "rosetta-lerobot-train"
"""Namespace of the placeholder ``--dataset.repo_id`` that ``rosetta-lerobot-train`` sets.

No dataset holds the contract of a checkpoint trained from a frame reader, so
:func:`find_contract_for_pretrained` never looks a repo in this namespace up
on the Hub.
"""

_HUB_ETAG_TIMEOUT_S = 5.0  # best-effort probe; don't let a slow Hub stall lifecycle configure


def resolve_repo_file(
    path_or_repo_id: str,
    filename: str,
    *,
    repo_type: Literal["model", "dataset"],
    warn: Callable[[str], None] = logging.warning,
) -> Path | None:
    """Return a local path to ``filename`` under ``path_or_repo_id``, or ``None``.

    ``path_or_repo_id`` is tried as a local directory first, then as a Hub
    repo. Never raises: every miss returns ``None`` after calling ``warn``
    with what was missing.
    """
    local_dir = Path(path_or_repo_id)
    if local_dir.is_dir():
        candidate = local_dir / filename
        if not candidate.is_file():
            warn(f"{candidate} not found")
            candidate = None
        return candidate
    if local_dir.exists():
        warn(f"{path_or_repo_id} exists but is not a directory; not probing the Hub for {filename}")
        return None

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:
        warn(f"huggingface_hub not installed; cannot fetch {filename} from {path_or_repo_id!r}")
        return None

    try:
        path = hf_hub_download(
            repo_id=path_or_repo_id,
            filename=filename,
            repo_type=repo_type,
            etag_timeout=_HUB_ETAG_TIMEOUT_S,
        )
        return Path(path)
    except HfHubHTTPError as e:
        warn(f"Hub HTTP error fetching {filename} from {path_or_repo_id!r}: {e}")
        return None
    except Exception as e:
        # hf_hub_download also raises OSError/ValueError for a bad repo id or
        # missing file, depending on version.
        warn(f"Could not fetch {filename} from {path_or_repo_id!r}: {e}")
        return None


def find_contract_for_pretrained(
    pretrained: str,
    *,
    warn: Callable[[str], None] = logging.warning,
) -> Path | None:
    """The contract ``pretrained`` was trained with, or ``None``.

    First the checkpoint's own ``rosetta_contract.yaml``, which
    ``rosetta-lerobot-train`` writes next to the weights. Then the chase a
    ported dataset needs: checkpoint -> ``train_config.json`` -> dataset -> its
    contract sidecar. Never raises: any missing link returns ``None`` after a
    warning naming it.

    The dataset's ``root`` is a local directory, resolved against the current
    directory when relative, and never fetched from the Hub. ``repo_id`` is
    tried as a local directory, then on the Hub. When neither finds the
    contract, one warning lists every attempt.
    """
    if not pretrained:
        return None

    # Most checkpoints trained on a ported dataset don't have it, so a miss
    # here is quiet; the chase below warns about what it can't find.
    own = resolve_repo_file(pretrained, CONTRACT_SIDECAR_FILENAME, repo_type="model", warn=lambda _: None)
    if own is not None:
        return own

    train_config_path = resolve_repo_file(pretrained, "train_config.json", repo_type="model", warn=warn)
    if train_config_path is None:
        return None

    try:
        dataset = json.loads(train_config_path.read_text())["dataset"]
        dataset_refs = [dataset.get("root"), dataset.get("repo_id")]
    except Exception as e:
        warn(f"Failed to read dataset reference from {train_config_path}: {e}")
        return None
    dataset_refs = list(dict.fromkeys(ref for ref in dataset_refs if isinstance(ref, str) and ref))
    placeholders = [ref for ref in dataset_refs if ref.startswith(f"{READER_DATASET_NAMESPACE}/")]
    for ref in placeholders:
        warn(
            f"{pretrained} was trained from a frame reader ({ref}) and holds no {CONTRACT_SIDECAR_FILENAME}, "
            "as a model pushed to the Hub doesn't. Pass contract_path."
        )
    dataset_refs = [ref for ref in dataset_refs if ref not in placeholders]
    if not dataset_refs:
        if not placeholders:
            warn(f"{train_config_path}: dataset has neither a usable 'root' nor 'repo_id'")
        return None

    root, repo_id = dataset.get("root"), dataset.get("repo_id")
    return _dataset_contract(
        pretrained,
        root if root in dataset_refs else None,
        repo_id if repo_id in dataset_refs else None,
        warn,
    )


def _dataset_contract(
    pretrained: str, root: str | None, repo_id: str | None, warn: Callable[[str], None]
) -> Path | None:
    """The dataset's contract sidecar from its local ``root``, then from ``repo_id``.

    root first: a root recorded on the training machine is usually missing on
    the robot, while the Hub copy carries the sidecar too. Misses are reported
    in one warning, and only when neither finds the contract.
    """
    misses: list[str] = []
    relative_root_missing = False
    if root:
        if Path(root).is_dir():
            path = resolve_repo_file(root, LEROBOT_CONTRACT_SIDECAR_PATH, repo_type="dataset", warn=misses.append)
            if path is not None:
                return path
        elif Path(root).is_absolute():
            misses.append(f"dataset.root {root!r} is not a directory")
        else:
            relative_root_missing = True
            misses.append(f"dataset.root {root!r} (relative) is not a directory under {Path.cwd()}")
    # A repo_id equal to a root that is a local directory names the same place.
    if repo_id and not (repo_id == root and Path(root).is_dir()):
        path = resolve_repo_file(repo_id, LEROBOT_CONTRACT_SIDECAR_PATH, repo_type="dataset", warn=misses.append)
        if path is not None:
            return path
    hint = " Deploy from the directory training ran in, or pass contract_path." if relative_root_missing else ""
    warn(f"No contract found for {pretrained}: " + "; ".join(m.rstrip(".") for m in misses) + "." + hint)
    return None
