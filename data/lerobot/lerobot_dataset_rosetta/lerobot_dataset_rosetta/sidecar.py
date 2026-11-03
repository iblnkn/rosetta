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
:func:`find_contract_for_pretrained` follows a LeRobot checkpoint back to that
file. These helpers find files in a local directory or an HF Hub repo
(``huggingface_hub`` is optional). Failures go to the ``warn`` callback so ROS
callers can route them to their own logger.

``HF_HUB_OFFLINE=1`` keeps Hub probes off the network; cached files still
resolve.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Literal

from rosetta.contract.sidecar import CONTRACT_SIDECAR_FILENAME

__all__ = ["LEROBOT_CONTRACT_SIDECAR_PATH", "find_contract_for_pretrained", "resolve_repo_file"]

LEROBOT_CONTRACT_SIDECAR_PATH = f"meta/{CONTRACT_SIDECAR_FILENAME}"
"""Repo-relative sidecar path in the LeRobot dataset layout (extra file under ``meta/``)."""

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
    """Chase a pretrained checkpoint -> train_config.json -> dataset -> its contract sidecar.

    Never raises: any missing link returns ``None`` after a warning naming it.
    """
    if not pretrained:
        return None

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
    if not dataset_refs:
        warn(f"{train_config_path}: dataset has neither a usable 'root' nor 'repo_id'")
        return None

    # root first, then repo_id: a root recorded on the training machine is
    # usually missing on the robot, while the Hub copy carries the sidecar too.
    for ref in dataset_refs:
        found = resolve_repo_file(ref, LEROBOT_CONTRACT_SIDECAR_PATH, repo_type="dataset", warn=warn)
        if found is not None:
            return found
    return None
