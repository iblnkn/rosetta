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

"""The contract sidecar: its canonical on-disk names and best-effort resolution.

A dataset (or a checkpoint's dataset) carries its contract as a sidecar file
next to the data. This module names that placement and resolves it -- from a
local directory or an HF Hub repo -- without requiring callers to know which
they have, and without hard-depending on ``huggingface_hub``. Failures are
reported through an injectable ``warn`` callback (default: stdlib
``logging.warning``) so ROS callers can route them to their own logger.

Set ``HF_HUB_OFFLINE=1`` to keep Hub probes off the network entirely; probes
then fail fast and already-cached files still resolve.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Literal

import yaml

CONTRACT_SIDECAR_FILENAME = "rosetta_contract.yaml"
"""Bare sidecar filename; vla_foundry's tar-shard layout puts it at the output root."""

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

    ``path_or_repo_id`` is tried as a local directory first; if it isn't one,
    ``filename`` is fetched from the Hub via ``huggingface_hub.hf_hub_download``
    (with a short etag timeout -- this is a probe, not a required download).
    Never raises: a local miss returns ``None`` quietly, while every real
    failure -- Hub error, network problem, ``huggingface_hub`` not installed --
    returns ``None`` after reporting through ``warn``.
    """
    local_dir = Path(path_or_repo_id)
    if local_dir.is_dir():
        candidate = local_dir / filename
        return candidate if candidate.is_file() else None
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
        # hf_hub_download also raises plain OSError/ValueError for a bad repo
        # id or missing entry depending on version; this is a best-effort
        # probe, not a required path, so any failure just means "not found".
        warn(f"Could not fetch {filename} from {path_or_repo_id!r}: {e}")
        return None


def find_contract_for_pretrained(
    pretrained: str,
    *,
    warn: Callable[[str], None] = logging.warning,
) -> Path | None:
    """Chase a pretrained checkpoint -> train_config.json -> dataset -> its contract sidecar.

    Best-effort and pure (no ROS): any missing link -- no checkpoint given, no
    train_config.json, no usable dataset reference, no sidecar on the dataset --
    resolves to ``None`` after a warning naming the broken link. Never raises.
    """
    if not pretrained:
        return None

    train_config_path = resolve_repo_file(pretrained, "train_config.json", repo_type="model", warn=warn)
    if train_config_path is None:
        return None

    try:
        dataset = json.loads(train_config_path.read_text())["dataset"]
        dataset_ref = dataset.get("root") or dataset.get("repo_id")
    except Exception as e:
        warn(f"Failed to read dataset reference from {train_config_path}: {e}")
        return None
    if not isinstance(dataset_ref, str) or not dataset_ref:
        warn(f"{train_config_path}: dataset has neither a usable 'root' nor 'repo_id'")
        return None

    return resolve_repo_file(dataset_ref, LEROBOT_CONTRACT_SIDECAR_PATH, repo_type="dataset", warn=warn)


def scan_inline_codec_paths(contract_path: Path | str) -> list[tuple[str, str]]:
    """Raw-YAML scan for inline codec declarations, importing nothing.

    Walks the parsed YAML document and returns every ``("decoder"|"encoder",
    path)`` pair with a non-empty string value. Deliberately structure-blind:
    it exists to run BEFORE ``load_contract`` -- whose parse imports and
    executes those paths -- so it cannot rely on the schema. Over-matching an
    unexpected nested key is fine; under-matching is not. YAML parse errors
    propagate: ``load_contract`` would fail identically one step later.
    """
    found: list[tuple[str, str]] = []

    def walk(obj) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in ("decoder", "encoder") and isinstance(value, str) and value.strip():
                    found.append((key, value))
                walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(yaml.safe_load(Path(contract_path).read_text()))
    return found
