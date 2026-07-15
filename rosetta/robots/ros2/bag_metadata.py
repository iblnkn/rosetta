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

"""Bag metadata custom_data: the recorder/porter seam.

The episode recorder writes the operator prompt, embedded contract, and
contract hash into rosbag2's metadata.yaml ``custom_data`` block; the bag
porter reads them back. The key constants and read/write/hash helpers live
here — one home for the writer/reader pair, so a drifted key literal cannot
make the porter silently stop finding prompts or hashes in new bags (both
readers default to "" by design).

ROS-free (yaml/hashlib/pathlib only).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

BAG_METADATA_KEY = "rosbag2_bagfile_information"
BAG_CUSTOM_DATA_KEY = "custom_data"
BAG_PROMPT_KEY = "lerobot.operator_prompt"
BAG_CONTRACT_KEY = "rosetta.contract_yaml"
BAG_CONTRACT_HASH_KEY = "rosetta.contract_hash"


def contract_hash(path: Path | str) -> str:
    """sha256 hex digest of the contract file's BYTES.

    Bytes, not text: the recorder embeds this hash at record time and the
    porter compares it against its own ``--contract`` hash — text mode's
    universal-newline translation would make the identical file hash
    differently (e.g. CRLF checkouts) and produce spurious mismatch warnings.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_bag_metadata(bag_dir: Path) -> dict[str, Any]:
    """Read a bag's metadata.yaml ({} when absent)."""
    meta_path = Path(bag_dir) / "metadata.yaml"
    if not meta_path.exists():
        return {}
    with meta_path.open() as f:
        return yaml.safe_load(f) or {}


def read_custom_field(meta: dict[str, Any], key: str) -> str:
    """One custom_data value from loaded bag metadata ("" when absent).

    Tolerates absent/null blocks: bags recorded outside rosetta legitimately
    have no custom_data, and the porter treats that as "no prompt/hash".
    """
    custom = (meta.get(BAG_METADATA_KEY) or {}).get(BAG_CUSTOM_DATA_KEY) or {}
    if not isinstance(custom, dict):
        return ""
    return str(custom.get(key, ""))


def update_custom_data(meta_path: Path, entries: dict[str, str]) -> None:
    """Read-modify-write ``entries`` into an existing metadata.yaml's custom_data.

    Raises on a missing/unreadable file — retry policy (rosbag2 writes
    metadata.yaml asynchronously at bag close) belongs to the caller.
    """
    with meta_path.open("r") as f:
        meta = yaml.safe_load(f) or {}
    info = meta.get(BAG_METADATA_KEY) or {}
    meta[BAG_METADATA_KEY] = info
    custom = info.get(BAG_CUSTOM_DATA_KEY) or {}
    info[BAG_CUSTOM_DATA_KEY] = custom
    custom.update(entries)
    with meta_path.open("w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
