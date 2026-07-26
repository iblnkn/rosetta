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

"""Bag metadata custom_data helpers: the recorder/porter seam.

The episode recorder writes the operator prompt and embedded contract into
rosbag2's metadata.yaml ``custom_data`` block. The bag porter reads them back.
Keeping the key constants and the read/write helpers in one module means a
drifted key literal cannot silently stop the porter from finding prompts or
contracts in new bags. Readers default to empty on any absence.

ROS-free (yaml/pathlib only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

BAG_METADATA_KEY = "rosbag2_bagfile_information"
BAG_CUSTOM_DATA_KEY = "custom_data"
BAG_PROMPT_KEY = "lerobot.operator_prompt"
BAG_CONTRACT_KEY = "rosetta.contract_yaml"
#: UUID of the RecordEpisode goal that produced the bag, as the canonical
#: 8-4-4-4-12 hex string. Empty for a service-started recording, which has no
#: goal. Every action goal already carries a unique id; writing it here is what
#: lets a bag be traced back to the request that made it, and lets a client
#: that kept its goal id find the bag afterwards.
BAG_GOAL_ID_KEY = "rosetta.goal_id"


def read_bag_metadata(bag_dir: Path) -> dict[str, Any]:
    """Read a bag's metadata.yaml ({} when absent)."""
    meta_path = Path(bag_dir) / "metadata.yaml"
    if not meta_path.exists():
        return {}
    with meta_path.open() as f:
        return yaml.safe_load(f) or {}


def read_custom_field(meta: dict[str, Any], key: str) -> str:
    """One custom_data value from loaded bag metadata ("" when absent).

    Bags recorded outside rosetta legitimately carry no custom_data, and the
    porter treats that as no prompt and no contract. The isinstance guard also
    swallows a custom_data that parsed as a non-mapping, returning "".
    """
    custom = (meta.get(BAG_METADATA_KEY) or {}).get(BAG_CUSTOM_DATA_KEY) or {}
    if not isinstance(custom, dict):
        return ""
    return str(custom.get(key, ""))


def update_custom_data(meta_path: Path, entries: dict[str, str]) -> None:
    """Read-modify-write ``entries`` into an existing metadata.yaml's custom_data.

    Raises on a missing or unreadable file. rosbag2 writes metadata.yaml
    asynchronously at bag close, so the file may not exist on early attempts.
    The retry policy for that race belongs to the caller.
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
