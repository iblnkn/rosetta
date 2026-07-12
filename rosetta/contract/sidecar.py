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

"""Resolve one named file out of a local directory or an HF Hub repo.

Shared by anything that needs to probe for an optional sidecar file (e.g. a
contract) alongside a LeRobot dataset or checkpoint, without requiring the
caller to know in advance whether ``path_or_repo_id`` is a local path or a
Hub repo id, and without hard-depending on ``huggingface_hub`` when it isn't
installed.
"""

from __future__ import annotations

from pathlib import Path


def resolve_repo_file(path_or_repo_id: str, filename: str, repo_type: str) -> Path | None:
    """Return a local path to ``filename`` under ``path_or_repo_id``, or ``None``.

    ``path_or_repo_id`` is tried as a local directory first; if it isn't one,
    ``filename`` is fetched from the Hub via ``huggingface_hub.hf_hub_download``
    (``repo_type`` is ``"model"`` or ``"dataset"``). Any failure -- missing
    file, network error, ``huggingface_hub`` not installed -- resolves to
    ``None`` rather than raising, since this is always a best-effort fallback
    probe with a well-defined "not found" behavior at the call site.
    """
    local_dir = Path(path_or_repo_id)
    if local_dir.is_dir():
        candidate = local_dir / filename
        return candidate if candidate.is_file() else None

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import HfHubHTTPError

        path = hf_hub_download(repo_id=path_or_repo_id, filename=filename, repo_type=repo_type)
        return Path(path)
    except ImportError:
        return None
    except HfHubHTTPError:
        return None
    except Exception:
        # hf_hub_download also raises plain OSError/ValueError for a bad repo
        # id or missing entry depending on version; this is a best-effort
        # probe, not a required path, so any failure just means "not found".
        return None
