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

"""resolve_repo_file: local dir first, else a best-effort Hub probe, never raises."""

import pytest

from rosetta.contract.sidecar import resolve_repo_file


def test_local_dir_hit(tmp_path):
    (tmp_path / "train_config.json").write_text("{}")
    result = resolve_repo_file(str(tmp_path), "train_config.json", repo_type="model")
    assert result == tmp_path / "train_config.json"


def test_local_dir_miss_returns_none(tmp_path):
    assert resolve_repo_file(str(tmp_path), "train_config.json", repo_type="model") is None


def test_nonexistent_local_path_falls_through_to_hub_probe(monkeypatch, tmp_path):
    """A path_or_repo_id that isn't a local dir is treated as a Hub repo id."""
    hf_hub = pytest.importorskip("huggingface_hub")

    seen = {}

    def _fake_download(*, repo_id, filename, repo_type):
        seen["repo_id"] = repo_id
        seen["filename"] = filename
        seen["repo_type"] = repo_type
        return str(tmp_path / "downloaded.json")

    monkeypatch.setattr(hf_hub, "hf_hub_download", _fake_download)

    result = resolve_repo_file("org/repo", "train_config.json", repo_type="model")

    assert seen == {"repo_id": "org/repo", "filename": "train_config.json", "repo_type": "model"}
    assert result == tmp_path / "downloaded.json"


def test_hub_not_found_returns_none(monkeypatch):
    hf_hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub.errors import HfHubHTTPError

    def _raise_not_found(**_kw):
        raise HfHubHTTPError("404")

    monkeypatch.setattr(hf_hub, "hf_hub_download", _raise_not_found)

    assert resolve_repo_file("org/does-not-exist", "train_config.json", repo_type="model") is None


def test_dataset_repo_type_and_nested_filename(monkeypatch, tmp_path):
    hf_hub = pytest.importorskip("huggingface_hub")

    seen = {}

    def _fake_download(*, repo_id, filename, repo_type):
        seen.update(repo_id=repo_id, filename=filename, repo_type=repo_type)
        return str(tmp_path / "rosetta_contract.yaml")

    monkeypatch.setattr(hf_hub, "hf_hub_download", _fake_download)

    resolve_repo_file("org/dataset", "meta/rosetta_contract.yaml", repo_type="dataset")

    assert seen == {"repo_id": "org/dataset", "filename": "meta/rosetta_contract.yaml", "repo_type": "dataset"}
