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

"""find_contract_for_pretrained chases pretrained -> train_config.json ->
dataset -> contract sidecar; every missing link degrades to None with a
warning naming the broken link, and never raises."""

import json

import pytest

import rosetta.contract.sidecar as sidecar_mod
from rosetta.contract.sidecar import LEROBOT_CONTRACT_SIDECAR_PATH, find_contract_for_pretrained


def _make_checkpoint(tmp_path, dataset: dict) -> str:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "train_config.json").write_text(json.dumps({"dataset": dataset}))
    return str(checkpoint)


def test_empty_pretrained_yields_none():
    warnings = []
    assert find_contract_for_pretrained("", warn=warnings.append) is None
    assert warnings == []


def test_full_chain_resolves_to_dataset_contract(tmp_path):
    """End-to-end through the real local filesystem, no mocks."""
    dataset_root = tmp_path / "dataset"
    sidecar = dataset_root / LEROBOT_CONTRACT_SIDECAR_PATH
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("robot_type: x\n")
    pretrained = _make_checkpoint(tmp_path, {"repo_id": "org/my_data", "root": str(dataset_root)})

    warnings = []
    assert find_contract_for_pretrained(pretrained, warn=warnings.append) == sidecar
    assert warnings == []


def test_missing_train_config_yields_none(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    assert find_contract_for_pretrained(str(checkpoint)) is None


def test_missing_dataset_contract_yields_none(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    pretrained = _make_checkpoint(tmp_path, {"repo_id": "org/my_data", "root": str(dataset_root)})
    assert find_contract_for_pretrained(pretrained) is None


@pytest.mark.parametrize(
    "train_config_body",
    [
        pytest.param("not json", id="not-json"),
        pytest.param(json.dumps({}), id="no-dataset-key"),
        pytest.param(json.dumps({"dataset": {"repo_id": None, "root": None}}), id="repo-id-null"),
        pytest.param(json.dumps({"dataset": {"repo_id": 42}}), id="repo-id-not-str"),
        pytest.param(json.dumps({"dataset": "x"}), id="dataset-not-mapping"),
    ],
)
def test_malformed_train_config_warns_and_yields_none(tmp_path, train_config_body):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "train_config.json").write_text(train_config_body)

    warnings = []
    assert find_contract_for_pretrained(str(checkpoint), warn=warnings.append) is None
    assert len(warnings) == 1


def test_repo_id_used_when_root_is_none(monkeypatch, tmp_path):
    pretrained = _make_checkpoint(tmp_path, {"repo_id": "org/my_data", "root": None})
    seen = {}

    def _fake_resolve(path_or_repo_id, filename, *, repo_type, warn):
        if filename == "train_config.json":
            return tmp_path / "checkpoint" / "train_config.json"
        seen.update(path_or_repo_id=path_or_repo_id, filename=filename, repo_type=repo_type)
        return None

    monkeypatch.setattr(sidecar_mod, "resolve_repo_file", _fake_resolve)

    assert find_contract_for_pretrained(pretrained) is None
    assert seen == {
        "path_or_repo_id": "org/my_data",
        "filename": LEROBOT_CONTRACT_SIDECAR_PATH,
        "repo_type": "dataset",
    }


def test_root_preferred_over_repo_id(monkeypatch, tmp_path):
    pretrained = _make_checkpoint(tmp_path, {"repo_id": "org/my_data", "root": "/data/local"})
    seen = {}

    def _fake_resolve(path_or_repo_id, filename, *, repo_type, warn):
        if filename == "train_config.json":
            return tmp_path / "checkpoint" / "train_config.json"
        seen["path_or_repo_id"] = path_or_repo_id
        return None

    monkeypatch.setattr(sidecar_mod, "resolve_repo_file", _fake_resolve)

    assert find_contract_for_pretrained(pretrained) is None
    assert seen["path_or_repo_id"] == "/data/local"
