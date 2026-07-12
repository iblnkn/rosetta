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

"""_resolve_fallback_contract_path chases pretrained_name_or_path ->
train_config.json -> dataset.{repo_id,root} -> the dataset's contract
sidecar. Every missing link must degrade to "", never raise (the caller
falls back to today's required-contract_path error)."""

import json

from rosetta.robots.ros2.nodes import policy_runner_node as node_mod
from rosetta.robots.ros2.nodes.policy_runner_node import PolicyRunnerNode


class _FakeLogger:
    def info(self, _msg):
        pass

    def warning(self, _msg):
        pass

    def error(self, _msg):
        pass


class _FakeParam:
    def __init__(self, value):
        self.value = value


class _FakeSelf:
    """Enough of PolicyRunnerNode for _resolve_fallback_contract_path to run unbound."""

    def __init__(self, pretrained_name_or_path: str):
        self._pretrained = pretrained_name_or_path

    def get_parameter(self, name):
        assert name == "pretrained_name_or_path"
        return _FakeParam(self._pretrained)

    def get_logger(self):
        return _FakeLogger()


def _resolve(pretrained_name_or_path: str) -> str:
    return PolicyRunnerNode._resolve_fallback_contract_path(_FakeSelf(pretrained_name_or_path))


def test_empty_pretrained_path_yields_no_fallback():
    assert _resolve("") == ""


def test_full_chain_resolves_to_dataset_contract(monkeypatch, tmp_path):
    train_config = tmp_path / "checkpoint" / "train_config.json"
    train_config.parent.mkdir(parents=True)
    train_config.write_text(json.dumps({"dataset": {"repo_id": "my_data", "root": str(tmp_path / "dataset")}}))

    contract_sidecar = tmp_path / "dataset" / "meta" / "rosetta_contract.yaml"
    contract_sidecar.parent.mkdir(parents=True)
    contract_sidecar.write_text("robot_type: x\n")

    def _fake_resolve_repo_file(path_or_repo_id, filename, repo_type):
        if repo_type == "model":
            assert path_or_repo_id == str(tmp_path / "checkpoint")
            assert filename == "train_config.json"
            return train_config
        assert repo_type == "dataset"
        assert path_or_repo_id == str(tmp_path / "dataset")
        assert filename == "meta/rosetta_contract.yaml"
        return contract_sidecar

    monkeypatch.setattr(node_mod, "resolve_repo_file", _fake_resolve_repo_file)

    assert _resolve(str(tmp_path / "checkpoint")) == str(contract_sidecar)


def test_missing_train_config_yields_no_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(node_mod, "resolve_repo_file", lambda *_a, **_k: None)
    assert _resolve(str(tmp_path / "checkpoint")) == ""


def test_missing_dataset_contract_yields_no_fallback(monkeypatch, tmp_path):
    train_config = tmp_path / "train_config.json"
    train_config.write_text(json.dumps({"dataset": {"repo_id": "my_data", "root": None}}))

    def _fake_resolve_repo_file(_path_or_repo_id, filename, repo_type):
        if repo_type == "model":
            return train_config
        return None  # dataset unreachable / no sidecar

    monkeypatch.setattr(node_mod, "resolve_repo_file", _fake_resolve_repo_file)

    assert _resolve(str(tmp_path)) == ""


def test_malformed_train_config_yields_no_fallback(monkeypatch, tmp_path):
    train_config = tmp_path / "train_config.json"
    train_config.write_text("not json")
    monkeypatch.setattr(node_mod, "resolve_repo_file", lambda *_a, **_k: train_config)

    assert _resolve(str(tmp_path)) == ""


def test_dataset_repo_id_used_when_root_is_none(monkeypatch, tmp_path):
    """A Hub-hosted (not locally-rooted) dataset falls back to repo_id."""
    train_config = tmp_path / "train_config.json"
    train_config.write_text(json.dumps({"dataset": {"repo_id": "org/my_data", "root": None}}))

    seen = {}

    def _fake_resolve_repo_file(path_or_repo_id, filename, repo_type):
        if repo_type == "model":
            return train_config
        seen["path_or_repo_id"] = path_or_repo_id
        return tmp_path / "contract.yaml"

    monkeypatch.setattr(node_mod, "resolve_repo_file", _fake_resolve_repo_file)

    _resolve(str(tmp_path))
    assert seen["path_or_repo_id"] == "org/my_data"
