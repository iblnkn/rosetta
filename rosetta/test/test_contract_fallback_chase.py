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
dataset -> contract sidecar. Each link is a local directory or, failing that,
an HF Hub repo; every missing link degrades to None with a warning naming it,
a clean local miss stays quiet, and nothing raises."""

import json
import sys

import pytest

from rosetta.contract.sidecar import LEROBOT_CONTRACT_SIDECAR_PATH, find_contract_for_pretrained


def _make_checkpoint(tmp_path, dataset: dict) -> str:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "train_config.json").write_text(json.dumps({"dataset": dataset}))
    return str(checkpoint)


def _make_dataset(tmp_path):
    """A local dataset root holding a contract sidecar; returns (root, sidecar)."""
    dataset_root = tmp_path / "dataset"
    sidecar = dataset_root / LEROBOT_CONTRACT_SIDECAR_PATH
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("robot_type: x\n")
    return dataset_root, sidecar


@pytest.fixture
def hub(monkeypatch):
    """Serve hf_hub_download from a {(repo_id, filename, repo_type): local path} dict; anything else 404s."""
    hf_hub = pytest.importorskip("huggingface_hub")
    files = {}

    def _download(*, repo_id, filename, repo_type, **_kw):
        try:
            return str(files[(repo_id, filename, repo_type)])
        except KeyError:
            raise _hub_http_error() from None

    monkeypatch.setattr(hf_hub, "hf_hub_download", _download)
    return files


def _hub_http_error():
    import httpx
    from huggingface_hub.errors import HfHubHTTPError

    response = httpx.Response(404, request=httpx.Request("GET", "https://hub/probe"))
    return HfHubHTTPError("404", response=response)


def test_empty_pretrained_yields_none():
    warnings = []
    assert find_contract_for_pretrained("", warn=warnings.append) is None
    assert warnings == []


def test_full_chain_resolves_to_dataset_contract(tmp_path):
    """End-to-end through the real local filesystem, no mocks."""
    dataset_root, sidecar = _make_dataset(tmp_path)
    pretrained = _make_checkpoint(tmp_path, {"repo_id": "org/my_data", "root": str(dataset_root)})

    warnings = []
    assert find_contract_for_pretrained(pretrained, warn=warnings.append) == sidecar
    assert warnings == []


def test_missing_train_config_yields_none_quietly(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    warnings = []
    assert find_contract_for_pretrained(str(checkpoint), warn=warnings.append) is None
    assert warnings == []


def test_missing_dataset_contract_yields_none_quietly(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    pretrained = _make_checkpoint(tmp_path, {"repo_id": "org/my_data", "root": str(dataset_root)})
    warnings = []
    assert find_contract_for_pretrained(pretrained, warn=warnings.append) is None
    assert warnings == []


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


# --- Hub links -------------------------------------------------------------------


def test_repo_id_used_when_root_is_none(hub, tmp_path):
    _, sidecar = _make_dataset(tmp_path)
    # Literal on purpose: pins the on-the-wire filename behind the constant.
    hub["org/my_data", "meta/rosetta_contract.yaml", "dataset"] = sidecar
    pretrained = _make_checkpoint(tmp_path, {"repo_id": "org/my_data", "root": None})
    assert find_contract_for_pretrained(pretrained) == sidecar


def test_non_directory_pretrained_is_probed_as_hub_model(hub, tmp_path):
    dataset_root, sidecar = _make_dataset(tmp_path)
    train_config = tmp_path / "train_config.json"
    train_config.write_text(json.dumps({"dataset": {"root": str(dataset_root)}}))
    hub["org/policy", "train_config.json", "model"] = train_config
    assert find_contract_for_pretrained("org/policy") == sidecar


def test_local_file_warns_and_skips_hub_probe(monkeypatch, tmp_path):
    """A path that exists but isn't a directory is never a repo id; no network."""
    hf_hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(hf_hub, "hf_hub_download", lambda **_kw: pytest.fail("Hub probed for a file path"))
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("")

    warnings = []
    assert find_contract_for_pretrained(str(checkpoint), warn=warnings.append) is None
    assert len(warnings) == 1 and "not a directory" in warnings[0]


@pytest.mark.parametrize(
    ("make_exception", "fragment"),
    [
        pytest.param(_hub_http_error, "Hub HTTP error", id="hub-http-error"),
        pytest.param(lambda: ValueError("bad repo id"), "Could not fetch", id="value-error"),
        pytest.param(lambda: OSError("connection reset"), "Could not fetch", id="os-error"),
    ],
)
def test_hub_failure_returns_none_with_warning(monkeypatch, make_exception, fragment):
    hf_hub = pytest.importorskip("huggingface_hub")
    exception = make_exception()

    def _raise(**_kw):
        raise exception

    monkeypatch.setattr(hf_hub, "hf_hub_download", _raise)

    warnings = []
    assert find_contract_for_pretrained("org/does-not-exist", warn=warnings.append) is None
    assert len(warnings) == 1
    assert fragment in warnings[0] and "org/does-not-exist" in warnings[0]


def test_hub_not_installed_returns_none_with_warning(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    warnings = []
    assert find_contract_for_pretrained("org/repo", warn=warnings.append) is None
    assert len(warnings) == 1 and "not installed" in warnings[0]
