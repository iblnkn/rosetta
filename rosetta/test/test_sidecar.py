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

"""resolve_repo_file: local dir first, else a best-effort Hub probe; never raises,
failures are reported through the injected warn callback, clean misses stay quiet.

scan_inline_codec_paths: the pre-load audit of a hub-resolved contract must
surface every inline decoder/encoder WITHOUT importing anything -- it exists
precisely because load_contract's parse already imports those paths."""

import sys

import pytest

from rosetta.contract.sidecar import (
    LEROBOT_CONTRACT_SIDECAR_PATH,
    resolve_repo_file,
    scan_inline_codec_paths,
)


def test_local_dir_hit(tmp_path):
    (tmp_path / "train_config.json").write_text("{}")
    result = resolve_repo_file(str(tmp_path), "train_config.json", repo_type="model")
    assert result == tmp_path / "train_config.json"


def test_local_dir_miss_returns_none_quietly(tmp_path):
    warnings = []
    result = resolve_repo_file(str(tmp_path), "train_config.json", repo_type="model", warn=warnings.append)
    assert result is None
    assert warnings == []


def test_nested_filename_in_local_dir(tmp_path):
    """The production combination: a local dataset root with the meta/ sidecar."""
    sidecar = tmp_path / LEROBOT_CONTRACT_SIDECAR_PATH
    sidecar.parent.mkdir()
    sidecar.write_text("robot_type: x\n")
    result = resolve_repo_file(str(tmp_path), LEROBOT_CONTRACT_SIDECAR_PATH, repo_type="dataset")
    assert result == sidecar


def test_local_file_warns_and_skips_hub_probe(monkeypatch, tmp_path):
    """A path that exists but isn't a directory is never a repo id; no network."""
    hf_hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(hf_hub, "hf_hub_download", lambda **_kw: pytest.fail("Hub probed for a file path"))
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("")

    warnings = []
    result = resolve_repo_file(str(checkpoint), "train_config.json", repo_type="model", warn=warnings.append)

    assert result is None
    assert len(warnings) == 1 and "not a directory" in warnings[0]


def test_nonexistent_local_path_falls_through_to_hub_probe(monkeypatch, tmp_path):
    """A path_or_repo_id that isn't a local dir is treated as a Hub repo id."""
    hf_hub = pytest.importorskip("huggingface_hub")

    seen = {}

    def _fake_download(*, repo_id, filename, repo_type, etag_timeout):
        seen.update(repo_id=repo_id, filename=filename, repo_type=repo_type, etag_timeout=etag_timeout)
        return str(tmp_path / "downloaded.json")

    monkeypatch.setattr(hf_hub, "hf_hub_download", _fake_download)

    result = resolve_repo_file("org/repo", "train_config.json", repo_type="model")

    assert seen == {
        "repo_id": "org/repo",
        "filename": "train_config.json",
        "repo_type": "model",
        "etag_timeout": 5.0,
    }
    assert result == tmp_path / "downloaded.json"


def test_dataset_repo_type_and_nested_filename(monkeypatch, tmp_path):
    hf_hub = pytest.importorskip("huggingface_hub")

    seen = {}

    def _fake_download(*, repo_id, filename, repo_type, etag_timeout):
        seen.update(repo_id=repo_id, filename=filename, repo_type=repo_type, etag_timeout=etag_timeout)
        return str(tmp_path / "rosetta_contract.yaml")

    monkeypatch.setattr(hf_hub, "hf_hub_download", _fake_download)

    resolve_repo_file("org/dataset", LEROBOT_CONTRACT_SIDECAR_PATH, repo_type="dataset")

    # Literal on purpose: pins the on-the-wire value behind the constant.
    assert seen == {
        "repo_id": "org/dataset",
        "filename": "meta/rosetta_contract.yaml",
        "repo_type": "dataset",
        "etag_timeout": 5.0,
    }


def _hub_http_error():
    import httpx
    from huggingface_hub.errors import HfHubHTTPError

    response = httpx.Response(404, request=httpx.Request("GET", "https://hub/probe"))
    return HfHubHTTPError("404", response=response)


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
    result = resolve_repo_file("org/does-not-exist", "train_config.json", repo_type="model", warn=warnings.append)

    assert result is None
    assert len(warnings) == 1
    assert fragment in warnings[0] and "org/does-not-exist" in warnings[0]


def test_hub_not_installed_returns_none_with_warning(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    warnings = []
    result = resolve_repo_file("org/repo", "train_config.json", repo_type="model", warn=warnings.append)

    assert result is None
    assert len(warnings) == 1 and "not installed" in warnings[0]


def test_scan_finds_nested_inline_codecs_and_ignores_blanks(tmp_path):
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        "robot_interface: ros2\n"
        "fps: 30\n"
        "observations:\n"
        "  observation.state:\n"
        "    channel:\n"
        "      topic: /joint_states\n"
        "      decoder: 'my_pkg.codecs:decode_state'\n"
        "actions:\n"
        "  action:\n"
        "    - channel:\n"
        "        topic: /cmd_a\n"
        "        encoder: 'my_pkg.codecs:encode_a'\n"
        "    - channel:\n"
        "        topic: /cmd_b\n"
        "        encoder: '   '\n"  # whitespace-only: treated as absent, like the schema does
    )

    assert sorted(scan_inline_codec_paths(contract)) == [
        ("decoder", "my_pkg.codecs:decode_state"),
        ("encoder", "my_pkg.codecs:encode_a"),
    ]


def test_scan_of_codec_free_contract_is_empty(tmp_path):
    contract = tmp_path / "contract.yaml"
    contract.write_text("robot_interface: ros2\nfps: 30\nobservations: {}\n")
    assert scan_inline_codec_paths(contract) == []


def test_scan_imports_nothing(tmp_path):
    """The whole point: the audit must run before any import happens."""
    module_name = "definitely_not_installed_codec_module_xyz"
    contract = tmp_path / "contract.yaml"
    contract.write_text(f"observations:\n  s:\n    channel:\n      decoder: '{module_name}:fn'\n")

    paths = scan_inline_codec_paths(contract)

    assert paths == [("decoder", f"{module_name}:fn")]
    assert module_name not in sys.modules
