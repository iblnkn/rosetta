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

"""port() must thread contract_path/embed_contract to the writer and warn on
a bag/--contract hash mismatch, without ever failing the port because of it
(--contract always wins for decoding)."""

import logging
from pathlib import Path
from types import SimpleNamespace

from rosetta.robots.ros2.offline import port as port_mod


class _StubWriter:
    def __init__(self):
        self.open_kwargs: dict = {}

    def open(self, **kw):
        self.open_kwargs = kw

    def add_frame(self, _frame):
        pass

    def save_episode(self):
        pass

    def discard_episode(self):
        pass

    def finalize(self):
        pass


class _StubSource:
    def __init__(self, *_a, **_k):
        pass

    def bag_dirs(self):
        return [Path("ep0")]


def _stub_iter_bag_frames(_bag_dir, _specs, warmup_keys=None):
    _ = warmup_keys
    yield {"observation.state": [0.0]}


def _patch_common(monkeypatch, writer):
    monkeypatch.setattr(port_mod, "load_contract", lambda _p: SimpleNamespace(tasks=[]))
    monkeypatch.setattr(port_mod, "iter_specs", lambda _c: iter(()))
    monkeypatch.setattr(port_mod, "iter_observation_specs", lambda _c: iter(()))
    monkeypatch.setattr(port_mod, "BagFrameSource", _StubSource)
    monkeypatch.setattr(port_mod, "load_dataset_writer", lambda _b: writer)
    monkeypatch.setattr(port_mod, "iter_bag_frames", _stub_iter_bag_frames)


def test_writer_receives_contract_path_and_embed_flag(monkeypatch, tmp_path):
    writer = _StubWriter()
    _patch_common(monkeypatch, writer)
    monkeypatch.setattr(port_mod, "read_bag_contract_hash", lambda _bag_dir: "")

    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text("robot_type: x\n")

    port_mod.port(tmp_path, "repo", contract_path, embed_contract=True)

    assert writer.open_kwargs["contract_path"] == contract_path
    assert writer.open_kwargs["embed_contract"] is True


def test_no_embed_contract_flag_propagates(monkeypatch, tmp_path):
    writer = _StubWriter()
    _patch_common(monkeypatch, writer)
    monkeypatch.setattr(port_mod, "read_bag_contract_hash", lambda _bag_dir: "")

    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text("robot_type: x\n")

    port_mod.port(tmp_path, "repo", contract_path, embed_contract=False)

    assert writer.open_kwargs["embed_contract"] is False


def test_hash_mismatch_warns_but_does_not_fail(monkeypatch, tmp_path, caplog):
    writer = _StubWriter()
    _patch_common(monkeypatch, writer)
    # Bag was recorded with a different contract than the one we're porting with.
    monkeypatch.setattr(port_mod, "read_bag_contract_hash", lambda _bag_dir: "stale-hash")

    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text("robot_type: x\n")

    with caplog.at_level(logging.WARNING):
        port_mod.port(tmp_path, "repo", contract_path, embed_contract=True)

    assert any("different contract" in rec.message for rec in caplog.records)
    # --contract still wins: the writer gets the --contract path regardless.
    assert writer.open_kwargs["contract_path"] == contract_path


def test_bag_without_embedded_contract_still_gets_dataset_sidecar(monkeypatch, tmp_path):
    """Embedding must not depend on the bag having recorded one.

    A bag with no rosetta.contract_hash at all -- recorded before this
    feature existed, with embed_contract:=false, or by plain `ros2 bag
    record` -- must still get the --contract file embedded in the dataset,
    since that's driven by --contract itself, not by anything read back
    from the bag.
    """
    writer = _StubWriter()
    _patch_common(monkeypatch, writer)
    monkeypatch.setattr(port_mod, "read_bag_contract_hash", lambda _bag_dir: "")  # nothing recorded

    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text("robot_type: x\n")

    port_mod.port(tmp_path, "repo", contract_path, embed_contract=True)

    assert writer.open_kwargs["contract_path"] == contract_path
    assert writer.open_kwargs["embed_contract"] is True


def test_matching_hash_does_not_warn(monkeypatch, tmp_path, caplog):
    writer = _StubWriter()
    _patch_common(monkeypatch, writer)

    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text("robot_type: x\n")
    import hashlib

    matching_hash = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    monkeypatch.setattr(port_mod, "read_bag_contract_hash", lambda _bag_dir: matching_hash)

    with caplog.at_level(logging.WARNING):
        port_mod.port(tmp_path, "repo", contract_path, embed_contract=True)

    assert not any("different contract" in rec.message for rec in caplog.records)
