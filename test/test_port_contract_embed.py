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

"""Contract embedding: open-kwargs threading and the bag-hash heads-up.

write_dataset must thread contract_path/embed_contract to the writer, and
warn_if_contract_mismatch must warn on a bag/--contract hash mismatch without
ever failing the port (--contract always wins for decoding).

No monkeypatching: write_dataset takes its writer as an argument, and
warn_if_contract_mismatch takes the bag hash as a value.
"""

import hashlib
import logging

from rosetta.robots.ros2.offline.port import warn_if_contract_mismatch, write_dataset


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


def _one_episode():
    return [("ep0", iter([{"observation.state": [0.0]}]))]


# ---------------------------------------------------------------------------
# open-kwargs threading
# ---------------------------------------------------------------------------


def test_writer_receives_contract_path_and_embed_flag(tmp_path):
    writer = _StubWriter()
    contract_path = tmp_path / "contract.yaml"

    write_dataset(writer, _one_episode(), contract=None, repo_id="repo", contract_path=contract_path)

    assert writer.open_kwargs["contract_path"] == contract_path
    assert writer.open_kwargs["embed_contract"] is True  # default


def test_no_embed_contract_flag_propagates(tmp_path):
    writer = _StubWriter()

    write_dataset(
        writer,
        _one_episode(),
        contract=None,
        repo_id="repo",
        contract_path=tmp_path / "contract.yaml",
        embed_contract=False,
    )

    assert writer.open_kwargs["embed_contract"] is False


# ---------------------------------------------------------------------------
# bag-hash heads-up (a warning, never a failure)
# ---------------------------------------------------------------------------


def test_hash_mismatch_warns(tmp_path, caplog):
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text("robot_type: x\n")

    with caplog.at_level(logging.WARNING):
        warn_if_contract_mismatch(contract_path, "stale-hash")

    assert any("different contract" in rec.message for rec in caplog.records)


def test_matching_hash_does_not_warn(tmp_path, caplog):
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text("robot_type: x\n")
    matching_hash = hashlib.sha256(contract_path.read_bytes()).hexdigest()

    with caplog.at_level(logging.WARNING):
        warn_if_contract_mismatch(contract_path, matching_hash)

    assert not any("different contract" in rec.message for rec in caplog.records)


def test_bag_without_recorded_hash_does_not_warn(tmp_path, caplog):
    """A bag with no rosetta.contract_hash (pre-feature recording, plain
    `ros2 bag record`, embed_contract:=false) skips the comparison; embedding
    is driven by --contract itself, not by anything read back from the bag."""
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text("robot_type: x\n")

    with caplog.at_level(logging.WARNING):
        warn_if_contract_mismatch(contract_path, "")

    assert not caplog.records


def test_unreadable_contract_path_does_not_warn_or_raise(tmp_path, caplog):
    """The check must never be why a port fails: --contract already loaded
    fine upstream, so a raw re-read failing here silently skips."""
    with caplog.at_level(logging.WARNING):
        warn_if_contract_mismatch(tmp_path / "missing.yaml", "any-hash")

    assert not caplog.records
