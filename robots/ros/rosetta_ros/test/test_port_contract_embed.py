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

"""Contract embedding: open-kwargs threading and the contract-drift heads-up.

write_dataset must thread contract_path/embed_contract to the writer, and
warn_if_contract_mismatch must warn when a bag's embedded contract differs from
--contract without ever failing the port (--contract always wins for decoding).

No monkeypatching: write_dataset takes its writer as an argument, and
warn_if_contract_mismatch takes the bag's embedded contract as a value.
"""

import logging

import pytest
from bridge_helpers import StubWriter

from rosetta_ros.offline.port import warn_if_contract_mismatch, write_dataset


def _one_episode():
    return [("ep0", iter([{"observation.state": [0.0]}]))]


@pytest.mark.parametrize(
    ("embed_kwargs", "expected"),
    [({}, True), ({"embed_contract": False}, False)],
    ids=["default", "no-embed-contract"],
)
def test_writer_receives_contract_path_and_embed_flag(tmp_path, embed_kwargs, expected):
    writer = StubWriter()
    contract_path = tmp_path / "contract.yaml"

    write_dataset(writer, _one_episode(), contract=None, repo_id="repo", contract_path=contract_path, **embed_kwargs)

    assert writer.open_kwargs["contract_path"] == contract_path
    assert writer.open_kwargs["embed_contract"] is expected


# ---------------------------------------------------------------------------
# contract-drift heads-up (a warning, never a failure)
# ---------------------------------------------------------------------------


def test_different_contract_warns(tmp_path, caplog):
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text("robot_type: x\n")

    with caplog.at_level(logging.WARNING):
        warn_if_contract_mismatch(contract_path, "robot_type: y\n")

    assert any("different contract" in rec.message for rec in caplog.records)


@pytest.mark.parametrize(
    ("contract_file", "embedded"),
    [
        ("contract.yaml", "robot_type: x\n"),  # identical
        ("contract.yaml", "robot_type:   x\n\n"),  # compared as parsed YAML, not text
        ("contract.yaml", ""),  # plain `ros2 bag record` or embed_contract:=false: nothing to compare
        ("missing.yaml", "robot_type: x\n"),  # unreadable --contract: skip, never fail the port
    ],
    ids=["matching", "whitespace", "no-embedded", "unreadable-path"],
)
def test_no_warning_or_error(tmp_path, caplog, contract_file, embedded):
    (tmp_path / "contract.yaml").write_text("robot_type: x\n")

    with caplog.at_level(logging.WARNING):
        warn_if_contract_mismatch(tmp_path / contract_file, embedded)

    assert not caplog.records
