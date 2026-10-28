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

"""The contract-drift heads-up: a warning, never a failure.

warn_if_contract_mismatch must warn when a bag's embedded contract differs from
--contract without ever failing the port (--contract always wins for decoding).
That --contract and --no-embed-contract reach the writer is
test_port_cli_hub_opts's job.
"""

import logging

import pytest

from rosetta_ros.offline.port import warn_if_contract_mismatch


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
