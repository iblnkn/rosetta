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

"""scan_inline_codec_paths: the pre-load audit of a hub-resolved contract must
surface every inline decoder/encoder WITHOUT importing anything -- it exists
precisely because load_contract's parse already imports those paths."""

import sys

from rosetta.contract.sidecar import scan_inline_codec_paths


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


def test_scan_imports_nothing(tmp_path):
    """The whole point: the audit must run before any import happens."""
    module_name = "definitely_not_installed_codec_module_xyz"
    contract = tmp_path / "contract.yaml"
    contract.write_text(f"observations:\n  s:\n    channel:\n      decoder: '{module_name}:fn'\n")

    paths = scan_inline_codec_paths(contract)

    assert paths == [("decoder", f"{module_name}:fn")]
    assert module_name not in sys.modules
