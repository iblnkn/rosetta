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

"""main()'s --hub-public/--hub-tags flags must reach port()'s writer_opts;
without them the writer-level hub_private/hub_tags kwargs (which do exist and
default safely) had no way to be set from the CLI at all."""

import sys

from rosetta.robots.ros2.offline import port as port_mod


def _run_main(monkeypatch, argv, captured):
    def fake_port(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(port_mod, "port", fake_port)
    monkeypatch.setattr(sys, "argv", ["port.py", *argv])
    port_mod.main()


def _base_argv(tmp_path):
    return ["--raw-dir", str(tmp_path), "--contract", str(tmp_path / "c.yaml")]


def test_hub_private_defaults_true_with_no_flag(tmp_path, monkeypatch):
    captured = {}
    _run_main(monkeypatch, _base_argv(tmp_path), captured)
    assert captured["writer_opts"]["hub_private"] is True
    assert "hub_tags" not in captured["writer_opts"]


def test_hub_public_flag_flips_hub_private_false(tmp_path, monkeypatch):
    captured = {}
    _run_main(monkeypatch, [*_base_argv(tmp_path), "--hub-public"], captured)
    assert captured["writer_opts"]["hub_private"] is False


def test_hub_tags_flag_splits_on_comma(tmp_path, monkeypatch):
    captured = {}
    _run_main(monkeypatch, [*_base_argv(tmp_path), "--hub-tags", "a, b ,c"], captured)
    assert captured["writer_opts"]["hub_tags"] == ["a", "b", "c"]
