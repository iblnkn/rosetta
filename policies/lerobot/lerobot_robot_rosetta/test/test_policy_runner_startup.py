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

"""Eager policy-server start in LeRobotPolicyRunner.setup().

The server subprocess must launch at setup (node configure) time — with
preload args derived from ROS params — so the model is warm before the first
goal. No real subprocesses or sockets: Popen and create_connection are
monkeypatched.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from lerobot_robot_rosetta import policy_runner as pr


class _FakeLogger:
    def __init__(self):
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def info(self, msg):
        self.infos.append(msg)

    def warning(self, msg):
        self.warnings.append(msg)

    def error(self, msg):
        pass


class _FakeNode:
    """Duck-typed rclpy node: dict-backed parameters."""

    def __init__(self, params):
        self._params = dict(params)
        self._logger = _FakeLogger()

    def declare_parameter(self, name, default, _descriptor=None):
        self._params.setdefault(name, default)

    def has_parameter(self, name):
        return name in self._params

    def get_parameter(self, name):
        return SimpleNamespace(value=self._params[name])

    def get_logger(self):
        return self._logger


class _FakeProc:
    def __init__(self, argv):
        self.argv = argv
        self.returncode = None
        self.poll_value = None

    def poll(self):
        return self.poll_value


@pytest.fixture
def spawned(monkeypatch):
    """Neutralize contract validation, subprocess, and socket; record spawns."""
    import lerobot_robot_rosetta.rosetta as adapter
    import rosetta.contract.specs as cu

    monkeypatch.setattr(cu, "iter_observation_specs", lambda contract: [])
    monkeypatch.setattr(cu, "iter_action_specs", lambda contract: [])
    monkeypatch.setattr(adapter, "ensure_live_observation_layout", lambda obs, act: None)

    procs: list[_FakeProc] = []

    def fake_popen(cmd, **kwargs):
        procs.append(_FakeProc(cmd))
        return procs[-1]

    monkeypatch.setattr(pr.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(pr.socket, "create_connection", lambda addr, timeout=None: nullcontext())
    return procs


def _setup_runner(param_overrides=None):
    params = {"contract_path": "/tmp/contract.yaml", "is_classifier": False}
    params.update(param_overrides or {})
    node = _FakeNode(params)
    runner = pr.LeRobotPolicyRunner()
    runner.setup(node, contract=object())
    return runner, node


def test_setup_spawns_server_with_preload_args(spawned):
    _setup_runner({"pretrained_name_or_path": "user/repo"})
    assert len(spawned) == 1
    argv = spawned[0].argv
    assert "-m" in argv and "lerobot_robot_rosetta.policy_server" in argv
    assert "--policy-type=act" in argv
    assert "--pretrained-name-or-path=user/repo" in argv
    assert "--policy-device=cuda" in argv


def test_classifier_flag_picks_classifier_module(spawned):
    _setup_runner({"pretrained_name_or_path": "user/repo", "is_classifier": True})
    assert "lerobot_robot_rosetta.classifier_server" in spawned[0].argv


def test_empty_pretrained_spawns_without_preload(spawned):
    _runner, node = _setup_runner()  # pretrained_name_or_path defaults to ''
    argv = spawned[0].argv
    assert not any(a.startswith("--policy-type") for a in argv)
    assert not any(a.startswith("--pretrained-name-or-path") for a in argv)
    assert any("load the model on the first goal" in w for w in node.get_logger().warnings)


def test_no_spawn_when_local_server_disabled(spawned):
    _setup_runner({"launch_local_server": False, "pretrained_name_or_path": "user/repo"})
    assert spawned == []


def test_server_crash_at_startup_fails_setup(spawned, monkeypatch):
    def dead_popen(cmd, **kwargs):
        proc = _FakeProc(cmd)
        proc.poll_value = 1
        proc.returncode = 1
        return proc

    monkeypatch.setattr(pr.subprocess, "Popen", dead_popen)
    with pytest.raises(RuntimeError, match="exited with code 1"):
        _setup_runner({"pretrained_name_or_path": "user/repo"})


def test_dead_server_respawns_with_same_args(spawned):
    runner, _node = _setup_runner({"pretrained_name_or_path": "user/repo"})
    spawned[0].poll_value = 1  # server died between goals

    runner._start_policy_server()  # what run() calls as self-heal
    assert len(spawned) == 2
    assert spawned[1].argv == spawned[0].argv


def test_live_server_not_respawned(spawned):
    runner, _node = _setup_runner({"pretrained_name_or_path": "user/repo"})
    runner._start_policy_server()
    assert len(spawned) == 1
