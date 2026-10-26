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

import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from lerobot_rosetta import policy_runner as pr


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


class _FakeHost:
    """A RunnerHost backed by a dict."""

    def __init__(self, params):
        self.params = dict(params)
        self.logger = _FakeLogger()

    def declare(self, name, default, description="", read_only=False):
        self.params.setdefault(name, default)

    def get(self, name):
        return self.params[name]


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
    host = _FakeHost(params)
    runner = pr.LeRobotPolicyRunner()
    runner.setup(host, contract=object())
    return runner, host


def test_setup_spawns_server_with_preload_args(spawned):
    # policy_device=cpu: always available, so device resolution is a no-op and
    # the assertion holds on GPU-less runners (CI) and CUDA machines alike.
    _setup_runner({"pretrained_name_or_path": "user/repo", "policy_device": "cpu"})
    assert len(spawned) == 1
    argv = spawned[0].argv
    assert "-m" in argv and "lerobot_rosetta.policy_server" in argv
    assert "--policy-type=act" in argv
    assert "--pretrained-name-or-path=user/repo" in argv
    assert "--policy-device=cpu" in argv


def test_classifier_flag_picks_classifier_module(spawned):
    _setup_runner({"pretrained_name_or_path": "user/repo", "is_classifier": True})
    assert "lerobot_rosetta.classifier_server" in spawned[0].argv


def test_empty_pretrained_spawns_without_preload(spawned):
    _runner, host = _setup_runner()  # pretrained_name_or_path defaults to ''
    argv = spawned[0].argv
    assert not any(a.startswith("--policy-type") for a in argv)
    assert not any(a.startswith("--pretrained-name-or-path") for a in argv)
    assert any("load the model on the first goal" in w for w in host.logger.warnings)


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
    runner, _host = _setup_runner({"pretrained_name_or_path": "user/repo"})
    spawned[0].poll_value = 1  # server died between goals

    runner._start_policy_server()  # what run() calls as self-heal
    assert len(spawned) == 2
    assert spawned[1].argv == spawned[0].argv


def test_live_server_not_respawned(spawned):
    runner, _host = _setup_runner({"pretrained_name_or_path": "user/repo"})
    runner._start_policy_server()
    assert len(spawned) == 1


# -------------------- Sim-time pacing config --------------------


def _robot_config(fps=30):
    from lerobot_robot_rosetta.config_rosetta import RosettaConfig

    cfg = RosettaConfig(config_path="")  # no contract load
    cfg.fps = fps
    return cfg


def test_wall_clock_fps_is_contract_fps(spawned):
    runner, _host = _setup_runner({"pretrained_name_or_path": "user/repo", "policy_device": "cpu"})
    config = runner._build_client_config(_robot_config(fps=30), task="t")
    assert config.fps == 30


def test_sim_time_lifts_fps_ceiling(spawned):
    runner, host = _setup_runner({"pretrained_name_or_path": "user/repo", "policy_device": "cpu", "use_sim_time": True})
    config = runner._build_client_config(_robot_config(fps=30), task="t")
    # The ceiling is the gate's slice-resolution rate: the loop must not bind
    # before the gate itself runs out of resolution.
    assert config.fps == int(1 / pr.GATE_SLICE_SEC)
    assert config.fps > 30
    assert any("Sim time active" in msg for msg in host.logger.infos)


def test_sim_time_multiplier_gone(spawned):
    _runner, host = _setup_runner()
    assert "sim_time_multiplier" not in host.params


def test_run_injects_bridge_and_stop_event(spawned, monkeypatch):
    # run() hands the frame I/O AND the goal stop_event to the robot config;
    # the stop_event is what releases a sim-pacing gate under a paused sim.
    import lerobot_robot_rosetta.config_rosetta as cr

    monkeypatch.setattr(cr, "load_contract", lambda path: SimpleNamespace(fps=30, robot_type="bot"))
    monkeypatch.setattr(cr, "iter_observation_specs", lambda c: [])
    monkeypatch.setattr(cr, "iter_action_specs", lambda c: [])
    runner, _host = _setup_runner({"pretrained_name_or_path": "user/repo", "policy_device": "cpu"})

    captured = {}

    class _FakeClient:
        def __init__(self, config):
            captured["config"] = config
            self.shutdown_event = threading.Event()

        def start(self):
            return False  # short-circuit run() before any threads start

    monkeypatch.setattr(pr, "RobotClient", _FakeClient)

    frames = object()
    stop = threading.Event()
    result = runner.run(frames, task="t", stop_event=stop)
    assert result.success is False
    robot_config = captured["config"].robot
    assert robot_config._external_bridge is frames
    assert robot_config._stop_event is stop
