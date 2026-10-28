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
monkeypatched. The contract is the bundled so_101.yaml.
"""

import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from ament_index_python.packages import get_package_share_directory

from lerobot_rosetta import policy_runner as pr
from rosetta.contract.schema import load_contract
from rosetta.frames.codecs import discover_codecs

discover_codecs()  # register the built-in codecs

CONTRACT = Path(get_package_share_directory("rosetta_ros")) / "contracts" / "so_101.yaml"


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
def spawned(monkeypatch, tmp_path):
    """Neutralize subprocess and socket; record spawns."""
    # Each spawn opens a server log with delete=False; keep them in tmp_path.
    monkeypatch.setattr(pr.tempfile, "tempdir", str(tmp_path))
    procs: list[_FakeProc] = []

    def fake_popen(cmd, **kwargs):
        procs.append(_FakeProc(cmd))
        return procs[-1]

    monkeypatch.setattr(pr.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(pr.socket, "create_connection", lambda addr, timeout=None: nullcontext())
    return procs


@pytest.fixture(autouse=True)
def checkpoint_types(monkeypatch):
    """Checkpoint path -> the type its config.json records; a path not listed has no readable config."""
    types = {"user/repo": "act"}

    def from_pretrained(path):
        if path not in types:
            raise FileNotFoundError(f"config.json not found in {path}")
        return SimpleNamespace(type=types[path])

    monkeypatch.setattr(pr, "PreTrainedConfig", SimpleNamespace(from_pretrained=from_pretrained))
    return types


def _setup_runner(param_overrides=None):
    params = {"contract_path": str(CONTRACT), "is_classifier": False}
    params.update(param_overrides or {})
    host = _FakeHost(params)
    runner = pr.LeRobotPolicyRunner()
    runner.setup(host, contract=load_contract(CONTRACT))
    return runner, host


def test_setup_spawns_server_with_preload_args(spawned):
    # The requested device goes through as-is, even on a robot without CUDA:
    # the server resolves it on the host that runs the model.
    _setup_runner({"pretrained_name_or_path": "user/repo", "policy_device": "cuda"})
    assert len(spawned) == 1
    argv = spawned[0].argv
    assert "-m" in argv and "lerobot_rosetta.policy_server" in argv
    assert "--pretrained-name-or-path=user/repo" in argv
    assert "--policy-device=cuda" in argv


def test_classifier_flag_picks_classifier_module(spawned):
    _setup_runner({"pretrained_name_or_path": "user/repo", "is_classifier": True})
    assert "lerobot_rosetta.classifier_server" in spawned[0].argv


def test_empty_pretrained_spawns_without_preload(spawned):
    _runner, host = _setup_runner()  # pretrained_name_or_path defaults to ''
    argv = spawned[0].argv
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


@pytest.mark.parametrize(("poll_value", "spawns"), [(1, 2), (None, 1)], ids=["dead", "live"])
def test_only_a_dead_server_respawns_with_same_args(spawned, poll_value, spawns):
    runner, _host = _setup_runner({"pretrained_name_or_path": "user/repo"})
    spawned[0].poll_value = poll_value  # 1: server died between goals

    runner._start_policy_server()  # what run() calls as self-heal
    assert len(spawned) == spawns
    assert all(proc.argv == spawned[0].argv for proc in spawned)


# -------------------- Sim-time pacing config --------------------


def _robot_config(fps=30):
    from lerobot_robot_rosetta.config_rosetta import RosettaConfig

    cfg = RosettaConfig(config_path="")  # no contract load
    cfg.fps = fps
    return cfg


# Under sim time the ceiling is the gate's slice-resolution rate: the loop
# must not bind before the gate itself runs out of resolution.
@pytest.mark.parametrize(
    ("use_sim_time", "fps"), [(False, 30), (True, int(1 / pr.GATE_SLICE_SEC))], ids=["wall_clock", "sim_time"]
)
def test_client_fps_is_contract_fps_unless_sim_time_lifts_it(spawned, use_sim_time, fps):
    runner, _host = _setup_runner(
        {"pretrained_name_or_path": "user/repo", "policy_device": "cpu", "use_sim_time": use_sim_time}
    )
    config = runner._build_client_config(_robot_config(fps=30), task="t")
    assert config.fps == fps


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"pretrained_name_or_path": "user/repo"}, "act"),
        ({"pretrained_name_or_path": "user/repo", "policy_type": "diffusion"}, "diffusion"),
        ({"pretrained_name_or_path": "user/classifier", "is_classifier": True}, "reward_classifier"),
    ],
    ids=["from_checkpoint", "parameter_wins", "classifier"],
)
def test_client_config_carries_the_policy_type(spawned, overrides, expected):
    runner, _host = _setup_runner(overrides)
    assert runner._build_client_config(_robot_config(), task="t").policy_type == expected


def test_unreadable_checkpoint_fails_setup_naming_policy_type(spawned):
    # A checkpoint only on a remote policy server: the robot can't read its type.
    with pytest.raises(RuntimeError, match="set policy_type"):
        _setup_runner({"pretrained_name_or_path": "/gpu_host/only/pretrained_model"})


def test_client_config_forwards_the_requested_device(spawned):
    runner, _host = _setup_runner({"pretrained_name_or_path": "user/repo", "policy_device": "cuda"})
    assert runner._build_client_config(_robot_config(), task="t").policy_device == "cuda"


def test_run_injects_bridge_and_stop_event(spawned, monkeypatch):
    # run() hands the frame I/O AND the goal stop_event to the robot config;
    # the stop_event is what releases a sim-pacing gate under a paused sim.
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


def test_run_uses_the_hosts_contract_when_contract_path_is_empty(spawned, monkeypatch):
    # The node falls back to the checkpoint's contract when contract_path is
    # empty; the runner must use the contract it was handed, not the parameter.
    contract = load_contract(CONTRACT)
    host = _FakeHost(
        {"contract_path": "", "is_classifier": False, "pretrained_name_or_path": "user/repo", "policy_device": "cpu"}
    )
    runner = pr.LeRobotPolicyRunner()
    runner.setup(host, contract=contract)

    captured = {}

    class _FakeClient:
        def __init__(self, config):
            captured["config"] = config
            self.shutdown_event = threading.Event()

        def start(self):
            return False

    monkeypatch.setattr(pr, "RobotClient", _FakeClient)

    runner.run(object(), task="t", stop_event=threading.Event())
    robot_config = captured["config"].robot
    assert robot_config.contract is contract
    assert robot_config.fps == contract.fps


@pytest.mark.parametrize(("stop_requested", "message"), [(True, "Stopped"), (False, "Completed")])
def test_a_stopped_run_succeeds_like_a_completed_one(spawned, monkeypatch, stop_requested, message):
    # RunnerResult: a host-requested stop is a successful run. Reporting it as
    # a failure makes the node end every stopped goal with reason "error".
    runner, _host = _setup_runner({"pretrained_name_or_path": "user/repo", "policy_device": "cpu"})
    stop = threading.Event()

    class _FakeClient:
        def __init__(self, config):
            self.shutdown_event = threading.Event()

        def start(self):
            return True

        def receive_actions(self):
            pass

        def control_loop(self, task):
            if stop_requested:
                stop.set()
                # The runner forwards the host's stop to LeRobot's shutdown_event.
                assert self.shutdown_event.wait(timeout=5.0)

        def stop(self):
            pass

    monkeypatch.setattr(pr, "RobotClient", _FakeClient)

    result = runner.run(object(), task="t", stop_event=stop)
    assert (result.success, result.message) == (True, message)
