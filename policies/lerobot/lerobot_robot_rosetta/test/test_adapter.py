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

"""The Rosetta(Robot) flatten/unflatten adapter, plus injected-mode connect/pacing behavior against a fake bridge."""

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ament_index_python.packages import get_package_share_directory

import lerobot_robot_rosetta.rosetta as adapter
from lerobot_robot_rosetta.config_rosetta import RosettaConfig
from lerobot_robot_rosetta.rosetta import Rosetta, _TickGate
from rosetta.frames.codecs import discover_codecs
from rosetta_ros.topic_bridge import TopicBridge

discover_codecs()  # register the built-in codecs

CONTRACTS = Path(get_package_share_directory("rosetta_ros")) / "contracts"


def _robot(contract_name):
    cfg = RosettaConfig(config_path=str(CONTRACTS / contract_name))
    return Rosetta(cfg)


class _FakeClock:
    def __init__(self, active=True, t=0):
        self.ros_time_is_active = active
        self.t = t

    def now(self):
        return SimpleNamespace(nanoseconds=self.t)


class _FakeBridge:
    """Stands in for the TopicBridge a policy runner injects; frame and published are test-side only."""

    warmed_up = True

    def __init__(self, clock):
        self._clock = clock
        self.frame = {}
        self.published = []

    @property
    def clock(self):
        return self._clock

    def sample_frame(self):
        return self.frame

    def publish_frame(self, action_frame):
        self.published.append(action_frame)

    def send_safety_action(self):
        pass

    def reset_state(self):
        pass


def _injected_robot(clock, stop_event=None, contract="so_101.yaml"):
    cfg = RosettaConfig(config_path=str(CONTRACTS / contract))
    cfg._external_bridge = _FakeBridge(clock)
    cfg._stop_event = stop_event
    robot = Rosetta(cfg)
    robot.connect()
    # A frame the contract's layout can flatten.
    bridge = robot._bridge
    names = robot._obs_vector_names["observation.state"]
    bridge.frame = {"observation.state": np.arange(len(names), dtype=np.float32)}
    for full_key in robot._obs_image_keys:
        bridge.frame[full_key] = np.zeros((4, 4, 3), dtype=np.uint8)
    return robot


def test_fake_bridge_only_stands_in_for_what_topic_bridge_has():
    stand_ins = [name for name in vars(_FakeBridge) if not name.startswith("_")]
    assert [name for name in stand_ins if not hasattr(TopicBridge, name)] == []


@pytest.mark.parametrize("contract", ["so_101.yaml", "turtlebot3.yaml"])
def test_action_unflatten(contract):
    robot = _robot(contract)
    # Every action key's namespaced names, in order.
    for key, names in robot._act_vector_names.items():
        vec = np.arange(len(names), dtype=np.float32) + 0.5
        flat = {name: float(vec[i]) for i, name in enumerate(names)}
        out = robot._unflatten_action(flat)
        assert key in out
        assert np.allclose(out[key], vec), (key, out[key], vec)


@pytest.mark.parametrize("contract", ["so_101.yaml", "turtlebot3.yaml"])
def test_observation_flatten_state_and_images(contract):
    robot = _robot(contract)
    names = robot._obs_vector_names["observation.state"]
    vec = np.arange(len(names), dtype=np.float32) + 1.0

    frame = {"observation.state": vec}
    for full_key in robot._obs_image_keys:
        frame[full_key] = np.zeros((4, 4, 3), dtype=np.uint8)

    obs = robot._flatten_observation(frame)

    # state floats land under their namespaced names, in order
    assert [obs[name] for name in names] == pytest.approx(list(vec))
    # images pass through under their short keys
    for full_key, short in robot._obs_image_keys.items():
        assert obs[short] is frame[full_key]


# -------------------- Injected-mode connect / sim pacing --------------------


def test_connect_builds_gates_from_bridge_clock():
    clock = _FakeClock()
    robot = _injected_robot(clock)
    assert isinstance(robot._obs_gate, _TickGate)
    assert isinstance(robot._act_gate, _TickGate)
    assert robot._obs_gate is not robot._act_gate  # shared gate would halve the rate
    assert robot._obs_gate._clock is clock
    assert robot._obs_gate._period_ns == int(1e9 / robot.config.fps)


def test_get_observation_paces_on_sim_clock(monkeypatch):
    clock = _FakeClock(active=True, t=0)
    robot = _injected_robot(clock)
    period_ns = robot._obs_gate._period_ns
    sleeps = []

    def _sleep(sec):
        sleeps.append(sec)
        clock.t += period_ns // 4

    monkeypatch.setattr(adapter.time, "sleep", _sleep)
    robot.get_observation()  # first call anchors, no wait
    assert sleeps == []
    robot.get_observation()  # second call waits one contract period of sim time
    assert len(sleeps) > 0
    assert clock.t >= period_ns


def test_stop_event_prevents_hang_on_paused_sim():
    stop = threading.Event()
    stop.set()
    robot = _injected_robot(_FakeClock(active=True, t=0), stop_event=stop)  # frozen clock
    robot.get_observation()  # anchors
    robot.get_observation()  # would block forever without the stop escape
    action = {name: 0.0 for names in robot._act_vector_names.values() for name in names}
    assert action  # sanity: contract has actions
    robot.send_action(action)  # act gate anchors
    robot.send_action(action)  # same escape on the action path


def test_reset_makes_the_next_tick_immediate(monkeypatch):
    """A new episode starts on its first frame instead of waiting out the last episode's period."""
    clock = _FakeClock(active=True, t=0)
    robot = _injected_robot(clock)
    sleeps = []

    def _sleep(sec):
        sleeps.append(sec)
        clock.t += robot._obs_gate._period_ns

    monkeypatch.setattr(adapter.time, "sleep", _sleep)
    action = {name: 0.0 for names in robot._act_vector_names.values() for name in names}
    robot.get_observation()
    robot.send_action(action)
    robot.reset()
    robot.get_observation()
    robot.send_action(action)
    assert sleeps == []


@pytest.mark.parametrize("use_sim_time", [False, True])
def test_standalone_node_gets_use_sim_time_override(monkeypatch, use_sim_time):
    cfg = RosettaConfig(config_path=str(CONTRACTS / "so_101.yaml"), use_sim_time=use_sim_time)
    robot = Rosetta(cfg)
    captured = {}

    class _CaptureNode:
        def __init__(self, name, obs_specs, act_specs, fps, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(adapter, "BridgeLifecycleNode", _CaptureNode)
    robot._host = SimpleNamespace(start=lambda make_node: make_node(None))
    robot._create_node()

    overrides = captured["parameter_overrides"]
    if use_sim_time:
        assert len(overrides) == 1
        assert overrides[0].name == "use_sim_time"
        assert overrides[0].value is True
    else:
        assert overrides is None
