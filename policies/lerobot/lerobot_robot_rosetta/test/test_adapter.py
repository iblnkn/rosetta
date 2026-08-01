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

"""Round-trip tests for the Rosetta(Robot) flatten/unflatten adapter,
plus injected-mode connect/pacing behavior against a fake bridge."""

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ament_index_python.packages import get_package_share_directory

import lerobot_robot_rosetta.rosetta as adapter
import rosetta.robots.ros2.offline.bag_frames  # noqa: F401  register decoders
from lerobot_robot_rosetta.config_rosetta import RosettaConfig
from lerobot_robot_rosetta.rosetta import Rosetta, _TickGate

CONTRACTS = Path(get_package_share_directory("rosetta")) / "contracts"


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
    """FrameIO double with the robot-side clock accessor TopicBridge grows."""

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


@pytest.mark.parametrize("contract", ["so_101.yaml", "turtlebot3.yaml"])
def test_action_unflatten_roundtrip(contract):
    robot = _robot(contract)
    # Every action key's namespaced names, in order.
    for key, names in robot._act_vector_names.items():
        vec = np.arange(len(names), dtype=np.float32) + 0.5
        flat = {name: float(vec[i]) for i, name in enumerate(names)}
        out = robot._unflatten_action(flat)
        assert key in out
        assert np.allclose(out[key], vec), (key, out[key], vec)


def test_observation_flatten_state_and_images():
    robot = _robot("so_101.yaml")
    names = robot._obs_vector_names["observation.state"]
    vec = np.arange(len(names), dtype=np.float32) + 1.0

    frame = {"observation.state": vec}
    for full_key in robot._obs_image_keys:
        frame[full_key] = np.zeros((4, 4, 3), dtype=np.uint8)

    obs = robot._flatten_observation(frame)

    # state floats land under their namespaced names, in order
    for i, name in enumerate(names):
        assert obs[name] == float(vec[i])
    # images pass through under their short keys
    for full_key, short in robot._obs_image_keys.items():
        assert short in obs
        assert obs[short] is frame[full_key]


@pytest.mark.parametrize("contract", ["so_101.yaml", "turtlebot3.yaml"])
def test_adapter_names_agree_with_frame_layout(contract):
    # The adapter's per-key name order must match the canonical FrameLayout
    # (key-major, declaration order within key) that assembles the frames it
    # flattens — otherwise live features and frame data disagree.
    from rosetta.contract.schema import load_contract
    from rosetta.contract.specs import iter_action_specs, iter_observation_specs
    from rosetta.frames.layout import FrameLayout

    robot = _robot(contract)
    c = load_contract(CONTRACTS / contract)
    specs = list(iter_observation_specs(c)) + list(iter_action_specs(c))
    feats = FrameLayout(specs).lerobot_features()
    for key, names in robot._obs_vector_names.items():
        assert feats[key]["names"] == names, key
    for key, names in robot._act_vector_names.items():
        assert feats[key]["names"] == names, key


def test_namespaced_multi_spec_state(contract="turtlebot3.yaml"):
    # turtlebot3 aggregates 3 specs under observation.state -> namespaced names.
    robot = _robot(contract)
    names = robot._obs_vector_names["observation.state"]
    # names carry a namespace prefix (derived from the 3 topics)
    assert any("." in n for n in names)
    vec = np.arange(len(names), dtype=np.float32)
    frame = {"observation.state": vec}
    for full_key in robot._obs_image_keys:
        frame[full_key] = np.zeros((4, 4, 3), dtype=np.uint8)
    obs = robot._flatten_observation(frame)
    reconstructed = np.array([obs[n] for n in names], dtype=np.float32)
    assert np.allclose(reconstructed, vec)


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


def test_get_observation_unpaced_on_wall_clock(monkeypatch):
    robot = _injected_robot(_FakeClock(active=False))
    monkeypatch.setattr(adapter.time, "sleep", lambda sec: pytest.fail("gate slept under wall clock"))
    robot.get_observation()
    robot.get_observation()


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


def test_reset_clears_gate_anchors():
    robot = _injected_robot(_FakeClock(active=True, t=0))
    robot.get_observation()
    assert robot._obs_gate._last_ns is not None
    robot.reset()
    assert robot._obs_gate._last_ns is None
    assert robot._act_gate._last_ns is None


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
