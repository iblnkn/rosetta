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

"""The Rosetta(Robot) adapter in injected mode against a fake bridge: observation/action flattening and sim pacing."""

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ament_index_python.packages import get_package_share_directory

import lerobot_robot_rosetta.rosetta as adapter
from lerobot_robot_rosetta.config_rosetta import RosettaConfig
from lerobot_robot_rosetta.rosetta import Rosetta
from rosetta.frames.codecs import discover_codecs
from rosetta_ros.topic_bridge import TopicBridge

discover_codecs()  # register the built-in codecs

CONTRACTS = Path(get_package_share_directory("rosetta_ros")) / "contracts"


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
    """Connect a robot to a fake bridge holding a frame the contract's layout can flatten."""
    cfg = RosettaConfig(config_path=str(CONTRACTS / contract))
    bridge = _FakeBridge(clock)
    cfg._external_bridge = bridge
    cfg._stop_event = stop_event
    robot = Rosetta(cfg)
    robot.connect()
    state_names = [name for name, ft in robot.observation_features.items() if ft is float]
    bridge.frame = {"observation.state": np.arange(len(state_names), dtype=np.float32) + 1.0}
    for short, shape in robot.observation_features.items():
        if shape is not float:
            bridge.frame[f"observation.images.{short}"] = np.zeros(shape, dtype=np.uint8)
    return robot, bridge


def _zero_action(robot):
    return {name: 0.0 for name in robot.action_features}


def test_fake_bridge_only_stands_in_for_what_topic_bridge_has():
    stand_ins = [name for name in vars(_FakeBridge) if not name.startswith("_")]
    assert [name for name in stand_ins if not hasattr(TopicBridge, name)] == []


@pytest.mark.parametrize("contract", ["so_101.yaml", "turtlebot3.yaml"])
def test_send_action_publishes_named_floats_as_one_vector_in_order(contract):
    robot, bridge = _injected_robot(_FakeClock(active=False), contract=contract)
    names = list(robot.action_features)
    vec = np.arange(len(names), dtype=np.float32) + 0.5
    robot.send_action({name: float(vec[i]) for i, name in enumerate(names)})
    assert list(bridge.published[-1]) == ["action"]
    assert np.allclose(bridge.published[-1]["action"], vec)


@pytest.mark.parametrize("contract", ["so_101.yaml", "turtlebot3.yaml"])
def test_get_observation_flattens_state_and_passes_images_through(contract):
    robot, bridge = _injected_robot(_FakeClock(active=False), contract=contract)
    obs = robot.get_observation()

    # state floats land under their namespaced names, in order
    state_names = [name for name, ft in robot.observation_features.items() if ft is float]
    assert [obs[name] for name in state_names] == pytest.approx(list(bridge.frame["observation.state"]))
    # images pass through under their short keys
    image_keys = [short for short, ft in robot.observation_features.items() if ft is not float]
    assert image_keys
    for short in image_keys:
        assert obs[short] is bridge.frame[f"observation.images.{short}"]


# -------------------- Injected-mode sim pacing --------------------


@pytest.mark.parametrize("step_fraction", [0.1, 0.3, 2.5])  # slow, moderate, fast sim
def test_get_observation_paces_on_sim_clock(monkeypatch, step_fraction):
    clock = _FakeClock(active=True, t=0)
    robot, _bridge = _injected_robot(clock)
    period_ns = int(1e9 / robot.config.fps)
    step_ns = int(period_ns * step_fraction)
    sleeps = []

    def _sleep(sec):
        sleeps.append(sec)
        assert len(sleeps) < 1000, "gate never released"
        clock.t += step_ns

    monkeypatch.setattr(adapter.time, "sleep", _sleep)
    robot.get_observation()  # first call anchors, no wait
    assert sleeps == []
    robot.get_observation()  # second call waits one contract period of sim time
    assert period_ns <= clock.t < period_ns + step_ns  # released on the first slice past the period


def test_stop_event_releases_gated_calls_on_paused_sim():
    stop = threading.Event()
    stop.set()
    robot, _bridge = _injected_robot(_FakeClock(active=True, t=0), stop_event=stop)  # frozen clock

    def _gated_calls():
        robot.get_observation()  # anchors
        robot.get_observation()  # would block forever without the stop escape
        robot.send_action(_zero_action(robot))  # act gate anchors
        robot.send_action(_zero_action(robot))  # same escape on the action path

    worker = threading.Thread(target=_gated_calls, daemon=True)
    worker.start()
    worker.join(timeout=5.0)
    assert not worker.is_alive(), "a gated call stayed blocked after stop was requested"


def test_reset_makes_the_next_tick_immediate(monkeypatch):
    """A new episode starts on its first frame instead of waiting out the last episode's period.

    Also pins that observations and actions pace on separate gates: a shared
    gate would make send_action wait right after get_observation.
    """
    clock = _FakeClock(active=True, t=0)
    robot, _bridge = _injected_robot(clock)
    sleeps = []

    def _sleep(sec):
        sleeps.append(sec)
        clock.t += int(1e9 / robot.config.fps)

    monkeypatch.setattr(adapter.time, "sleep", _sleep)
    robot.get_observation()
    robot.send_action(_zero_action(robot))
    robot.reset()
    robot.get_observation()
    robot.send_action(_zero_action(robot))
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
