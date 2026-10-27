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

"""Watchdog and safety-action TopicBridge tests against a real rclpy graph.

The safety path is the one part of the bridge that commands hardware when
something else has already gone wrong, so every branch is pinned here:
watchdog fire/disarm, the clock-reset and inactive gates, the zeros/hold/none
behavior matrix, and the invariants that back them (``_last_sent`` only holds
commands that reached the wire; ``reset_state`` forgets the previous episode).

Determinism notes: the watchdog is driven by rewinding ``_last_action_ns`` and
calling ``_on_watchdog()`` directly (established private-call style, see
test_topic_bridge_state). "Nothing was published" is asserted with a
sentinel frame on the same publisher -- per-publisher ordering means any
earlier (wrong) safety message would arrive before the sentinel -- never with
a wait-and-see window.
"""

import numpy as np
import pytest
from bridge_helpers import act_spec, bridge_on_graph, obs_spec, spin_until, subscribe
from sensor_msgs.msg import JointState

from rosetta.contract.schema import SafetyBehavior
from rosetta_ros.topic_bridge import WATCHDOG_PERIODS, TopicBridge

FPS = 30
TIMEOUT_NS = int(WATCHDOG_PERIODS * 1e9 / FPS)


def _rewind_past_timeout(bridge):
    assert bridge._last_action_ns is not None, "watchdog was never armed"
    bridge._last_action_ns -= 2 * TIMEOUT_NS


# -------------------- watchdog --------------------


def test_watchdog_fires_zeros_and_disarms():
    with bridge_on_graph(
        "wd_zeros", [], [act_spec("action", ["position.j1", "position.j2"], "/arm_cmd", SafetyBehavior.ZEROS)]
    ) as (
        bridge,
        _node,
        fixture_node,
        executor,
    ):
        assert bridge._watchdog_timer is not None
        received = subscribe(bridge, fixture_node, executor, ["/arm_cmd"])

        bridge.publish_frame({"action": np.array([1.0, 2.0])})
        _rewind_past_timeout(bridge)
        bridge._on_watchdog()

        assert bridge._last_action_ns is None  # disarmed until the next published frame
        assert spin_until(executor, lambda: [0.0, 0.0] in received["/arm_cmd"]), received


def test_watchdog_hold_republishes_last_sent_command():
    with bridge_on_graph(
        "wd_hold", [], [act_spec("action", ["position.j1", "position.j2"], "/arm_cmd", SafetyBehavior.HOLD)]
    ) as (
        bridge,
        _node,
        fixture_node,
        executor,
    ):
        received = subscribe(bridge, fixture_node, executor, ["/arm_cmd"])

        bridge.publish_frame({"action": np.array([1.0, 2.0])})
        _rewind_past_timeout(bridge)
        bridge._on_watchdog()

        # The real command and its held safety copy: [1, 2] exactly twice.
        assert spin_until(executor, lambda: received["/arm_cmd"].count([1.0, 2.0]) == 2), received


def test_watchdog_never_armed_never_fires():
    with bridge_on_graph("wd_unarmed", [], [act_spec("action", ["position.j1"], "/arm_cmd", SafetyBehavior.ZEROS)]) as (
        bridge,
        _node,
        fixture_node,
        executor,
    ):
        received = subscribe(bridge, fixture_node, executor, ["/arm_cmd"])

        bridge._on_watchdog()  # no action ever published: early return

        # Sentinel: anything the watchdog (wrongly) published would precede it.
        def sentinel_seen():
            bridge.publish_frame({"action": np.array([9.0])})
            return bool(received["/arm_cmd"])

        assert spin_until(executor, sentinel_seen)
        assert all(msg == [9.0] for msg in received["/arm_cmd"]), received


def test_watchdog_clock_reset_disarms_without_publishing():
    with bridge_on_graph("wd_reset", [], [act_spec("action", ["position.j1"], "/arm_cmd", SafetyBehavior.ZEROS)]) as (
        bridge,
        node,
        fixture_node,
        executor,
    ):
        received = subscribe(bridge, fixture_node, executor, ["/arm_cmd"])

        # A last-action stamp in the future = the sampling clock was reset.
        bridge._last_action_ns = node.get_clock().now().nanoseconds + int(10e9)
        bridge._on_watchdog()
        assert bridge._last_action_ns is None

        def sentinel_seen():
            bridge.publish_frame({"action": np.array([9.0])})
            return bool(received["/arm_cmd"])

        assert spin_until(executor, sentinel_seen)
        assert all(msg == [9.0] for msg in received["/arm_cmd"]), received


def test_watchdog_gated_while_node_inactive():
    with bridge_on_graph(
        "wd_inactive", [], [act_spec("action", ["position.j1"], "/arm_cmd", SafetyBehavior.ZEROS)], activate=False
    ) as (bridge, _node, _fixture_node, _executor):
        bridge._last_action_ns = 1  # ancient: would time out if the gate failed
        bridge._on_watchdog()
        assert bridge._last_action_ns == 1  # untouched: gate returned before any logic


@pytest.mark.parametrize(
    ("safeties", "expected"),
    [
        ([], False),
        ([SafetyBehavior.NONE, SafetyBehavior.NONE], False),
        ([SafetyBehavior.NONE, SafetyBehavior.ZEROS], True),
    ],
)
def test_should_use_watchdog_matrix(safeties, expected):
    specs = [act_spec("action", [f"position.j{i}"], f"/cmd_{i}", s) for i, s in enumerate(safeties)]
    assert TopicBridge([], specs, fps=FPS)._should_use_watchdog() is expected


# -------------------- send_safety_action and state invariants --------------------


def test_safety_skips_none_channels():
    specs = [
        act_spec("action", ["position.j1"], "/arm_cmd", SafetyBehavior.ZEROS),
        act_spec("action", ["position.grip"], "/grip_cmd", SafetyBehavior.NONE),
    ]
    with bridge_on_graph("sa_skip_none", [], specs) as (bridge, _node, fixture_node, executor):
        received = subscribe(bridge, fixture_node, executor, ["/arm_cmd", "/grip_cmd"])

        bridge.send_safety_action()

        # Sentinel on both channels; the NONE channel must show only it.
        def sentinels_seen():
            bridge.publish_frame({"action": np.array([9.0, 9.0])})
            return received["/arm_cmd"] and received["/grip_cmd"]

        assert spin_until(executor, sentinels_seen)
        assert [0.0] in received["/arm_cmd"], received
        assert all(msg == [9.0] for msg in received["/grip_cmd"]), received


def test_publish_frame_on_inactive_node_records_nothing():
    """Inactive lifecycle publishers silently drop -- the hold cache and the
    watchdog arm must not pretend those commands reached the wire."""
    with bridge_on_graph(
        "pf_inactive", [], [act_spec("action", ["position.j1"], "/arm_cmd", SafetyBehavior.HOLD)], activate=False
    ) as (bridge, _node, _fixture_node, _executor):
        bridge.publish_frame({"action": np.array([1.0])})
        assert bridge._last_sent == [None]
        assert bridge._last_action_ns is None


def test_reset_state_forgets_previous_episode():
    """After reset_state, safety 'hold' must not replay episode N's last
    command, and the observation buffers must need re-warming."""
    obs = [obs_spec("observation.state", ["position.j1"], "/js")]
    act = [act_spec("action", ["position.j1", "position.j2"], "/arm_cmd", SafetyBehavior.HOLD)]
    with bridge_on_graph("reset", obs, act) as (bridge, _node, fixture_node, executor):
        received = subscribe(bridge, fixture_node, executor, ["/arm_cmd"])

        js_pub = fixture_node.create_publisher(JointState, "/js", 10)
        msg = JointState()
        msg.name = ["j1"]
        msg.position = [0.5]

        def warmed():
            js_pub.publish(msg)
            return bridge.warmed_up

        assert spin_until(executor, warmed)
        bridge.publish_frame({"action": np.array([1.0, 2.0])})

        bridge.reset_state()
        assert bridge._last_sent == [None]
        assert not bridge.warmed_up  # buffers cleared: next episode re-warms

        bridge.send_safety_action()  # hold source gone -> zeros, not [1, 2]
        assert spin_until(executor, lambda: [0.0, 0.0] in received["/arm_cmd"]), received
        assert received["/arm_cmd"].count([1.0, 2.0]) == 1, received


def test_setup_twice_raises():
    with bridge_on_graph("double_setup", [], [act_spec("action", ["position.j1"], "/arm_cmd")]) as (
        bridge,
        node,
        _fixture_node,
        _executor,
    ):
        with pytest.raises(RuntimeError, match="setup"):
            bridge.setup(node)
