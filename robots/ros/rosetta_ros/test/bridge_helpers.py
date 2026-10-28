# Copyright 2026 Isaac Blankenau
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

"""Shared test helpers: spec builders, graph fixtures, bag writer, fakes.

Imports here stay light (no rclpy at module level) so ROS-free test modules
can use the spec builders and stub writer too; graph helpers import rclpy
lazily.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rosetta.contract.schema import Align, Channel, Contract, SafetyBehavior, Source
from rosetta.contract.specs import ActionStreamSpec, ObservationStreamSpec
from rosetta.policies.protocols import DatasetWriter

JOINT_STATE = "sensor_msgs/msg/JointState"


def obs_spec(key, names, topic, *, fps=30, align=None, qos=None, namespace=None):
    """JointState observation spec, hold on the receive timeline by default."""
    return ObservationStreamSpec(
        key=key,
        names=list(names),
        fps=fps,
        namespace=namespace,
        source=Source(
            channel=Channel(topic=topic, type=JOINT_STATE, qos=qos),
            align=align or Align("hold", "receive"),
        ),
        is_image=False,
        image_resize=None,
        dtype="float64",
    )


def act_spec(key, names, topic, safety=SafetyBehavior.NONE, *, fps=30):
    """JointState action spec with the given safety behavior."""
    return ActionStreamSpec(
        key=key,
        names=list(names),
        fps=fps,
        source=Source(
            channel=Channel(topic=topic, type=JOINT_STATE, safety=safety),
            align=Align("hold", "receive"),
        ),
        dtype="float64",
    )


class FakeLogger:
    def __init__(self):
        self.warnings: list[str] = []
        self.infos: list[str] = []
        self.errors: list[str] = []

    def warning(self, msg, **_kw):
        self.warnings.append(msg)

    def info(self, msg, **_kw):
        self.infos.append(msg)

    def error(self, msg, **_kw):
        # TopicBridge passes throttle_duration_sec here.
        self.errors.append(msg)


class FakeNode:
    """Stands in for the bridge's host node: a fixed clock and a recording logger."""

    def __init__(self, now_ns=1_000_000_000):
        self.logger = FakeLogger()
        self._clock = SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=now_ns))

    def get_clock(self):
        return self._clock

    def get_logger(self):
        return self.logger


class StubWriter:
    """DatasetWriter double that records the call sequence and open() kwargs."""

    def __init__(self):
        self.calls: list[str] = []
        self.open_kwargs: dict = {}

    def open(
        self,
        *,
        contract: Contract,
        repo_id: str,
        root: Path | None = None,
        contract_path: Path | None = None,
        embed_contract: bool = True,
        **opts: Any,
    ) -> None:
        self.calls.append("open")
        self.open_kwargs = {
            "contract": contract,
            "repo_id": repo_id,
            "root": root,
            "contract_path": contract_path,
            "embed_contract": embed_contract,
            **opts,
        }

    def add_frame(self, _frame):
        self.calls.append("add_frame")

    def save_episode(self):
        self.calls.append("save_episode")

    def discard_episode(self):
        self.calls.append("discard_episode")

    def finalize(self):
        self.calls.append("finalize")


# Fails at import, for every test that uses the stub, when DatasetWriter gains
# or renames a method the stub lacks.
assert isinstance(StubWriter(), DatasetWriter), "StubWriter no longer implements DatasetWriter"


def spin_until(executor, predicate, timeout_s=5.0):
    """Spin until predicate() is true; False on timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.05)
        if predicate():
            return True
    return False


@contextmanager
def bridge_on_graph(name, obs_specs, act_specs, *, fps=30, activate=True):
    """TopicBridge on a real LifecycleNode plus a fixture node, on one executor.

    Yields (bridge, node, fixture_node, executor). Subscriptions the test
    creates on the fixture node must be matched before publishing (see
    subscribe); teardown is unconditional.
    """
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.lifecycle import LifecycleNode

    from rosetta_ros.topic_bridge import TopicBridge

    node = LifecycleNode(f"bridge_host_{name}")
    fixture_node = rclpy.create_node(f"fixture_{name}")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(fixture_node)
    bridge = TopicBridge(obs_specs, act_specs, fps=fps)
    try:
        bridge.setup(node)
        node.trigger_configure()
        if activate:
            node.trigger_activate()
        yield bridge, node, fixture_node, executor
        bridge.teardown()
    finally:
        executor.remove_node(node)
        executor.remove_node(fixture_node)
        node.destroy_node()
        fixture_node.destroy_node()


def subscribe(bridge, fixture_node, executor, topics):
    """Collect JointState positions per topic, matched before returning.

    Waiting for get_subscription_count() on the bridge's own publishers closes
    the publish-before-matched race; afterwards, per-publisher ordering makes
    received sequences deterministic.
    """
    from sensor_msgs.msg import JointState

    received = {t: [] for t in topics}
    for t in topics:
        fixture_node.create_subscription(JointState, t, lambda m, t_=t: received[t_].append(list(m.position)), 10)
    assert spin_until(executor, lambda: all(pub.get_subscription_count() > 0 for _, pub in bridge._act_publishers)), (
        "fixture subscriptions never matched the bridge publishers"
    )
    return received


def joint_state(names, positions, stamp_ns=None):
    from sensor_msgs.msg import JointState

    msg = JointState()
    msg.name = list(names)
    msg.position = [float(p) for p in positions]
    if stamp_ns is not None:
        msg.header.stamp.sec = stamp_ns // 1_000_000_000
        msg.header.stamp.nanosec = stamp_ns % 1_000_000_000
    return msg


def _type_name(msg):
    """'sensor_msgs/msg/JointState' for a sensor_msgs.msg.JointState instance."""
    cls = type(msg)
    return f"{cls.__module__.split('.')[0]}/msg/{cls.__name__}"


def write_bag(bag_dir, events, extra_topics=None):
    """Write (topic, receive_ns, msg) events to an mcap bag.

    Topic types come from the message classes. ``extra_topics`` maps topic to
    type for topics registered with no messages.
    """
    import rosbag2_py
    from rclpy.serialization import serialize_message

    topic_types = {topic: _type_name(msg) for topic, _, msg in events}
    topic_types.update(extra_topics or {})
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    for tid, (topic, type_str) in enumerate(sorted(topic_types.items())):
        try:
            meta = rosbag2_py.TopicMetadata(id=tid, name=topic, type=type_str, serialization_format="cdr")
        except TypeError:  # pre-Jazzy signature without id
            meta = rosbag2_py.TopicMetadata(topic, type_str, "cdr")
        writer.create_topic(meta)
    for topic, ns, msg in sorted(events, key=lambda e: e[1]):
        writer.write(topic, serialize_message(msg), ns)
    del writer  # flush + write metadata.yaml
