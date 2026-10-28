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

"""A node, publisher and control switch that let a CommandForwarder run without rclpy spinning."""

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from ament_index_python.packages import get_package_share_directory

from rosetta_ros_hil.forwarding import CommandForwarder
from rosetta_ros_hil.hil_episode import Control

STONE = Path(get_package_share_directory("rosetta_ros")) / "contracts" / "stone.yaml"
SO_101_HIL = Path(__file__).resolve().parent.parent / "contracts" / "so_101_hil.yaml"
STAMP_NS = 1_500_000_123


class FakePublisher:
    """Collects what is published; like rclpy's, raises TypeError on a message of another type."""

    def __init__(self, msg_cls, published=None):
        self.msg_cls = msg_cls
        self.published = [] if published is None else published

    def publish(self, msg):
        if not isinstance(msg, self.msg_cls):
            raise TypeError(f"Expected {self.msg_cls}, got {type(msg)}")
        self.published.append(msg)


class FakeNode:
    """Records publishers and subscription callbacks by topic; the clock reads STAMP_NS."""

    def __init__(self):
        self.published = defaultdict(list)
        self.callbacks = defaultdict(list)
        self.destroyed = 0

    def create_publisher(self, msg_cls, topic, qos):
        return FakePublisher(msg_cls, self.published[topic])

    def create_subscription(self, msg_cls, topic, callback, qos, callback_group=None):
        self.callbacks[topic].append(callback)
        return topic

    def destroy_publisher(self, pub):
        self.destroyed += 1

    def destroy_subscription(self, sub):
        self.destroyed += 1

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=STAMP_NS))

    def get_logger(self):
        return SimpleNamespace(info=lambda *_, **__: None, error=lambda *_, **__: None)

    def deliver(self, topic, msg):
        for callback in self.callbacks[topic]:
            callback(msg)


class Controls:
    """Who has control, switchable from the test."""

    def __init__(self):
        self.now = Control.POLICY

    def __call__(self):
        return self.now


def forwarder(contract):
    """A CommandForwarder for ``contract`` on a FakeNode; returns the node and its control switch."""
    node, control = FakeNode(), Controls()
    CommandForwarder(node, contract, "/hil/policy", control, callback_group=None)
    return node, control
