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

"""
Frame I/O protocols: how the robot side and the policy side meet.

A live robot, to Rosetta, is a duck-typed feed of frame dicts
(``{contract_key: np.ndarray | str}``, see :mod:`rosetta.frames` for what a
frame is): observations sampled out on the clock set by the contract's
``fps``, actions published back in. These protocols name that surface so a
:class:`~rosetta.policies.PolicyRunner` can be written and tested against
frames alone, with no dependency on any pub/sub ecosystem.

``rosetta_ros.topic_bridge.TopicBridge`` is the ROS2 implementation.
A future robot interface (ROS1, zenoh, MQTT, ...) implements the same methods,
and every policy framework works with it unchanged.

Where the robot side hosts a run (a policy runner node), it builds the
``FrameIO`` and injects it. Where a framework hosts the run itself (LeRobot's
CLIs driving a ``Robot``), it opens a :class:`Connection` through the
contract's robot interface (:meth:`~rosetta.robots.protocols.RobotInterface.connect`).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Connection",
    "FrameIO",
    "RobotClock",
]


@runtime_checkable
class RobotClock(Protocol):
    """The clock a live robot runs on: wall time, or a simulator's.

    A framework that paces its own loop waits on this clock, so a simulation
    running slower or faster than real time is still sampled at the
    contract's ``fps`` in simulated time.
    """

    @property
    def simulated(self) -> bool:
        """True when the robot runs on a simulator's clock rather than wall time."""
        ...

    def now_ns(self) -> int:
        """Current time on this clock, in nanoseconds."""
        ...


@runtime_checkable
class FrameIO(Protocol):
    """Bidirectional frame I/O with a live robot.

    What a :class:`~rosetta.policies.PolicyRunner` receives per goal: sample
    observations out, publish actions in, reset between episodes.

    This is the policy-facing surface only. A concrete implementation may
    expose richer robot-side methods beyond the protocol (e.g. TopicBridge's
    ``sample_values()``, the pre-assembly per-spec view); those never cross
    the robot/policy boundary.

    Once the robot is gone (its transport stopped, or the I/O was closed),
    every method and the clock raise. They never serve the last data as if
    it were current.
    """

    def sample_frame(self, *, fill_missing: bool = True) -> dict[str, Any]:
        """Current observation frame: ``{contract_key: np.ndarray | str}``.

        A stream with no data yet zero-fills, so every key is present. With
        ``fill_missing=False`` a key with any such stream is left out instead,
        for consumers that would act on a made-up zero (a teleop input drives
        motion).
        """
        ...

    @property
    def clock(self) -> RobotClock:
        """The clock the robot runs on."""
        ...

    @property
    def warmed_up(self) -> bool:
        """True once every observation stream has delivered at least one value."""
        ...

    def publish_frame(self, action_frame: dict[str, Any]) -> dict[str, Any]:
        """Publish one action frame (``{contract_key: np.ndarray}``). Returns it."""
        ...

    def send_safety_action(self) -> None:
        """Publish the contract's safety action (e.g. zero velocity)."""
        ...

    def reset_state(self) -> None:
        """Clear per-episode state (buffers, warmup) between runs."""
        ...


@runtime_checkable
class Connection(FrameIO, Protocol):
    """A live robot a framework opened itself, through its robot interface.

    See :meth:`~rosetta.robots.protocols.RobotInterface.connect`. The caller
    owns it and closes it.
    """

    def events(self) -> dict[str, bool]:
        """Latest value of each event in the event map the connection was opened with.

        Keyed by the contract's event names; empty without an event map or
        before the first event message.
        """
        ...

    def close(self) -> None:
        """Stop sampling and publishing and release the middleware. Idempotent.

        A close that raises has still released what it could, so the caller
        drops the connection either way.
        """
        ...
