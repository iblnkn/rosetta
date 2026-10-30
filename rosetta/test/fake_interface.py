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

"""A robot interface with no middleware behind it, for testing core alone.

``fake/Vector`` carries named float fields, read through ``select``.
``fake/Data`` carries one float32 ``data`` payload and needs no ``select``.
``fake/Text`` carries a string and only decodes. Every type provides one
timeline, ``arrival``; qos is ``{level: 0|1|2}``. ``connect`` opens a
:class:`FakeConnection`, which a test feeds by hand. The ``fake_interface``
fixture in conftest.py installs it as ``fake``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rosetta.frames.layout import FrameLayout

VECTOR = "fake/Vector"
DATA = "fake/Data"
TEXT = "fake/Text"


@dataclass
class FakeVector:
    """A message: named float fields."""

    fields: dict[str, float] = field(default_factory=dict)


@dataclass
class FakeData:
    """A message: one ``data`` payload, a float or a list of floats."""

    data: Any


@dataclass
class FakeClock:
    """A clock a test sets by hand."""

    simulated: bool = False
    ns: int = 0

    def now_ns(self) -> int:
        return self.ns


class FakeConnection:
    """:class:`~rosetta.frames.protocols.Connection` with no middleware.

    A test sets the latest value per observation spec with :meth:`push` and
    per event with :meth:`set_event`; published frames collect in ``published``.
    """

    def __init__(self, observations: Sequence[Any], events: Any, simulated: bool) -> None:
        self._layout = FrameLayout(observations)
        self._values: list[Any] = [None] * len(observations)
        self._event_names = set(events.select) if events is not None else set()
        self._events: dict[str, bool] = {}
        self.clock = FakeClock(simulated=simulated)
        self.published: list[dict[str, Any]] = []
        self.safety_actions = 0
        self.closed = False

    def push(self, index: int, value: Any) -> None:
        self._values[index] = value

    def set_event(self, name: str, value: bool) -> None:
        assert name in self._event_names, name
        self._events[name] = value

    def sample_frame(self, *, fill_missing: bool = True) -> dict[str, Any]:
        return self._layout.assemble(self._values, fill_missing=fill_missing)

    @property
    def warmed_up(self) -> bool:
        return all(value is not None for value in self._values)

    def publish_frame(self, action_frame: dict[str, Any]) -> dict[str, Any]:
        self.published.append(action_frame)
        return action_frame

    def send_safety_action(self) -> None:
        self.safety_actions += 1

    def reset_state(self) -> None:
        self._values = [None] * len(self._values)

    def events(self) -> dict[str, bool]:
        return dict(self._events)

    def close(self) -> None:
        self.closed = True


class FakeInterface:
    """:class:`~rosetta.robots.protocols.RobotInterface` for ``fake/*`` types."""

    def register_codecs(self) -> None:
        import fake_codecs  # noqa: F401

    def provided_timelines(self, msg_type: str) -> frozenset[str]:
        if msg_type not in (VECTOR, DATA, TEXT):
            raise ValueError(f"Unknown message type '{msg_type}'")
        return frozenset({"arrival"})

    def validate_qos(self, qos: dict[str, Any]) -> None:
        if set(qos) - {"level"} or qos.get("level", 0) not in (0, 1, 2):
            raise ValueError(f"qos must be {{level: 0|1|2}}, got {qos}")

    def connect(
        self,
        *,
        observations: Sequence[Any],
        actions: Sequence[Any],
        fps: int,
        name: str,
        events: Any = None,
        use_sim_time: bool = False,
    ) -> FakeConnection:
        return FakeConnection(observations, events, simulated=use_sim_time)
