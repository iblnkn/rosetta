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
timeline, ``arrival``; qos is ``{level: 0|1|2}``. The ``fake_interface``
fixture in conftest.py installs it as ``fake``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
