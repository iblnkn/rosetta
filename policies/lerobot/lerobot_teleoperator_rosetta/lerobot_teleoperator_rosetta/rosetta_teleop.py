#!/usr/bin/env python
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

"""LeRobot Teleoperator over a live robot connection.

connect() opens the human device through the contract's robot interface
(``RobotInterface.connect``): teleop inputs are its observation streams,
feedback its action streams, and the contract's teleop events its event map.
Only the LeRobot side lives here: flattening per-key vectors to the
{name: float} shape lerobot expects, and mapping contract event names onto
lerobot's TeleopEvents. It imports no robot middleware.

Feedback channels cannot declare a safety behavior (rejected at contract load),
so closing the connection publishes nothing.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import cached_property
from typing import Any, Optional

import numpy as np
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from rosetta.frames.layout import FrameLayout
from rosetta.frames.protocols import Connection
from rosetta.robots.registry import load_robot_interface

from .config_rosetta_teleop import RosettaTeleopConfig

# Contract event names this plugin acts on, mapped onto lerobot's TeleopEvents.
# Contract event names are open; names not mapped here are ignored. The episode-ending
# events compose two lerobot events (a reward label plus TERMINATE_EPISODE), and
# ``start_episode`` has no lerobot counterpart at all (lerobot's record loop
# starts episodes itself), so the rosetta-native path (rosetta_ros_hil's HIL
# manager) is the only consumer of that event.
EVENT_NAME_TO_ENUMS: dict[str, tuple[TeleopEvents, ...]] = {
    "is_intervention": (TeleopEvents.IS_INTERVENTION,),
    "success": (TeleopEvents.SUCCESS,),
    "failure": (TeleopEvents.FAILURE,),
    "end_success": (TeleopEvents.SUCCESS, TeleopEvents.TERMINATE_EPISODE),
    "end_failure": (TeleopEvents.FAILURE, TeleopEvents.TERMINATE_EPISODE),
}

# One source of truth for the event key set: every lerobot TeleopEvents member,
# so consumers always see the full, stable key set (RERECORD_EPISODE stays
# False — the contract has no event for it).
DEFAULT_EVENTS: dict[TeleopEvents, bool] = dict.fromkeys(TeleopEvents, False)


def teleop_events(event_values: Mapping[str, bool]) -> dict[TeleopEvents, bool]:
    """LeRobot's TeleopEvents from the contract's event values.

    Several contract events may assert the same lerobot event (``success`` and
    ``end_success`` both assert SUCCESS), so each lerobot event is the OR of
    its sources. Contract events with no lerobot mapping are ignored.
    """
    events = dict(DEFAULT_EVENTS)
    for name, value in event_values.items():
        for enum in EVENT_NAME_TO_ENUMS.get(name, ()):
            events[enum] = events[enum] or bool(value)
    return events


class RosettaTeleop(Teleoperator):
    """LeRobot Teleoperator over a Rosetta robot connection."""

    config_class = RosettaTeleopConfig
    name = "rosetta_teleop"

    def __init__(self, config: RosettaTeleopConfig):
        super().__init__(config)
        self.config = config
        self._calibrated = True
        self._connection: Optional[Connection] = None

    @cached_property
    def _input_layout(self) -> FrameLayout:
        return FrameLayout(self.config.input_specs)

    @cached_property
    def _feedback_layout(self) -> FrameLayout:
        return FrameLayout(self.config.feedback_specs)

    @property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for spec in self.config.input_specs:
            for name in spec.namespaced_names:
                features[name] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for spec in self.config.feedback_specs:
            for name in spec.namespaced_names:
                features[name] = float
        return features

    @property
    def is_connected(self) -> bool:
        return self._connection is not None

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        """Nothing to configure: connect() opens the device ready to run."""

    def connect(self, calibrate: bool = True) -> None:
        """Open the device through the contract's robot interface.

        Raises DeviceAlreadyConnectedError when already connected, matching the
        robot adapter and lerobot's own devices.
        """
        del calibrate  # Unused - the teleop device needs no calibration
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        events = self.config.events_spec
        if events is not None:
            unmapped = [name for name in events.select if name not in EVENT_NAME_TO_ENUMS]
            if unmapped:
                logging.info(
                    "Teleop events with no lerobot mapping, ignored on this path "
                    "(a HIL app such as rosetta_ros_hil handles them): %s",
                    unmapped,
                )
        interface = load_robot_interface(self.config.contract.robot_interface)
        self._connection = interface.connect(
            observations=self.config.input_specs,
            actions=self.config.feedback_specs,
            fps=self.config.fps,
            name=f"rosetta_teleop_{self.id}",
            events=events,
        )

    def get_action(self) -> dict[str, Any]:
        """Flatten the current inputs to {namespaced_name: float}.

        A key whose streams have no data yet is omitted, not zero-filled: a
        fake zero teleop value would command motion.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        frame = self._connection.sample_frame(fill_missing=False)
        action: dict[str, Any] = {}
        for key, vec in frame.items():
            names = [name for sl in self._input_layout[key].slices for name in sl.spec.namespaced_names]
            for name, value in zip(names, np.asarray(vec).flatten(), strict=True):
                action[name] = float(value)
        return action

    def get_teleop_events(self) -> dict[TeleopEvents, bool]:
        """Current event state, or all-False defaults before connect.

        lerobot polls events outside the connected window, so this tolerates a
        missing connection instead of raising like get_action and send_feedback.
        """
        if self._connection is None:
            return dict(DEFAULT_EVENTS)
        return teleop_events(self._connection.events())

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Regroup {namespaced_name: value} into per-key vectors and publish.

        All or nothing: a key is published only when every one of its names is
        present, because ``publish_frame`` needs the full per-key vector. A
        partial frame would be a malformed action.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        frame: dict[str, Any] = {}
        for key in self._feedback_layout.keys:
            names = [name for sl in self._feedback_layout[key].slices for name in sl.spec.namespaced_names]
            if not all(name in feedback for name in names):
                return
            frame[key] = np.array([feedback[name] for name in names], dtype=np.float64)
        if frame:
            self._connection.publish_frame(frame)

    def disconnect(self) -> None:
        """Close the connection; a no-op when not connected."""
        connection, self._connection = self._connection, None
        if connection is not None:
            connection.close()
