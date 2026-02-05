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

from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig

from rosetta.common.contract import Contract, load_contract
from rosetta.common.contract_utils import iter_teleop_input_specs, iter_teleop_feedback_specs


@TeleoperatorConfig.register_subclass("rosetta_teleop")
@dataclass
class RosettaTeleopConfig(TeleoperatorConfig):
    config_path: str = ""

    _contract: Contract | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if not self.config_path:
            return

        self._contract = load_contract(self.config_path)

        if self._contract.teleop is None:
            raise ValueError(f"Contract '{self.config_path}' has no 'teleop' section")

        if self.id is None:
            self.id = f"{self._contract.robot_type}_teleop"

    @property
    def contract(self):
        if self._contract is None:
            raise ValueError("No contract loaded")
        return self._contract

    @property
    def fps(self):
        return self.contract.fps

    @property
    def input_specs(self):
        return list(iter_teleop_input_specs(self.contract))

    @property
    def events_spec(self):
        return self.contract.teleop.events if self.contract.teleop else None

    @property
    def feedback_specs(self):
        return list(iter_teleop_feedback_specs(self.contract))
