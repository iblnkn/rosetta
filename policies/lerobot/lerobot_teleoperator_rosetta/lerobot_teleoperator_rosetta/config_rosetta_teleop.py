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
from rosetta.contract.schema import Contract, load_contract
from rosetta.contract.specs import iter_teleop_feedback_specs, iter_teleop_input_specs


@TeleoperatorConfig.register_subclass("rosetta_teleop")
@dataclass
class RosettaTeleopConfig(TeleoperatorConfig):
    config_path: str = ""

    _contract: Contract | None = field(default=None, init=False, repr=False)
    _input_specs: list | None = field(default=None, init=False, repr=False)
    _feedback_specs: list | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # TeleoperatorConfig (unlike RobotConfig) defines no __post_init__;
        # guard so a future lerobot adding one still gets called.
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        if not self.config_path:
            return

        self._contract = load_contract(self.config_path)

        if self._contract.teleop is None:
            raise ValueError(f"Contract '{self.config_path}' has no 'teleop' section")

        if self.id is None:
            self.id = f"{self._contract.robot_type}_teleop"

        # Resolve once (like RosettaConfig): repeated property access must not
        # re-run spec resolution or hand out fresh spec objects each call.
        self._input_specs = list(iter_teleop_input_specs(self._contract))
        self._feedback_specs = list(iter_teleop_feedback_specs(self._contract))

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
        if self._input_specs is None:
            raise ValueError("No contract loaded")
        return self._input_specs

    @property
    def events_spec(self):
        return self.contract.teleop.events if self.contract.teleop else None

    @property
    def feedback_specs(self):
        if self._feedback_specs is None:
            raise ValueError("No contract loaded")
        return self._feedback_specs
