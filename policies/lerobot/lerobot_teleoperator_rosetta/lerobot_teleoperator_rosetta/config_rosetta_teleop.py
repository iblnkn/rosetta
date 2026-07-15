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

"""Config for the ``rosetta_teleop`` LeRobot teleoperator.

Loads a contract and resolves its teleop input and feedback specs once, at
construction, so property access stays cheap and hands back the same objects.
"""

from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig
from rosetta.contract.schema import Contract, load_contract
from rosetta.contract.specs import iter_teleop_feedback_specs, iter_teleop_input_specs


@TeleoperatorConfig.register_subclass("rosetta_teleop")
@dataclass
class RosettaTeleopConfig(TeleoperatorConfig):
    """Contract-driven teleop config.

    Specs resolve in ``__post_init__``. The cached private fields are
    ``init=False`` so the dataclass never treats them as constructor args.
    """

    config_path: str = ""

    _contract: Contract | None = field(default=None, init=False, repr=False)
    _input_specs: list | None = field(default=None, init=False, repr=False)
    _feedback_specs: list | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # TeleoperatorConfig has no __post_init__ today. Call it defensively so a
        # future lerobot that adds one still runs.
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        # Empty config_path is a valid deferred state. Properties raise until a
        # contract is loaded.
        if not self.config_path:
            return

        self._contract = load_contract(self.config_path)

        if self._contract.teleop is None:
            raise ValueError(f"Contract '{self.config_path}' has no 'teleop' section")

        if self.id is None:
            self.id = f"{self._contract.robot_type}_teleop"

        # Resolve once. Repeated property access must not re-run resolution or
        # hand back fresh spec objects each call.
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
        # self.contract raises without a contract, and __post_init__ guarantees
        # teleop is present once one loads, so no None-teleop branch is needed.
        return self.contract.teleop.events

    @property
    def feedback_specs(self):
        if self._feedback_specs is None:
            raise ValueError("No contract loaded")
        return self._feedback_specs
