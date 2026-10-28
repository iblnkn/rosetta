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

"""Setup-time resolution of boolean launch flags (rosetta#27).

Left lazy, an autostart condition is evaluated only when its event handler
fires, in whatever configuration scope exists then — which read the wrong
value when the launch file was included from a launch XML on Jazzy. These
tests pin the fix: a condition built with a context must carry its resolved
value into any later scope, simulated here by evaluating against a fresh,
empty LaunchContext.
"""

import pytest
from launch import LaunchContext
from launch_ros.actions import LifecycleNode

from rosetta_ros.launch_utils import autostart_handlers, typed_config


def _context_with(**flags: str) -> LaunchContext:
    context = LaunchContext()
    context.launch_configurations.update(flags)
    return context


def _node() -> LifecycleNode:
    return LifecycleNode(package="rosetta_ros", executable="probe", name="probe", namespace="")


def _conditions(handlers) -> list:
    """The condition on each handler's emitted transition: configure, then activate."""
    return [handler.event_handler.describe()[1][0].condition for handler in handlers]


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("True", True), ("1", True), ("false", False), ("False", False), ("0", False)],
)
def test_frozen_condition_survives_foreign_scope(value, expected):
    handlers = autostart_handlers(_node(), _context_with(configure=value, activate=value))
    assert [c.evaluate(LaunchContext()) for c in _conditions(handlers)] == [expected, expected]


def test_unparseable_flag_raises_at_setup_not_silently_false():
    # typed_config is what autostart_handlers resolves each flag with.
    with pytest.raises(ValueError, match="Cannot convert value 'banana'"):
        typed_config(_context_with(configure="banana"), "configure", bool)
    with pytest.raises(ValueError, match="Cannot convert value 'banana'"):
        autostart_handlers(_node(), _context_with(configure="banana", activate="true"))
