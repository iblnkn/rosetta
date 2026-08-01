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

from rosetta.robots.ros2.launch_utils import _flag_condition, typed_config


def _context_with(**flags: str) -> LaunchContext:
    context = LaunchContext()
    context.launch_configurations.update(flags)
    return context


@pytest.mark.parametrize("value", ["true", "True", "1"])
def test_frozen_condition_true_survives_foreign_scope(value):
    condition = _flag_condition("configure", _context_with(configure=value))
    assert condition.evaluate(LaunchContext()) is True


@pytest.mark.parametrize("value", ["false", "False", "0"])
def test_frozen_condition_false_survives_foreign_scope(value):
    condition = _flag_condition("configure", _context_with(configure=value))
    assert condition.evaluate(LaunchContext()) is False


def test_unparseable_flag_raises_at_setup_not_silently_false():
    with pytest.raises(Exception, match="[Cc]annot|[Ii]nvalid|convert"):
        _flag_condition("configure", _context_with(configure="banana"))


def test_typed_config_coerces_bool():
    assert typed_config(_context_with(activate="true"), "activate", bool) is True
    assert typed_config(_context_with(activate="0"), "activate", bool) is False
