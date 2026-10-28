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

"""yaml_params reads a node's parameters from any section ROS would apply to it."""

import pytest

from rosetta_ros.launch_utils import yaml_params


def _params(tmp_path, text):
    path = tmp_path / "params.yaml"
    path.write_text(text)
    return yaml_params(str(path), "policy_runner")


@pytest.mark.parametrize("key", ["policy_runner", "/policy_runner", "/**", "/**/policy_runner", "/robot/policy_runner"])
def test_every_section_ros_applies_is_read(tmp_path, key):
    assert _params(tmp_path, f"{key}:\n  ros__parameters:\n    policy_device: cuda\n") == {"policy_device": "cuda"}


def test_the_more_specific_section_wins(tmp_path):
    text = "policy_runner:\n  ros__parameters: {a: 2}\n/**:\n  ros__parameters: {a: 1, b: 1}\n"
    assert _params(tmp_path, text) == {"a": 2, "b": 1}


def test_a_file_for_other_nodes_raises(tmp_path):
    with pytest.raises(ValueError, match="no section applies to node 'policy_runner'"):
        _params(tmp_path, "episode_recorder:\n  ros__parameters: {a: 1}\n")


def test_an_empty_file_is_no_parameters(tmp_path):
    assert _params(tmp_path, "") == {}
