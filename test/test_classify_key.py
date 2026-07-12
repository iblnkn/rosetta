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

"""Unit tests for the shared framework-neutral key classifier + sanitizer."""

from rosetta.frames.naming import classify_key, sanitize_field_name


def test_classify_key_roles():
    assert classify_key("observation.images.front") == "image"
    assert classify_key("observation.images.wrist.right") == "image"
    assert classify_key("action") == "action"
    assert classify_key("action.left") == "action"
    assert classify_key("observation.state") == "state"
    assert classify_key("observation.environment_state") == "state"
    # only the images.* subtree is an image; a bare observation is state
    assert classify_key("observation.foo") == "state"


def test_sanitize_field_name_is_dot_free():
    assert sanitize_field_name("observation.state") == "observation_state"
    assert sanitize_field_name("action.left") == "action_left"
    assert sanitize_field_name("wrist.right") == "wrist_right"
    assert sanitize_field_name("already_ok") == "already_ok"
