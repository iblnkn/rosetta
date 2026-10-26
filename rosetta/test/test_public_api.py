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

"""rosetta's public Python API: the modules and names other packages may import.

Each public module declares ``__all__``; everything else in rosetta is internal.
This pins the exact names, so adding, renaming or removing one is a deliberate
change to this file rather than a side effect of a refactor.
"""

import importlib

import pytest

PUBLIC_API = {
    "rosetta.contract.errors": {"ContractValidationError"},
    "rosetta.contract.operators": {
        "Invertibility",
        "OPERATOR_ENTRY_POINT_GROUP",
        "Operator",
        "OperatorContext",
        "register_operator",
    },
    "rosetta.contract.schema": {
        "Align",
        "Channel",
        "Contract",
        "FRAME_SECTIONS",
        "FieldKind",
        "FrameEntry",
        "ResamplePolicy",
        "SafetyBehavior",
        "Source",
        "TELEOP_FEEDBACK_KEY",
        "TELEOP_INPUT_KEY",
        "Task",
        "Teleop",
        "TeleopEventMap",
        "TeleopFeedbackSource",
        "TeleopInputSource",
        "load_contract",
        "owning_key",
        "parse_contract",
    },
    "rosetta.contract.sidecar": {
        "CONTRACT_SIDECAR_FILENAME",
        "LEROBOT_CONTRACT_SIDECAR_PATH",
        "find_contract_for_pretrained",
        "scan_inline_codec_paths",
    },
    "rosetta.contract.specs": {
        "ActionStreamSpec",
        "ObservationStreamSpec",
        "StreamSpec",
        "iter_action_specs",
        "iter_observation_specs",
        "iter_policy_specs",
        "iter_reward_as_action_specs",
        "iter_specs",
        "iter_teleop_feedback_specs",
        "iter_teleop_input_specs",
    },
    "rosetta.frames.codecs": {
        "CODEC_ENTRY_POINT_GROUP",
        "DecoderFn",
        "EncoderFn",
        "NonFiniteActionError",
        "SUPPORTED_NUMERIC_DTYPES",
        "decode_value",
        "discover_codecs",
        "encode_value",
        "register_decoder",
        "register_encoder",
    },
    "rosetta.frames.layout": {"FrameLayout", "KeyLayout", "SpecSlice", "build_feature"},
    "rosetta.frames.naming": {"IMAGE_KEY_PREFIX", "camera_name", "sanitize_field_name"},
    "rosetta.frames.protocols": {"FrameIO"},
    "rosetta.policies": {
        "DatasetWriter",
        "PolicyRunner",
        "RunnerFeedback",
        "RunnerHost",
        "RunnerResult",
        "available_dataset_writers",
        "available_policy_runners",
        "load_dataset_writer",
        "load_policy_runner",
    },
    "rosetta.robots.protocols": {"RobotInterface"},
    "rosetta.robots.registry": {"ROBOT_INTERFACE_GROUP", "available_robot_interfaces", "load_robot_interface"},
}


@pytest.mark.parametrize("module_name", sorted(PUBLIC_API))
def test_public_module_exports_exactly_the_pinned_names(module_name):
    module = importlib.import_module(module_name)
    assert set(module.__all__) == PUBLIC_API[module_name]


@pytest.mark.parametrize("module_name", sorted(PUBLIC_API))
def test_every_exported_name_exists(module_name):
    module = importlib.import_module(module_name)
    assert [name for name in module.__all__ if not hasattr(module, name)] == []
