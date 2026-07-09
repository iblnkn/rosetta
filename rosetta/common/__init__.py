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

"""
Rosetta common utilities for contract processing and ROS message handling.

This package provides:
- contract: Dataclasses, enums, and YAML loading for contracts
- converters: Encoder/decoder registration and message field access
- contract_utils: Spec iteration, StreamBuffer, and timestamp utilities
- ros2_utils: QoS profile utilities
- decoders: ROS message decoders
- encoders: ROS message encoders
"""

# Contract types and loading
from .contract import (
    AlignSpec,
    ActionSpec,
    ActionStreamSpec,
    Contract,
    ContractValidationError,
    DEPTH_ENCODINGS,
    ObservationSpec,
    ObservationStreamSpec,
    ResetMode,
    ResetSpec,
    ResamplePolicy,
    SafetyBehavior,
    StampSource,
    StreamSpec,
    TaskSpec,
    TeleopEventsSpec,
    TeleopSpec,
    VisualizationSpec,
    is_valid_lerobot_dtype,
    LEROBOT_SPECIAL_DTYPES,
    load_contract,
)

# Codec registry
from .converters import (
    decode_value,
    encode_value,
    get_decoder_dtype,
    register_decoder,
    register_encoder,
    DECODERS,
    DTYPES,
    ENCODERS,
)

# Contract utilities (spec iteration, buffers)
from .contract_utils import (
    build_feature,
    get_namespaced_names,
    iter_action_specs,
    iter_extended_specs,
    iter_observation_specs,
    iter_specs,
    iter_teleop_feedback_specs,
    iter_teleop_input_specs,
    StreamBuffer,
    zeros_for_spec,
)

# ROS2 utilities. Guarded so the contract loading / validation modules stay
# importable without rclpy (training conda env, CI) — contract_validation and
# the sns_robot_learning contract tests rely on this.
try:
    from .ros2_utils import (
        dot_get,
        dot_set,
        get_message_timestamp_ns,
        qos_profile_from_dict,
        stamp_from_header_ns,
    )
except ModuleNotFoundError as e:  # pragma: no cover - no rclpy in ROS-free envs
    # Only a missing rclpy is expected here; anything else (a broken
    # transitive import, a typo inside ros2_utils) must fail loudly.
    if (e.name or "").partition(".")[0] != "rclpy":
        raise
    _HAVE_ROS2_UTILS = False
else:
    _HAVE_ROS2_UTILS = True

__all__ = [
    # Contract types and loading
    "AlignSpec",
    "ActionSpec",
    "ActionStreamSpec",
    "Contract",
    "ContractValidationError",
    "DEPTH_ENCODINGS",
    "ObservationSpec",
    "ObservationStreamSpec",
    "ResetMode",
    "ResetSpec",
    "ResamplePolicy",
    "SafetyBehavior",
    "StampSource",
    "StreamSpec",
    "TaskSpec",
    "TeleopEventsSpec",
    "TeleopSpec",
    "VisualizationSpec",
    "is_valid_lerobot_dtype",
    "LEROBOT_SPECIAL_DTYPES",
    "load_contract",
    # Codec registry
    "decode_value",
    "encode_value",
    "get_decoder_dtype",
    "register_decoder",
    "register_encoder",
    "DECODERS",
    "DTYPES",
    "ENCODERS",
    # Contract utilities
    "build_feature",
    "get_namespaced_names",
    "iter_action_specs",
    "iter_extended_specs",
    "iter_observation_specs",
    "iter_specs",
    "iter_teleop_feedback_specs",
    "iter_teleop_input_specs",
    "StreamBuffer",
    "zeros_for_spec",
]

# The ROS2 utility names are public only when rclpy is present (guarded
# import above); advertising them unconditionally broke star-imports in the
# exact ROS-free environments the guard exists for.
if _HAVE_ROS2_UTILS:
    __all__ += [
        "dot_get",
        "dot_set",
        "get_message_timestamp_ns",
        "qos_profile_from_dict",
        "stamp_from_header_ns",
    ]
