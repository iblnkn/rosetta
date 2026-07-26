# Copyright 2026 Brian Blankenau
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

r"""
Launch file for the Rosetta HIL system.

Launches 4 nodes:
  1. robot_policy (policy_runner_node) - policy inference with remapped action output
  2. reward_classifier (policy_runner_node) - optional reward classification
  3. episode_recorder (episode_recorder_node) - bag recording on real topics
  4. hil_manager (hil_manager_node) - orchestrator with muxing

The robot policy's action output is remapped to an intermediate topic so the HIL
manager can mux between policy and teleop input before publishing to the real
command topic. The episode recorder subscribes to the real topic, recording
whatever the robot actually receives.

All nodes are lifecycle nodes with auto-configure and auto-activate by default.

Usage:
    # Launch with defaults
    ros2 launch rosetta hil_launch.py

    # With reward classifier (uses same contract, is_classifier reads reward section)
    ros2 launch rosetta hil_launch.py \\
        enable_reward_classifier:=true \\
        reward_classifier_pretrained_name_or_path:=/path/to/reward_model

    # Without auto-activation (manual lifecycle control)
    ros2 launch rosetta hil_launch.py configure:=false activate:=false

    # Override robot policy model
    ros2 launch rosetta hil_launch.py \\
        pretrained_name_or_path:=/path/to/policy_model
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import (
    EqualsSubstitution,
    LaunchConfiguration,
    PythonExpression,
)
from launch_ros.actions import LifecycleNode

from rosetta.robots.ros2.launch_utils import autostart_handlers, typed_config, yaml_params


def _validate_remaps(context):
    """Fail at launch when the remap topology doesn't match the contract.

    The mux only works if the policy's remapped output topic is derived from
    a topic the contract actually declares — a mismatch (e.g. an overridden
    contract_path with a different action topic) would let the policy publish
    straight to the real command topic with no error anywhere.
    """
    from rosetta.contract.schema import load_contract

    contract = load_contract(LaunchConfiguration("contract_path").perform(context))
    action_topics = sorted({src.channel.topic for entry in contract.actions for src in entry.sources})
    remap_from = LaunchConfiguration("action_remap_from").perform(context)
    if remap_from not in action_topics:
        raise RuntimeError(
            f"action_remap_from '{remap_from}' is not an action topic of the contract "
            f"(action topics: {action_topics}); the HIL mux would never intercept policy output."
        )
    if typed_config(context, "enable_reward_classifier", bool):
        reward_topics = sorted({src.channel.topic for entry in contract.rewards for src in entry.sources})
        reward_from = LaunchConfiguration("reward_remap_from").perform(context)
        if reward_from not in reward_topics:
            raise RuntimeError(
                f"reward_remap_from '{reward_from}' is not a reward topic of the contract "
                f"(reward topics: {reward_topics}); the reward mux would never intercept classifier output."
            )
    return []


def generate_launch_description():
    """Generate the launch description for the human-in-the-loop system."""
    rosetta_share = get_package_share_directory("rosetta")

    default_contract = os.path.join(rosetta_share, "contracts", "so_101_hil.yaml")  # fallback only
    default_rosetta_params = os.path.join(rosetta_share, "params", "policy_runner.yaml")
    default_recorder_params = os.path.join(rosetta_share, "params", "episode_recorder.yaml")
    default_hil_params = os.path.join(rosetta_share, "params", "hil_manager.yaml")

    # Build per-node config dicts for launch argument defaults.
    #
    # hil_manager.yaml is the "super YAML" for this launch:
    #   - its robot_policy / episode_recorder sections override the node's own YAML
    #   - its reward_classifier section provides all classifier defaults
    #   - CLI arguments override everything
    #
    # Merge order (last wins): base node YAML < HIL super YAML < CLI arg
    with open(default_hil_params) as f:
        hil_full = yaml.safe_load(f)

    def _section(name):
        return hil_full.get(name, {}).get("ros__parameters", {})

    hil_cfg = _section("hil_manager")
    contract_path_default = hil_full.get("contract_path", default_contract)
    client_cfg = {
        **yaml_params(default_rosetta_params, "policy_runner"),
        **_section("robot_policy"),
    }
    recorder_cfg = {
        **yaml_params(default_recorder_params, "episode_recorder"),
        **_section("episode_recorder"),
    }
    reward_cfg = {
        **yaml_params(default_rosetta_params, "policy_runner"),
        **_section("reward_classifier"),
    }

    # ==================================================================
    # Launch arguments
    # ==================================================================

    launch_args = [
        # --- Shared (launch-specific, no YAML source) ---
        DeclareLaunchArgument(
            "contract_path",
            default_value=contract_path_default,
            description="Path to HIL contract YAML file",
        ),
        DeclareLaunchArgument(
            "log_level",
            default_value="info",
            description="Logging level (debug, info, warn, error)",
        ),
        DeclareLaunchArgument(
            "configure",
            default_value="true",
            description="Whether to auto-configure nodes on startup",
        ),
        DeclareLaunchArgument(
            "activate",
            default_value="true",
            description="Whether to auto-activate nodes on startup (requires configure:=true)",
        ),
        # --- Robot policy (defaults from policy_runner.yaml) ---
        DeclareLaunchArgument(
            "pretrained_name_or_path",
            default_value=client_cfg["pretrained_name_or_path"],
            description="HuggingFace model ID or local path to trained policy model",
        ),
        DeclareLaunchArgument(
            "server_address",
            default_value=client_cfg["server_address"],
            description="LeRobot policy server address (host:port)",
        ),
        DeclareLaunchArgument(
            "policy_type",
            default_value=client_cfg["policy_type"],
            description="Policy type: act, smolvla, diffusion, pi0, pi05, etc.",
        ),
        DeclareLaunchArgument(
            "policy_device",
            default_value=client_cfg["policy_device"],
            description="Inference device: cuda, cpu, mps, or cuda:0",
        ),
        DeclareLaunchArgument(
            "actions_per_chunk",
            default_value=str(client_cfg["actions_per_chunk"]),
            description="Number of actions per inference chunk",
        ),
        DeclareLaunchArgument(
            "chunk_size_threshold",
            default_value=str(client_cfg["chunk_size_threshold"]),
            description="Threshold for requesting new chunk (0.0-1.0)",
        ),
        DeclareLaunchArgument(
            "aggregate_fn_name",
            default_value=client_cfg["aggregate_fn_name"],
            description="Chunk aggregation: weighted_average, latest_only, average, conservative",
        ),
        DeclareLaunchArgument(
            "obs_similarity_atol",
            default_value=str(client_cfg["obs_similarity_atol"]),
            description="Observation filtering tolerance (-1.0 to disable)",
        ),
        # --- Action mux remapping ---
        # remap_from must be an action topic of the contract (validated at
        # launch); remap_to derives from prefix + remap_from so the pair
        # cannot silently disagree. The prefix is declared first: a derived
        # default resolves when its argument is declared.
        DeclareLaunchArgument(
            "policy_remap_prefix",
            default_value=hil_cfg["policy_remap_prefix"],
            description="Topic prefix for remapped policy output",
        ),
        DeclareLaunchArgument(
            "action_remap_from",
            default_value="/leader_arm/joint_states",
            description="Original action topic to remap (must match the contract)",
        ),
        DeclareLaunchArgument(
            "action_remap_to",
            default_value=[LaunchConfiguration("policy_remap_prefix"), LaunchConfiguration("action_remap_from")],
            description="Remapped action topic for policy output (default: prefix + action_remap_from)",
        ),
        # --- HIL manager (defaults from hil_manager.yaml) ---
        DeclareLaunchArgument(
            "enable_reward_classifier",
            default_value=str(hil_cfg["enable_reward_classifier"]).lower(),
            description="Enable reward classifier policy",
        ),
        DeclareLaunchArgument(
            "enable_recording",
            default_value=str(hil_cfg["enable_recording"]).lower(),
            description="Record episodes to a bag. Everything else (policy, mux, teleop "
            "events, intervention) runs identically when false.",
        ),
        DeclareLaunchArgument(
            "manage_policy_lifecycle",
            default_value=str(hil_cfg["manage_policy_lifecycle"]).lower(),
            description="Send/cancel a RunPolicy goal at episode start/end. False assumes "
            "a policy is already running externally and just mux/records/labels around it.",
        ),
        DeclareLaunchArgument(
            "default_prompt",
            default_value=hil_cfg["default_prompt"],
            description="Task prompt used whenever a goal, service call, or the start_episode "
            "teleop event leaves prompt empty.",
        ),
        DeclareLaunchArgument(
            "feedback_rate_hz",
            default_value=str(hil_cfg["feedback_rate_hz"]),
            description="Feedback publish rate (Hz) for all nodes",
        ),
        DeclareLaunchArgument(
            "human_reward_positive",
            default_value=str(hil_cfg["human_reward_positive"]),
            description="Reward value for human positive override",
        ),
        DeclareLaunchArgument(
            "human_reward_negative",
            default_value=str(hil_cfg["human_reward_negative"]),
            description="Reward value for human negative override",
        ),
        # --- Reward classifier (defaults from hil_manager.yaml; the installed
        # file is the source of truth — a missing key is a config error, not
        # something to paper over with a second default here) ---
        DeclareLaunchArgument(
            "reward_classifier_contract_path",
            default_value=reward_cfg["contract_path"],
            description="Contract YAML for reward classifier (defaults to contract_path)",
        ),
        DeclareLaunchArgument(
            "reward_classifier_pretrained_name_or_path",
            default_value=reward_cfg["pretrained_name_or_path"],
            description="Path to trained reward classifier model",
        ),
        DeclareLaunchArgument(
            "reward_classifier_policy_type",
            default_value=reward_cfg["policy_type"],
            description="Policy type for reward classifier model",
        ),
        DeclareLaunchArgument(
            "reward_classifier_server_address",
            default_value=reward_cfg["server_address"],
            description="Reward classifier policy server address (host:port)",
        ),
        DeclareLaunchArgument(
            "reward_remap_prefix",
            default_value=hil_cfg["reward_remap_prefix"],
            description="Topic prefix for remapped reward classifier output",
        ),
        DeclareLaunchArgument(
            "reward_remap_from",
            default_value="/reward",
            description="Original reward topic to remap (must match the contract rewards section)",
        ),
        DeclareLaunchArgument(
            "reward_remap_to",
            default_value=[LaunchConfiguration("reward_remap_prefix"), LaunchConfiguration("reward_remap_from")],
            description="Remapped reward topic for classifier output (default: prefix + reward_remap_from)",
        ),
        # --- Episode recorder (defaults from episode_recorder.yaml) ---
        DeclareLaunchArgument(
            "bag_base_dir",
            default_value=recorder_cfg["bag_base_dir"],
            description="Directory for rosbag output",
        ),
        DeclareLaunchArgument(
            "storage_id",
            default_value=recorder_cfg["storage_id"],
            description="Rosbag format: mcap (recommended) or sqlite3",
        ),
        DeclareLaunchArgument(
            "default_max_duration_s",
            default_value=str(recorder_cfg["default_max_duration_s"]),
            description="Max episode duration in seconds (recorder fallback)",
        ),
    ]

    # ==================================================================
    # Node 1: Robot policy (policy_runner_node)
    # ==================================================================
    # Remaps action output so HIL manager can mux between policy and teleop.

    # Base parameters are the MERGED dicts, not the node's params-file path:
    # a params file's bare top-level `policy_runner:` key only matches the
    # root-namespace node, so it is silently inert for the namespaced HIL
    # nodes — every YAML key not re-threaded as a launch arg (e.g.
    # server_startup_timeout_sec) never arrived. A dict applies
    # unconditionally; the launch-arg dict layered after it still wins.
    robot_policy_node = LifecycleNode(
        package="rosetta",
        executable="policy_runner_node",
        name="policy_runner",
        namespace="robot_policy",
        output="screen",
        emulate_tty=True,
        remappings=[
            (LaunchConfiguration("action_remap_from"), LaunchConfiguration("action_remap_to")),
        ],
        parameters=[
            client_cfg,
            {
                "contract_path": LaunchConfiguration("contract_path"),
                "pretrained_name_or_path": LaunchConfiguration("pretrained_name_or_path"),
                "server_address": LaunchConfiguration("server_address"),
                "policy_type": LaunchConfiguration("policy_type"),
                "policy_device": LaunchConfiguration("policy_device"),
                "actions_per_chunk": LaunchConfiguration("actions_per_chunk"),
                "chunk_size_threshold": LaunchConfiguration("chunk_size_threshold"),
                "aggregate_fn_name": LaunchConfiguration("aggregate_fn_name"),
                "feedback_rate_hz": LaunchConfiguration("feedback_rate_hz"),
                "launch_local_server": True,
                "obs_similarity_atol": LaunchConfiguration("obs_similarity_atol"),
            },
        ],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
    )

    # ==================================================================
    # Node 2: Reward classifier (policy_runner_node) - conditional
    # ==================================================================

    # Use main contract_path when reward_classifier_contract_path is empty
    reward_contract = PythonExpression(
        [
            "'",
            LaunchConfiguration("reward_classifier_contract_path"),
            "' if '",
            LaunchConfiguration("reward_classifier_contract_path"),
            "' else '",
            LaunchConfiguration("contract_path"),
            "'",
        ]
    )

    reward_classifier_node = LifecycleNode(
        package="rosetta",
        executable="policy_runner_node",
        name="policy_runner",
        namespace="reward_classifier",
        output="screen",
        emulate_tty=True,
        condition=IfCondition(EqualsSubstitution(LaunchConfiguration("enable_reward_classifier"), "true")),
        remappings=[
            (LaunchConfiguration("reward_remap_from"), LaunchConfiguration("reward_remap_to")),
        ],
        parameters=[
            reward_cfg,
            {
                "contract_path": reward_contract,
                "pretrained_name_or_path": LaunchConfiguration("reward_classifier_pretrained_name_or_path"),
                "server_address": LaunchConfiguration("reward_classifier_server_address"),
                "policy_type": LaunchConfiguration("reward_classifier_policy_type"),
                "policy_device": LaunchConfiguration("policy_device"),
                "actions_per_chunk": LaunchConfiguration("actions_per_chunk"),
                "chunk_size_threshold": LaunchConfiguration("chunk_size_threshold"),
                "aggregate_fn_name": LaunchConfiguration("aggregate_fn_name"),
                "feedback_rate_hz": LaunchConfiguration("feedback_rate_hz"),
                "launch_local_server": True,
                "obs_similarity_atol": LaunchConfiguration("obs_similarity_atol"),
                "is_classifier": True,
            },
        ],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
    )

    # ==================================================================
    # Node 3: Episode recorder
    # ==================================================================
    # Records from real (non-remapped) topics - captures muxed output.

    episode_recorder_node = LifecycleNode(
        package="rosetta",
        executable="episode_recorder_node",
        name="episode_recorder",
        namespace="",
        output="screen",
        emulate_tty=True,
        parameters=[
            recorder_cfg,
            {
                "contract_path": LaunchConfiguration("contract_path"),
                "bag_base_dir": LaunchConfiguration("bag_base_dir"),
                "storage_id": LaunchConfiguration("storage_id"),
                "default_max_duration_s": LaunchConfiguration("default_max_duration_s"),
                "feedback_rate_hz": LaunchConfiguration("feedback_rate_hz"),
            },
        ],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
    )

    # ==================================================================
    # Node 4: HIL manager
    # ==================================================================

    hil_manager_node = LifecycleNode(
        package="rosetta",
        executable="hil_manager_node",
        name="hil_manager",
        namespace="",
        output="screen",
        emulate_tty=True,
        parameters=[
            # hil_cfg as base delivers EVERY hil_manager-section key (the
            # hand-copied allowlist this replaces silently dropped
            # default_max_duration_s and anything added later).
            hil_cfg,
            {
                "contract_path": LaunchConfiguration("contract_path"),
                "enable_reward_classifier": LaunchConfiguration("enable_reward_classifier"),
                "enable_recording": LaunchConfiguration("enable_recording"),
                "manage_policy_lifecycle": LaunchConfiguration("manage_policy_lifecycle"),
                "default_prompt": LaunchConfiguration("default_prompt"),
                "policy_remap_prefix": LaunchConfiguration("policy_remap_prefix"),
                "reward_remap_prefix": LaunchConfiguration("reward_remap_prefix"),
                "human_reward_positive": LaunchConfiguration("human_reward_positive"),
                "human_reward_negative": LaunchConfiguration("human_reward_negative"),
                "feedback_rate_hz": LaunchConfiguration("feedback_rate_hz"),
            },
        ],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
    )

    # ==================================================================
    # Lifecycle auto-configure / auto-activate
    # ==================================================================
    # reward_classifier_node is conditional, but that's safe here: neither
    # OnProcessStart nor OnStateTransition ever fires for an unlaunched node.
    lifecycle_events = [
        handler
        for node in (robot_policy_node, episode_recorder_node, hil_manager_node, reward_classifier_node)
        for handler in autostart_handlers(node)
    ]

    # ==================================================================
    # Assemble launch description
    # ==================================================================

    return LaunchDescription(
        [
            *launch_args,
            OpaqueFunction(function=_validate_remaps),
            robot_policy_node,
            reward_classifier_node,
            episode_recorder_node,
            hil_manager_node,
            *lifecycle_events,
        ]
    )
