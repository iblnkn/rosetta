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

"""
Launch file for the Rosetta HIL system.

Launches 4 nodes:
  1. robot_policy (rosetta_client_node) - policy inference with remapped action output
  2. reward_classifier (rosetta_client_node) - optional reward classification
  3. episode_recorder (episode_recorder_node) - bag recording on real topics
  4. hil_manager (rosetta_hil_manager_node) - orchestrator with muxing

The robot policy's action output is remapped to an intermediate topic so the HIL
manager can mux between policy and teleop input before publishing to the real
command topic. The episode recorder subscribes to the real topic, recording
whatever the robot actually receives.

All nodes are lifecycle nodes with auto-configure and auto-activate by default.

Usage:
    # Launch with defaults
    ros2 launch rosetta rosetta_hil_launch.py

    # With reward classifier (uses same contract, is_classifier reads reward section)
    ros2 launch rosetta rosetta_hil_launch.py \\
        enable_reward_classifier:=true \\
        reward_classifier_pretrained_name_or_path:=/path/to/reward_model

    # Without auto-activation (manual lifecycle control)
    ros2 launch rosetta rosetta_hil_launch.py configure:=false activate:=false

    # Override robot policy model
    ros2 launch rosetta rosetta_hil_launch.py \\
        pretrained_name_or_path:=/path/to/policy_model
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnExecutionComplete, OnProcessStart
from launch.events import matches_action
from launch.substitutions import (
    EqualsSubstitution,
    LaunchConfiguration,
    PythonExpression,
)
from launch_ros.actions import LifecycleNode
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition


def generate_launch_description():
    rosetta_share = get_package_share_directory('rosetta')

    default_contract = os.path.join(rosetta_share, 'contracts', 'so_101_hil.yaml')
    default_rosetta_params = os.path.join(rosetta_share, 'params', 'rosetta_client.yaml')
    default_recorder_params = os.path.join(rosetta_share, 'params', 'episode_recorder.yaml')
    default_hil_params = os.path.join(rosetta_share, 'params', 'rosetta_hil_manager.yaml')

    # ==================================================================
    # Launch arguments
    # ==================================================================

    launch_args = [
        # --- Shared ---
        DeclareLaunchArgument(
            'contract_path',
            default_value=default_contract,
            description='Path to HIL contract YAML file'
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='Logging level (debug, info, warn, error)'
        ),
        DeclareLaunchArgument(
            'configure',
            default_value='true',
            description='Whether to auto-configure nodes on startup'
        ),
        DeclareLaunchArgument(
            'activate',
            default_value='true',
            description='Whether to auto-activate nodes on startup (requires configure:=true)'
        ),

        # --- Robot policy ---
        DeclareLaunchArgument(
            'pretrained_name_or_path',
            default_value='/workspaces/rosetta_ws/models/ACT/checkpoints/last/pretrained_model',
            description='HuggingFace model ID or local path to trained policy model'
        ),
        DeclareLaunchArgument(
            'server_address',
            default_value='127.0.0.1:8080',
            description='LeRobot policy server address (host:port)'
        ),
        DeclareLaunchArgument(
            'policy_type',
            default_value='act',
            description='Policy type: act, smolvla, diffusion, pi0, pi05, etc.'
        ),
        DeclareLaunchArgument(
            'policy_device',
            default_value='cuda',
            description='Inference device: cuda, cpu, mps, or cuda:0'
        ),
        DeclareLaunchArgument(
            'actions_per_chunk',
            default_value='30',
            description='Number of actions per inference chunk'
        ),
        DeclareLaunchArgument(
            'chunk_size_threshold',
            default_value='0.95',
            description='Threshold for requesting new chunk (0.0-1.0)'
        ),
        DeclareLaunchArgument(
            'aggregate_fn_name',
            default_value='weighted_average',
            description='Chunk aggregation: weighted_average, latest_only, average, conservative'
        ),
        DeclareLaunchArgument(
            'obs_similarity_atol',
            default_value='-1.0',
            description='Observation filtering tolerance (-1.0 to disable)'
        ),

        # --- Action mux remapping ---
        DeclareLaunchArgument(
            'action_remap_from',
            default_value='/leader_arm/joint_states',
            description='Original action topic to remap (from contract)'
        ),
        DeclareLaunchArgument(
            'action_remap_to',
            default_value='/hil/policy/leader_arm/joint_states',
            description='Remapped action topic for policy output'
        ),
        DeclareLaunchArgument(
            'policy_remap_prefix',
            default_value='/hil/policy',
            description='Topic prefix for remapped policy output (must match remap_to derivation)'
        ),

        # --- Reward classifier ---
        DeclareLaunchArgument(
            'enable_reward_classifier',
            default_value='false',
            description='Enable reward classifier policy'
        ),
        DeclareLaunchArgument(
            'reward_classifier_contract_path',
            default_value='',
            description='Contract YAML for reward classifier (defaults to contract_path)'
        ),
        DeclareLaunchArgument(
            'reward_classifier_pretrained_name_or_path',
            default_value='',
            description='Path to trained reward classifier model'
        ),
        DeclareLaunchArgument(
            'reward_classifier_policy_type',
            default_value='reward_classifier',
            description='Policy type for reward classifier model'
        ),
        DeclareLaunchArgument(
            'reward_classifier_server_address',
            default_value='127.0.0.1:8081',
            description='Reward classifier policy server address (host:port)'
        ),
        DeclareLaunchArgument(
            'reward_remap_from',
            default_value='/reward',
            description='Original reward topic to remap (from contract rewards section)'
        ),
        DeclareLaunchArgument(
            'reward_remap_to',
            default_value='/hil/reward/reward',
            description='Remapped reward topic for classifier output'
        ),
        DeclareLaunchArgument(
            'reward_remap_prefix',
            default_value='/hil/reward',
            description='Topic prefix for remapped reward classifier output'
        ),

        # --- Episode recorder ---
        DeclareLaunchArgument(
            'bag_base_dir',
            default_value='/workspaces/rosetta_ws/datasets/bags',
            description='Directory for rosbag output'
        ),
        DeclareLaunchArgument(
            'storage_id',
            default_value='mcap',
            description='Rosbag format: mcap (recommended) or sqlite3'
        ),
        DeclareLaunchArgument(
            'default_max_duration',
            default_value='300.0',
            description='Max episode duration in seconds (recorder fallback)'
        ),

        # --- HIL manager ---
        DeclareLaunchArgument(
            'feedback_rate_hz',
            default_value='30.0',
            description='Feedback publish rate (Hz) for all nodes'
        ),
        DeclareLaunchArgument(
            'human_reward_positive',
            default_value='1.0',
            description='Reward value for human positive override'
        ),
        DeclareLaunchArgument(
            'human_reward_negative',
            default_value='0.0',
            description='Reward value for human negative override'
        ),
    ]

    # ==================================================================
    # Node 1: Robot policy (rosetta_client_node)
    # ==================================================================
    # Remaps action output so HIL manager can mux between policy and teleop.

    robot_policy_node = LifecycleNode(
        package='rosetta',
        executable='rosetta_client_node',
        name='rosetta_client',
        namespace='robot_policy',
        output='screen',
        emulate_tty=True,
        remappings=[
            (LaunchConfiguration('action_remap_from'),
             LaunchConfiguration('action_remap_to')),
        ],
        parameters=[
            default_rosetta_params,
            {
                'contract_path': LaunchConfiguration('contract_path'),
                'pretrained_name_or_path': LaunchConfiguration('pretrained_name_or_path'),
                'server_address': LaunchConfiguration('server_address'),
                'policy_type': LaunchConfiguration('policy_type'),
                'policy_device': LaunchConfiguration('policy_device'),
                'actions_per_chunk': LaunchConfiguration('actions_per_chunk'),
                'chunk_size_threshold': LaunchConfiguration('chunk_size_threshold'),
                'aggregate_fn_name': LaunchConfiguration('aggregate_fn_name'),
                'feedback_rate_hz': LaunchConfiguration('feedback_rate_hz'),
                'launch_local_server': True,
                'obs_similarity_atol': LaunchConfiguration('obs_similarity_atol'),
            },
        ],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
    )

    # ==================================================================
    # Node 2: Reward classifier (rosetta_client_node) - conditional
    # ==================================================================

    # Use main contract_path when reward_classifier_contract_path is empty
    reward_contract = PythonExpression([
        "'", LaunchConfiguration('reward_classifier_contract_path'), "' if '",
        LaunchConfiguration('reward_classifier_contract_path'),
        "' else '", LaunchConfiguration('contract_path'), "'",
    ])

    reward_classifier_node = LifecycleNode(
        package='rosetta',
        executable='rosetta_client_node',
        name='rosetta_client',
        namespace='reward_classifier',
        output='screen',
        emulate_tty=True,
        condition=IfCondition(
            EqualsSubstitution(LaunchConfiguration('enable_reward_classifier'), 'true')
        ),
        remappings=[
            (LaunchConfiguration('reward_remap_from'),
             LaunchConfiguration('reward_remap_to')),
        ],
        parameters=[
            default_rosetta_params,
            {
                'contract_path': reward_contract,
                'pretrained_name_or_path': LaunchConfiguration(
                    'reward_classifier_pretrained_name_or_path'
                ),
                'server_address': LaunchConfiguration('reward_classifier_server_address'),
                'policy_type': LaunchConfiguration('reward_classifier_policy_type'),
                'policy_device': LaunchConfiguration('policy_device'),
                'actions_per_chunk': LaunchConfiguration('actions_per_chunk'),
                'chunk_size_threshold': LaunchConfiguration('chunk_size_threshold'),
                'aggregate_fn_name': LaunchConfiguration('aggregate_fn_name'),
                'feedback_rate_hz': LaunchConfiguration('feedback_rate_hz'),
                'launch_local_server': True,
                'obs_similarity_atol': LaunchConfiguration('obs_similarity_atol'),
                'is_classifier': True,
            },
        ],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
    )

    # ==================================================================
    # Node 3: Episode recorder
    # ==================================================================
    # Records from real (non-remapped) topics - captures muxed output.

    episode_recorder_node = LifecycleNode(
        package='rosetta',
        executable='episode_recorder_node',
        name='episode_recorder',
        namespace='',
        output='screen',
        emulate_tty=True,
        parameters=[
            default_recorder_params,
            {
                'contract_path': LaunchConfiguration('contract_path'),
                'bag_base_dir': LaunchConfiguration('bag_base_dir'),
                'storage_id': LaunchConfiguration('storage_id'),
                'default_max_duration': LaunchConfiguration('default_max_duration'),
                'feedback_rate_hz': LaunchConfiguration('feedback_rate_hz'),
            },
        ],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
    )

    # ==================================================================
    # Node 4: HIL manager
    # ==================================================================

    hil_manager_node = LifecycleNode(
        package='rosetta',
        executable='rosetta_hil_manager_node',
        name='hil_manager',
        namespace='',
        output='screen',
        emulate_tty=True,
        parameters=[
            default_hil_params,
            {
                'contract_path': LaunchConfiguration('contract_path'),
                'enable_reward_classifier': LaunchConfiguration('enable_reward_classifier'),
                'policy_remap_prefix': LaunchConfiguration('policy_remap_prefix'),
                'reward_remap_prefix': LaunchConfiguration('reward_remap_prefix'),
                'human_reward_positive': LaunchConfiguration('human_reward_positive'),
                'human_reward_negative': LaunchConfiguration('human_reward_negative'),
                'feedback_rate_hz': LaunchConfiguration('feedback_rate_hz'),
            },
        ],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
    )

    # ==================================================================
    # Lifecycle auto-configure / auto-activate
    # ==================================================================
    # Chain: process start -> configure -> activate for each node

    nodes = [robot_policy_node, episode_recorder_node, hil_manager_node]
    # reward_classifier_node handles its own condition internally

    lifecycle_events = []

    for node in nodes:
        configure_event = EmitEvent(
            event=ChangeState(
                lifecycle_node_matcher=matches_action(node),
                transition_id=Transition.TRANSITION_CONFIGURE,
            ),
            condition=IfCondition(
                EqualsSubstitution(LaunchConfiguration('configure'), 'true')
            ),
        )

        activate_event = EmitEvent(
            event=ChangeState(
                lifecycle_node_matcher=matches_action(node),
                transition_id=Transition.TRANSITION_ACTIVATE,
            ),
            condition=IfCondition(
                EqualsSubstitution(LaunchConfiguration('activate'), 'true')
            ),
        )

        lifecycle_events.append(
            RegisterEventHandler(
                OnProcessStart(
                    target_action=node,
                    on_start=[configure_event],
                )
            )
        )
        lifecycle_events.append(
            RegisterEventHandler(
                OnExecutionComplete(
                    target_action=configure_event,
                    on_completion=[activate_event],
                )
            )
        )

    # Reward classifier lifecycle (conditional node)
    reward_configure_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(reward_classifier_node),
            transition_id=Transition.TRANSITION_CONFIGURE,
        ),
        condition=IfCondition(
            EqualsSubstitution(LaunchConfiguration('configure'), 'true')
        ),
    )

    reward_activate_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(reward_classifier_node),
            transition_id=Transition.TRANSITION_ACTIVATE,
        ),
        condition=IfCondition(
            EqualsSubstitution(LaunchConfiguration('activate'), 'true')
        ),
    )

    lifecycle_events.append(
        RegisterEventHandler(
            OnProcessStart(
                target_action=reward_classifier_node,
                on_start=[reward_configure_event],
            )
        )
    )
    lifecycle_events.append(
        RegisterEventHandler(
            OnExecutionComplete(
                target_action=reward_configure_event,
                on_completion=[reward_activate_event],
            )
        )
    )

    # ==================================================================
    # Assemble launch description
    # ==================================================================

    return LaunchDescription(
        launch_args
        + [
            robot_policy_node,
            reward_classifier_node,
            episode_recorder_node,
            hil_manager_node,
        ]
        + lifecycle_events
    )
