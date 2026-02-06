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
Launch file for RosettaClientNode - runs LeRobot policy inference.

This is a lifecycle node. By default, it auto-configures and auto-activates.
Set configure:=false activate:=false for manual lifecycle control.

Configuration is loaded from params/rosetta_client.yaml (source of truth).

Usage:
    # Launch with auto-activate (default behavior)
    ros2 launch rosetta rosetta_client_launch.py

    # Launch without auto-activation (manual lifecycle control)
    ros2 launch rosetta rosetta_client_launch.py configure:=false activate:=false

    # Override contract path
    ros2 launch rosetta rosetta_client_launch.py \\
        contract_path:=/path/to/contract.yaml

    # Connect to remote server instead of launching local
    ros2 launch rosetta rosetta_client_launch.py \\
        launch_local_server:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnExecutionComplete, OnProcessStart
from launch.events import matches_action
from launch.substitutions import EqualsSubstitution, LaunchConfiguration
from launch_ros.actions import LifecycleNode
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition


def generate_launch_description():
    share = get_package_share_directory('rosetta')
    default_contract = os.path.join(share, 'contracts', 'so_101.yaml')
    default_params = os.path.join(share, 'params', 'rosetta_client.yaml')

    # Declare launch arguments
    # Defaults come from params file - launch args override when provided
    launch_description = [
        # Contract path - deployment-specific
        DeclareLaunchArgument(
            'contract_path',
            default_value=default_contract,
            description='Path to contract YAML file'
        ),
        # Node parameters (defaults from params/rosetta_client.yaml)
        DeclareLaunchArgument(
            'pretrained_name_or_path',
            default_value='/workspaces/rosetta_ws/models/act/act_pen_in_cup/050000/pretrained_model',
            description='HuggingFace model ID or local path to trained model'
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
            'feedback_rate_hz',
            default_value='2.0',
            description='Execution feedback publish rate (Hz)'
        ),
        DeclareLaunchArgument(
            'launch_local_server',
            default_value='true',
            description='Launch local policy server automatically (set false for remote server)'
        ),
        DeclareLaunchArgument(
            'obs_similarity_atol',
            default_value='-1.0',
            description='Observation filtering tolerance (-1.0 to disable)'
        ),
        # Log level
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='Logging level (debug, info, warn, error)'
        ),
        # Lifecycle control
        DeclareLaunchArgument(
            'configure',
            default_value='true',
            description='Whether to auto-configure the node on startup'
        ),
        DeclareLaunchArgument(
            'activate',
            default_value='true',
            description='Whether to auto-activate the node on startup (requires configure:=true)'
        ),
    ]

    # Create the lifecycle node
    # Parameters are loaded from params file first, then launch args override
    rosetta_client_node = LifecycleNode(
        package='rosetta',
        executable='rosetta_client_node',
        name='rosetta_client',
        namespace='',
        output='screen',
        emulate_tty=True,
        parameters=[
            # Load defaults from params file (source of truth)
            default_params,
            # Launch argument overrides (later values take precedence)
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
                'launch_local_server': LaunchConfiguration('launch_local_server'),
                'obs_similarity_atol': LaunchConfiguration('obs_similarity_atol'),
            },
        ],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
    )

    # Auto-configure event (triggered on process start)
    configure_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(rosetta_client_node),
            transition_id=Transition.TRANSITION_CONFIGURE,
        ),
        condition=IfCondition(EqualsSubstitution(LaunchConfiguration('configure'), 'true')),
    )

    # Auto-activate event (triggered after configure completes)
    activate_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(rosetta_client_node),
            transition_id=Transition.TRANSITION_ACTIVATE,
        ),
        condition=IfCondition(EqualsSubstitution(LaunchConfiguration('activate'), 'true')),
    )

    # Chain events: process start -> configure -> activate
    configure_event_handler = RegisterEventHandler(
        OnProcessStart(
            target_action=rosetta_client_node,
            on_start=[configure_event],
        )
    )

    activate_event_handler = RegisterEventHandler(
        OnExecutionComplete(
            target_action=configure_event,
            on_completion=[activate_event],
        )
    )

    launch_description.append(rosetta_client_node)
    launch_description.append(configure_event_handler)
    launch_description.append(activate_event_handler)

    return LaunchDescription(launch_description)
