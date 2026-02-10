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

All node parameter defaults are read from params/rosetta_client.yaml (single
source of truth).  Launch arguments override only when explicitly provided:

Usage:
    # Launch with defaults from YAML
    ros2 launch rosetta rosetta_client_launch.py

    # Override a parameter at launch time
    ros2 launch rosetta rosetta_client_launch.py \\
        pretrained_name_or_path:=my-org/my-model

    # Manual lifecycle control
    ros2 launch rosetta rosetta_client_launch.py configure:=false activate:=false
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.events import matches_action
from launch.substitutions import EqualsSubstitution, LaunchConfiguration
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition


def generate_launch_description():
    share = get_package_share_directory('rosetta')
    default_contract = os.path.join(share, 'contracts', 'so_101.yaml')
    default_params = os.path.join(share, 'params', 'rosetta_client.yaml')

    # Read defaults from the params YAML (single source of truth)
    with open(default_params) as f:
        _yaml = yaml.safe_load(f)
    defaults = _yaml['rosetta_client']['ros__parameters']

    # --- Declare launch arguments (defaults pulled from YAML) ---------------
    launch_description = [
        # Contract path - deployment-specific (not in the params YAML)
        DeclareLaunchArgument(
            'contract_path',
            default_value=default_contract,
            description='Path to contract YAML file'
        ),
        # Node parameters - defaults from params/rosetta_client.yaml
        DeclareLaunchArgument(
            'pretrained_name_or_path',
            default_value=str(defaults['pretrained_name_or_path']),
            description='HuggingFace model ID or local path to trained model'
        ),
        DeclareLaunchArgument(
            'server_address',
            default_value=str(defaults['server_address']),
            description='LeRobot policy server address (host:port)'
        ),
        DeclareLaunchArgument(
            'policy_type',
            default_value=str(defaults['policy_type']),
            description='Policy type: act, smolvla, diffusion, pi0, pi05, etc.'
        ),
        DeclareLaunchArgument(
            'policy_device',
            default_value=str(defaults['policy_device']),
            description='Inference device: cuda, cpu, mps, or cuda:0'
        ),
        DeclareLaunchArgument(
            'actions_per_chunk',
            default_value=str(defaults['actions_per_chunk']),
            description='Number of actions per inference chunk'
        ),
        DeclareLaunchArgument(
            'chunk_size_threshold',
            default_value=str(defaults['chunk_size_threshold']),
            description='Threshold for requesting new chunk (0.0-1.0)'
        ),
        DeclareLaunchArgument(
            'aggregate_fn_name',
            default_value=str(defaults['aggregate_fn_name']),
            description='Chunk aggregation: weighted_average, latest_only, average, conservative'
        ),
        DeclareLaunchArgument(
            'feedback_rate_hz',
            default_value=str(defaults['feedback_rate_hz']),
            description='Execution feedback publish rate (Hz)'
        ),
        DeclareLaunchArgument(
            'launch_local_server',
            default_value=str(defaults['launch_local_server']).lower(),
            description='Launch local policy server automatically (set false for remote server)'
        ),
        DeclareLaunchArgument(
            'obs_similarity_atol',
            default_value=str(defaults['obs_similarity_atol']),
            description='Observation filtering tolerance (-1.0 to disable)'
        ),
        # Launch-only arguments (not node params, not in YAML)
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='Logging level (debug, info, warn, error)'
        ),
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

    # --- Lifecycle node (params file + launch-arg overrides) ----------------
    rosetta_client_node = LifecycleNode(
        package='rosetta',
        executable='rosetta_client_node',
        name='rosetta_client',
        namespace='',
        output='screen',
        emulate_tty=True,
        parameters=[
            default_params,
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

    # --- Lifecycle auto-transition events -----------------------------------
    configure_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(rosetta_client_node),
            transition_id=Transition.TRANSITION_CONFIGURE,
        ),
        condition=IfCondition(EqualsSubstitution(LaunchConfiguration('configure'), 'true')),
    )

    activate_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(rosetta_client_node),
            transition_id=Transition.TRANSITION_ACTIVATE,
        ),
        condition=IfCondition(EqualsSubstitution(LaunchConfiguration('activate'), 'true')),
    )

    configure_event_handler = RegisterEventHandler(
        OnProcessStart(
            target_action=rosetta_client_node,
            on_start=[configure_event],
        )
    )

    activate_event_handler = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=rosetta_client_node,
            start_state='configuring',
            goal_state='inactive',
            entities=[activate_event],
        )
    )

    launch_description.append(rosetta_client_node)
    launch_description.append(configure_event_handler)
    launch_description.append(activate_event_handler)

    return LaunchDescription(launch_description)
