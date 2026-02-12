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
Launch arguments override only deployment-specific settings (paths, server address, etc.).
Algorithm/tuning parameters should be set in the YAML file.

Usage:
    # Launch with default params file
    ros2 launch rosetta rosetta_client_launch.py

    # Use custom params file
    ros2 launch rosetta rosetta_client_launch.py \\
        params_file:=/path/to/custom_params.yaml

    # Override deployment-specific settings
    ros2 launch rosetta rosetta_client_launch.py \\
        contract_path:=/path/to/contract.yaml \\
        pretrained_name_or_path:=/path/to/model \\
        server_address:=192.168.1.100:8080

    # Manual lifecycle control
    ros2 launch rosetta rosetta_client_launch.py \\
        configure:=false activate:=false

    # Connect to remote server (don't launch local)
    ros2 launch rosetta rosetta_client_launch.py \\
        launch_local_server:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler, OpaqueFunction
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch_ros.event_handlers import OnStateTransition
from launch.events import matches_action
from launch.substitutions import EqualsSubstitution, LaunchConfiguration
from launch_ros.actions import LifecycleNode
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition


def launch_setup(context, *args, **kwargs):
    """Build node with conditional parameter overrides."""
    
    # Resolve launch configurations in context
    params_file = LaunchConfiguration('params_file').perform(context)
    contract_path = LaunchConfiguration('contract_path').perform(context)
    pretrained_name_or_path = LaunchConfiguration('pretrained_name_or_path').perform(context)
    server_address = LaunchConfiguration('server_address').perform(context)
    launch_local_server = LaunchConfiguration('launch_local_server').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context)
    log_level = LaunchConfiguration('log_level').perform(context)
    
    # Build parameters list
    parameters = [params_file]  # Load YAML first
    
    # Build override dict with only non-empty values
    overrides = {'contract_path': contract_path}  # Always override contract
    
    if pretrained_name_or_path:  # Only add if non-empty
        overrides['pretrained_name_or_path'] = pretrained_name_or_path
    
    if server_address:  # Only add if non-empty
        overrides['server_address'] = server_address
    
    if launch_local_server:  # Only add if non-empty
        # Convert string to boolean
        overrides['launch_local_server'] = launch_local_server.lower() in ('true', '1', 'yes')
    
    if use_sim_time:  # Only add if non-empty
        # Convert string to boolean
        overrides['use_sim_time'] = use_sim_time.lower() in ('true', '1', 'yes')
    
    if overrides:
        parameters.append(overrides)
    
    # Create the lifecycle node
    rosetta_client_node = LifecycleNode(
        package='rosetta',
        executable='rosetta_client_node',
        name='rosetta_client',
        namespace='',
        output='screen',
        emulate_tty=True,
        parameters=parameters,
        arguments=['--ros-args', '--log-level', log_level],
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
        OnStateTransition(
            target_lifecycle_node=rosetta_client_node,
            goal_state='inactive',   # trigger when node reaches INACTIVE (configure finished)
            entities=[activate_event],
        )
    )

    return [
        rosetta_client_node,
        configure_event_handler,
        activate_event_handler,
    ]


def generate_launch_description():
    share = get_package_share_directory('rosetta')
    default_contract = os.path.join(share, 'contracts', 'so_101.yaml')
    default_params = os.path.join(share, 'params', 'rosetta_client.yaml')

    # Declare launch arguments
    # Only deployment-specific settings are exposed as launch args
    # Algorithm/tuning parameters should be set in the params YAML file
    launch_description = [
        # Parameters file path - source of truth for tuning params
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Path to ROS2 parameters YAML file (contains tuning params)'
        ),
        # Deployment-specific paths
        DeclareLaunchArgument(
            'contract_path',
            default_value=default_contract,
            description='Path to robot contract YAML file'
        ),
        DeclareLaunchArgument(
            'pretrained_name_or_path',
            default_value='',  # Empty = use value from params file
            description='Path or HF repo ID of trained policy (empty = use params file value)'
        ),
        # Server configuration
        DeclareLaunchArgument(
            'server_address',
            default_value='',  # Empty = use value from params file
            description='Policy server address host:port (empty = use params file value)'
        ),
        DeclareLaunchArgument(
            'launch_local_server',
            default_value='',  # Empty = use value from params file
            description='Launch local policy server (true/false, empty = use params file value)'
        ),
        # Runtime settings
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='',  # Empty = use value from params file
            description='Use simulated time from /clock topic (empty = use params file value)'
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='Logging level (debug, info, warn, error)'
        ),
        # Lifecycle control
        DeclareLaunchArgument(
            'configure',
            default_value='true',
            description='Auto-configure node on startup'
        ),
        DeclareLaunchArgument(
            'activate',
            default_value='true',
            description='Auto-activate node after configure (requires configure:=true)'
        ),
    ]

    # Use OpaqueFunction to build node with conditional parameter overrides
    launch_description.append(OpaqueFunction(function=launch_setup))

    return LaunchDescription(launch_description)
