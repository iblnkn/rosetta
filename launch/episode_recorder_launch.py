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
Launch file for EpisodeRecorderNode - records ROS2 topics to rosbag.

This is a lifecycle node. By default, it auto-configures and auto-activates.
Set configure:=false activate:=false for manual lifecycle control.

Usage:
    # Launch with auto-activate (default behavior)
    ros2 launch rosetta episode_recorder_launch.py

    # Launch without auto-activation (manual lifecycle control)
    ros2 launch rosetta episode_recorder_launch.py configure:=false activate:=false

    # Override contract path
    ros2 launch rosetta episode_recorder_launch.py contract_path:=/path/to/contract.yaml
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
    default_params = os.path.join(share, 'params', 'episode_recorder.yaml')
    default_bag_base_dir = '/workspaces/rosetta_ws/datasets/bags'
    default_use_sim_time = 'false'

    # Declare launch arguments
    # Defaults come from params file - launch args override when provided
    launch_description = [
        # Contract path - deployment-specific
        DeclareLaunchArgument(
            'contract_path',
            default_value=default_contract,
            description='Path to contract YAML file'
        ),
        # Node parameters (defaults from params/episode_recorder.yaml)
        DeclareLaunchArgument(
            'bag_base_dir',
            default_value=default_bag_base_dir,
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
            description='Max episode duration in seconds'
        ),
        DeclareLaunchArgument(
            'feedback_rate_hz',
            default_value='2.0',
            description='Recording feedback publish rate (Hz)'
        ),
        DeclareLaunchArgument(
            'default_qos_depth',
            default_value='10',
            description='QoS queue depth for subscriptions'
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
            'use_sim_time',
            default_value=default_use_sim_time,
            description='Run node with simulated time (pass to use_sim_time param)'
        ),
        DeclareLaunchArgument(
            'activate',
            default_value='true',
            description='Whether to auto-activate the node on startup (requires configure:=true)'
        ),
    ]

    # Create the lifecycle node
    # Parameters are loaded from params file first, then launch args override
    episode_recorder_node = LifecycleNode(
        package='rosetta',
        executable='episode_recorder_node',
        name='episode_recorder',
        namespace='',
        output='screen',
        emulate_tty=True,
        parameters=[
            # Load defaults from params file
            default_params,
            # Launch argument overrides (later values take precedence)
            {
                'contract_path': LaunchConfiguration('contract_path'),
                'bag_base_dir': LaunchConfiguration('bag_base_dir'),
                'storage_id': LaunchConfiguration('storage_id'),
                'default_max_duration': LaunchConfiguration('default_max_duration'),
                'feedback_rate_hz': LaunchConfiguration('feedback_rate_hz'),
                'default_qos_depth': LaunchConfiguration('default_qos_depth'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            },
        ],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
    )

    # Auto-configure event (triggered on process start)
    configure_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(episode_recorder_node),
            transition_id=Transition.TRANSITION_CONFIGURE,
        ),
        condition=IfCondition(EqualsSubstitution(LaunchConfiguration('configure'), 'true')),
    )

    # Auto-activate event (triggered after configure completes)
    activate_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(episode_recorder_node),
            transition_id=Transition.TRANSITION_ACTIVATE,
        ),
        condition=IfCondition(EqualsSubstitution(LaunchConfiguration('activate'), 'true')),
    )

    # Chain events: process start -> configure -> activate
    configure_event_handler = RegisterEventHandler(
        OnProcessStart(
            target_action=episode_recorder_node,
            on_start=[configure_event],
        )
    )

    activate_event_handler = RegisterEventHandler(
        OnExecutionComplete(
            target_action=configure_event,
            on_completion=[activate_event],
        )
    )

    launch_description.append(episode_recorder_node)
    launch_description.append(configure_event_handler)
    launch_description.append(activate_event_handler)

    return LaunchDescription(launch_description)
