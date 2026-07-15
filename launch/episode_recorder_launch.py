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

r"""
Launch file for EpisodeRecorderNode - records ROS2 topics to rosbag.

This is a lifecycle node. By default, it auto-configures and auto-activates.
Set configure:=false activate:=false for manual lifecycle control.

Configuration is loaded from params/episode_recorder.yaml (source of truth).
Launch arguments override only deployment-specific settings (paths, sim time, etc.).
Algorithm/tuning parameters should be set in the YAML file.

Usage:
    # Launch with default params file
    ros2 launch rosetta episode_recorder_launch.py

    # Use custom params file
    ros2 launch rosetta episode_recorder_launch.py \\
        params_file:=/path/to/custom_params.yaml

    # Override deployment-specific settings
    ros2 launch rosetta episode_recorder_launch.py \\
        contract_path:=/path/to/contract.yaml \\
        bag_base_dir:=/custom/dataset/path \\
        use_sim_time:=true

    # Manual lifecycle control
    ros2 launch rosetta episode_recorder_launch.py \\
        configure:=false activate:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode
from rosetta.robots.ros2.launch_utils import autostart_handlers, typed_config


def launch_setup(context, *args, **kwargs):
    """Build node with conditional parameter overrides."""
    # Resolve launch configurations in context
    params_file = LaunchConfiguration("params_file").perform(context)
    contract_path = LaunchConfiguration("contract_path").perform(context)
    bag_base_dir = LaunchConfiguration("bag_base_dir").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context)
    log_level = LaunchConfiguration("log_level").perform(context)

    # YAML first; overrides carry only the non-empty launch args (empty =
    # keep the params-file value).
    overrides = {"contract_path": contract_path}
    if bag_base_dir:
        overrides["bag_base_dir"] = bag_base_dir
    if use_sim_time:
        overrides["use_sim_time"] = typed_config(context, "use_sim_time", bool)

    episode_recorder_node = LifecycleNode(
        package="rosetta",
        executable="episode_recorder_node",
        name="episode_recorder",
        namespace="",
        output="screen",
        emulate_tty=True,
        parameters=[params_file, overrides],
        arguments=["--ros-args", "--log-level", log_level],
    )

    return [episode_recorder_node, *autostart_handlers(episode_recorder_node)]


def generate_launch_description():
    """Generate the launch description for the episode recorder node."""
    share = get_package_share_directory("rosetta")
    default_contract = os.path.join(share, "contracts", "so_101.yaml")
    default_params = os.path.join(share, "params", "episode_recorder.yaml")

    # Declare launch arguments
    # Only deployment-specific settings are exposed as launch args
    # Algorithm/tuning parameters should be set in the params YAML file
    launch_description = [
        # Parameters file path - source of truth for tuning params
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="Path to ROS2 parameters YAML file (contains tuning params)",
        ),
        # Deployment-specific paths
        DeclareLaunchArgument(
            "contract_path",
            default_value=default_contract,
            description="Path to robot contract YAML file",
        ),
        DeclareLaunchArgument(
            "bag_base_dir",
            default_value="",  # Empty = use value from params file
            description="Directory for rosbag output (empty = use params file value)",
        ),
        # Runtime settings
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="",  # Empty = use value from params file
            description="Use simulated time from /clock topic (empty = use params file value)",
        ),
        DeclareLaunchArgument(
            "log_level",
            default_value="info",
            description="Logging level (debug, info, warn, error)",
        ),
        # Lifecycle control
        DeclareLaunchArgument("configure", default_value="true", description="Auto-configure node on startup"),
        DeclareLaunchArgument(
            "activate",
            default_value="true",
            description="Auto-activate node after configure (requires configure:=true)",
        ),
    ]

    # Use OpaqueFunction to build node with conditional parameter overrides
    launch_description.append(OpaqueFunction(function=launch_setup))

    return LaunchDescription(launch_description)
