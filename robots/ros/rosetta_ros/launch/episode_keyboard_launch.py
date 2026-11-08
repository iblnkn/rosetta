# Copyright 2026 Franco Cipollone
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
Launch file for episode_keyboard_node - keyboard controller for the episode recorder.

Run this in a dedicated terminal alongside the episode recorder. It drives the
recorder's record_episode action and calls its delete_last_bag service, and
lets you control recording with simple key presses.

Keys:
  r / →        Start recording
  s / ←        Stop and save
  d / Backspace  Discard episode (stop + delete bag)
  t            Edit task prompt for next episode
  h / ?        Help
  q            Quit

Usage:
    # Default — /record_episode and /episode_recorder/delete_last_bag
    ros2 launch rosetta_ros episode_keyboard_launch.py

    # A recorder launched under a namespace
    ros2 launch rosetta_ros episode_keyboard_launch.py \
        recorder_ns:=/my_robot/episode_recorder recorder_action_name:=/my_robot/record_episode

    # Pre-set task prompt
    ros2 launch rosetta_ros episode_keyboard_launch.py default_prompt:="pick up the cube"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "recorder_ns",
                default_value="/episode_recorder",
                description="The episode recorder node's full name; delete_last_bag is resolved under it",
            ),
            DeclareLaunchArgument(
                "recorder_action_name",
                default_value="/record_episode",
                description="The recorder's RecordEpisode action",
            ),
            DeclareLaunchArgument(
                "default_prompt",
                default_value="",
                description="Initial task prompt used when starting recordings",
            ),
            Node(
                package="rosetta_ros",
                executable="episode_keyboard_node",
                name="episode_keyboard",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {
                        "recorder_ns": LaunchConfiguration("recorder_ns"),
                        "recorder_action_name": LaunchConfiguration("recorder_action_name"),
                        "default_prompt": LaunchConfiguration("default_prompt"),
                    }
                ],
            ),
        ]
    )
