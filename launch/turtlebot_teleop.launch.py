import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    joy_vel = LaunchConfiguration('joy_vel')
    joy_config = LaunchConfiguration('joy_config')
    joy_dev = LaunchConfiguration('joy_dev')
    publish_stamped_twist = LaunchConfiguration('publish_stamped_twist')
    config_filepath = LaunchConfiguration('config_filepath')

    return LaunchDescription([
        # Args (keep parity with the original)
        DeclareLaunchArgument('joy_vel', default_value='cmd_vel'),
        DeclareLaunchArgument('joy_config', default_value='ps3'),
        DeclareLaunchArgument('joy_dev', default_value='0'),
        DeclareLaunchArgument('publish_stamped_twist', default_value='false'),
        DeclareLaunchArgument(
            'config_filepath',
            default_value=[
                TextSubstitution(text=os.path.join(
                    get_package_share_directory('teleop_twist_joy'),
                    'config', '')),
                joy_config,
                TextSubstitution(text='.config.yaml')
            ]
        ),

        # Joystick node
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            parameters=[
                {
                    'device_id': joy_dev,
                    'deadzone': 0.1,
                    'autorepeat_rate': 20.0,
                },
                # Keep supporting per-controller button/axis maps
                config_filepath
            ],
        ),

        # Teleop node (load standard config, then override)
        Node(
            package='teleop_twist_joy',
            executable='teleop_node',
            name='teleop_twist_joy_node',
            parameters=[
                # Load the selected joystick mapping first…
                config_filepath,
                # …then override with our changes
                {
                    # Your requests:
                    'require_enable_button': False,   # no deadman
                    'enable_turbo_button': -1,        # keep turbo disabled

                    # Halve speeds (defaults are 0.5 linear / 0.5 yaw -> 0.25 each)
                    'scale_linear.x': 0.5,
                    'scale_angular.yaw': 0.65,

                    # If someone does enable turbo later, keep it at half of default
                    'scale_linear_turbo.x': 0.5,
                    'scale_angular_turbo.yaw': 0.5,

                    # Pass-through from arg
                    'publish_stamped_twist': publish_stamped_twist,
                }
            ],
            # Remap output velocity topic if requested
            remappings=[
                ('/cmd_vel', joy_vel),
            ],
        ),
    ])
