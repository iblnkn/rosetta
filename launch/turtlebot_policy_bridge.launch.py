from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    share = get_package_share_directory('rosetta')
    # contract = os.path.join(share, 'contracts', 'fomo_test.yaml')
    contract = os.path.join(share, 'contracts', 'fomo_test2.yaml')
    log_level_arg = DeclareLaunchArgument(
            'log_level',
            default_value='info',  # Default log level
            description='Logging level for the node (e.g., debug, info, warn, error, fatal)'
        )
    #ensure --ros-args --log-level policy_bridge:=DEBUG
    return LaunchDescription([
        log_level_arg,
        Node(
            package='rosetta',
            executable='policy_bridge_node',
            name='policy_bridge',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'contract_path': contract},
                # {'policy_path': 'robot-com/kiwibot_test'},
                {'policy_path': 'robot-com/kiwibot_test_maincam'},
                {'use_sim_time': True},
                #TODO add inference specific parameters here
            ],
            arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        ),
    ])
