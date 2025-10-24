from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    share = get_package_share_directory('rosetta')
    contract = os.path.join(share, 'contracts', 'turtlebot.yaml')
    return LaunchDescription([
        Node(
            package='rosetta',
            executable='episode_recorder',
            name='episode_recorder',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'contract_path': contract},
                {'bag_base_dir': '/workspaces/reo_ws/datasets/bags'},
                {'episode_seconds': 10},
                {'use_sim_time': True},  
                ],
        ),
    ])
