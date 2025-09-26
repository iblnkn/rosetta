from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    share = get_package_share_directory('rosetta')
    contract = os.path.join(share, 'contracts', 'example.yaml')
    return LaunchDescription([
        Node(
            package='rosetta',
            executable='policy_bridge',
            name='policy_bridge',
            output='screen',
            emulate_tty=True,
            parameters=[{'contract_path': contract}],
        ),
    ])
