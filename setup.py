from setuptools import setup, find_packages
import os
from glob import glob

# Main Rosetta package - provides ROS2 nodes and common utilities
# The LeRobot plugins are in separate packages:
#   - lerobot_robot_rosetta: Robot plugin
#   - lerobot_teleoperator_rosetta: Teleoperator plugin
#   - rosetta_rl: RL training components
package_name = 'rosetta'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        # Install marker file in the package index
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # Include our package.xml file
        (os.path.join('share', package_name), ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*_launch.py')),
        # Include all contract files
        (os.path.join('share', package_name, 'contracts'), glob('contracts/*.yaml')),
        # Include all parameter files
        (os.path.join('share', package_name, 'params'), glob('params/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy', 'pyyaml', 'rclpy'],
    zip_safe=True,
    author='Isaac Blankenau',
    author_email='isaac.blankenau@gmail.com',
    maintainer='Isaac Blankenau',
    maintainer_email='isaac.blankenau@gmail.com',
    keywords=['ros2', 'lerobot', 'robotics', 'rosetta'],
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
    ],
    description='Rosetta: ROS 2 utilities, common contract handling, and nodes for LeRobot integration.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'episode_recorder_node = rosetta.episode_recorder_node:main',
            'rosetta_client_node = rosetta.rosetta_client_node:main',
            'rosetta_hil_manager_node = rosetta.rosetta_hil_manager_node:main',
            'port_bags = rosetta.port_bags:main',
        ],
    },
)
