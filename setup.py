from setuptools import setup
import os
from glob import glob

package_name = 'rosetta'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        # Install marker file in the package index
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # Include our package.xml file
        (os.path.join('share', package_name), ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Include all contract files
        (os.path.join('share', package_name, 'contracts'), glob('contracts/*.yaml')),
    ],
    # This is important as well
    install_requires=['setuptools'],
    zip_safe=True,
    author='ros',
    author_email='isaac.blankenau@gmail.com',
    maintainer='ros',
    maintainer_email='isaac.blankenau@gmail.com',
    keywords=['ros2', 'lerobot', 'robotics', 'policy', 'recording'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
    ],
    description='ROS 2–LeRobot bridge: contract-driven policy runner, episode recorder to rosbag2 (MCAP), and bag→LeRobot exporter using shared decoders/resampling.',
    license='Apache-2.0',
    # Like the CMakeLists add_executable macro, you can add your python
    # scripts here.
    entry_points={
        'console_scripts': [
            'episode_recorder = rosetta.episode_recorder:main',
            'policy_bridge_node = rosetta.policy_bridge_node:main',
        ],
    },
)