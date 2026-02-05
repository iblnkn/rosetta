#!/usr/bin/env python
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
Setup for lerobot_robot_rosetta - LeRobot Robot plugin for Rosetta.

This package name follows LeRobot's auto-discovery convention:
packages starting with 'lerobot_robot_*' are automatically discovered
and registered by LeRobot's register_third_party_plugins().
"""

import os
from setuptools import setup, find_packages

package_name = 'lerobot_robot_rosetta'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'lerobot',
        'rclpy',
        'rosetta',  # Depends on rosetta.common
    ],
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
    description='LeRobot Robot plugin for Rosetta - bridges ROS2 topics to LeRobot Robot interface.',
    license='Apache-2.0',
)
