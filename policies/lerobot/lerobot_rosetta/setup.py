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
Setup for lerobot_rosetta - Rosetta's LeRobot backend adapter.

Deliberately NOT named with a 'lerobot_robot_*'/'lerobot_teleoperator_*'
prefix: those names are reserved for LeRobot's plugin auto-discovery. The
LeRobot-discovered Robot plugin lives in lerobot_robot_rosetta; this package
is the other direction - it plugs LeRobot into rosetta via entry points.
"""

import os

from setuptools import find_packages, setup

package_name = "lerobot_rosetta"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "lerobot",
        "rclpy",
        "rosetta",  # Depends on rosetta.contract/frames/policies
        "lerobot_robot_rosetta",  # RosettaConfig; Robot plugin discovered by lerobot at runtime
        "grpcio",  # policy_server.py / classifier_server.py
        "torch",  # classifier_server.py
    ],
    entry_points={
        # Rosetta framework adapters: resolved by name by the framework-agnostic
        # node / porter so rosetta core never imports lerobot.
        "rosetta.policy_runners": [
            "lerobot = lerobot_rosetta.policy_runner:LeRobotPolicyRunner",
        ],
        "rosetta.dataset_writers": [
            "lerobot = lerobot_rosetta.dataset_writer:LeRobotDatasetWriter",
        ],
        "console_scripts": [
            "rosetta_classifier_server = lerobot_rosetta.classifier_server:main",
            "rosetta_policy_server = lerobot_rosetta.policy_server:main",
        ],
    },
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    author="Isaac Blankenau",
    author_email="isaac.blankenau@gmail.com",
    maintainer="Isaac Blankenau",
    maintainer_email="isaac.blankenau@gmail.com",
    keywords=["ros2", "lerobot", "robotics", "rosetta"],
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
    ],
    description="Rosetta's LeRobot backend adapter - dataset writer, policy runner, and inference servers.",
    license="Apache-2.0",
)
