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

"""
Setup for lerobot_dataset_rosetta - Rosetta's LeRobot dataset writer.

Not named with a LeRobot discovery prefix ('lerobot_robot_', 'lerobot_camera_',
'lerobot_teleoperator_', 'lerobot_policy_', 'lerobot_env_'), so LeRobot never
imports it. Rosetta loads it by name through the rosetta.dataset_writers
entry point.
"""

import os

from setuptools import find_packages, setup

package_name = "lerobot_dataset_rosetta"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "lerobot[dataset]<0.6.2",  # the same LeRobot the policy runner pins
        "rosetta",
    ],
    entry_points={
        "rosetta.dataset_writers": [
            "lerobot = lerobot_dataset_rosetta.dataset_writer:LeRobotDatasetWriter",
        ],
    },
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    author="Isaac Blankenau",
    author_email="isaac.blankenau@gmail.com",
    maintainer="Isaac Blankenau",
    maintainer_email="isaac.blankenau@gmail.com",
    keywords=["ros2", "lerobot", "robotics", "rosetta", "dataset"],
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
    ],
    description="Rosetta's LeRobot dataset writer - ports recordings to a LeRobotDataset.",
    license="Apache-2.0",
)
