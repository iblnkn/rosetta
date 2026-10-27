# Copyright 2026 Isaac Blankenau
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

import os
from glob import glob

from setuptools import find_packages, setup

package_name = "rosetta_ros_hil"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*_launch.py")),
        (os.path.join("share", package_name, "params"), glob("params/*.yaml")),
        (os.path.join("share", package_name, "contracts"), glob("contracts/*.yaml")),
    ],
    install_requires=["setuptools", "pyyaml", "rclpy"],
    entry_points={
        "console_scripts": [
            "hil_manager_node = rosetta_ros_hil.hil_manager_node:main",
        ],
    },
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    author="Isaac Blankenau",
    author_email="isaac.blankenau@gmail.com",
    maintainer="Isaac Blankenau",
    maintainer_email="isaac.blankenau@gmail.com",
    keywords=["ros2", "rosetta", "human-in-the-loop"],
    description="Human-in-the-loop episode manager built on Rosetta.",
    license="Apache-2.0",
)
