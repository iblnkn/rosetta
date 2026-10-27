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

import os

from setuptools import find_packages, setup

# Rosetta's core: the contract, frames, and the protocols and registries robot
# interfaces (e.g. rosetta_ros) and framework adapters (e.g. lerobot_rosetta)
# plug into. Imports no robot middleware.
package_name = "rosetta"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        # Install marker file in the package index
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        # Include our package.xml file
        (os.path.join("share", package_name), ["package.xml"]),
    ],
    python_requires=">=3.12",
    install_requires=["numpy", "pyyaml"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    author="Isaac Blankenau",
    author_email="isaac.blankenau@gmail.com",
    maintainer="Isaac Blankenau",
    maintainer_email="isaac.blankenau@gmail.com",
    keywords=["robotics", "rosetta"],
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
    ],
    description=(
        "Rosetta: contract-driven translation between pub/sub robots (ROS2, ...)"
        " and policy-learning frameworks (LeRobot, ...)."
    ),
    license="Apache-2.0",
)
