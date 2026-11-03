#!/usr/bin/env python
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

"""
Setup for rerun_rosetta - Rerun dataset backend for Rosetta.

Compiles a rosetta contract into a Rerun catalog dataloader query so
policies train directly from reflected .rrd recordings registered to a
Rerun server, with no export step. Registers RerunReader as the "rerun"
rosetta.frame_readers entry point.
"""

import os

from setuptools import find_packages, setup

package_name = "rerun_rosetta"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "rosetta",  # contract/specs/frames machinery; parity comes from reusing it
        "rerun-sdk[dataloader,catalog]>=0.41",
        "torch",  # ContractFrames is a torch IterableDataset
    ],
    entry_points={
        "rosetta.frame_readers": [
            "rerun = rerun_rosetta.reader:RerunReader",
        ],
    },
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    author="Isaac Blankenau",
    author_email="isaac.blankenau@gmail.com",
    maintainer="Isaac Blankenau",
    maintainer_email="isaac.blankenau@gmail.com",
    keywords=["ros2", "rerun", "robotics", "rosetta"],
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
    ],
    description="Rerun dataset backend for Rosetta - trains directly from contract-queried rrd recordings.",
    license="Apache-2.0",
)
