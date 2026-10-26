import os
from glob import glob

from setuptools import find_packages, setup

# The ROS 2 robot interface for Rosetta: codecs, the topic bridge, the recorder
# and policy runner nodes, the bag porter, and the node toolkit apps build on.
package_name = "rosetta_ros"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        # Install marker file in the package index
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        # Include our package.xml file
        (os.path.join("share", package_name), ["package.xml"]),
        # Include all launch files
        (os.path.join("share", package_name, "launch"), glob("launch/*_launch.py")),
        # Include all contract files
        (os.path.join("share", package_name, "contracts"), glob("contracts/*.yaml")),
        # Include all parameter files
        (os.path.join("share", package_name, "params"), glob("params/*.yaml")),
    ],
    install_requires=["setuptools", "numpy", "pyyaml", "rclpy", "rosetta"],
    entry_points={
        "console_scripts": [
            "episode_recorder_node = rosetta_ros.nodes.episode_recorder_node:main",
            "episode_keyboard_node = rosetta_ros.nodes.episode_keyboard_node:main",
            "policy_runner_node = rosetta_ros.nodes.policy_runner_node:main",
            "rosetta_port = rosetta_ros.offline.port:main",
        ],
        "rosetta.robot_interfaces": [
            "ros2 = rosetta_ros.interface:Ros2Interface",
        ],
    },
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    author="Isaac Blankenau",
    author_email="isaac.blankenau@gmail.com",
    maintainer="Isaac Blankenau",
    maintainer_email="isaac.blankenau@gmail.com",
    keywords=["ros2", "robotics", "rosetta"],
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
    ],
    description="The ROS 2 robot interface for Rosetta: codecs, topic bridge, recorder, policy runner and bag porter.",
    license="Apache-2.0",
)
