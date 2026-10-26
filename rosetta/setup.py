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
    install_requires=["setuptools", "numpy", "pyyaml"],
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
