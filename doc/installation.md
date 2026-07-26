# Installation

Rosetta is a plain ROS 2 Jazzy package set. Two paths install it.

## The pixi workspace

[`rosetta_ws`](https://github.com/iblnkn/rosetta_ws) installs ROS 2 Jazzy (via
[RoboStack](https://robostack.github.io/)), LeRobot, and every Rosetta package
together. Take this path if training is anywhere in your plans: ROS 2 and the ML
frameworks disagree about numpy, opencv, and torch, and the workspace resolves
that in a committed lockfile.

Install [pixi](https://pixi.sh) (v0.72+), then:

```bash
git clone https://github.com/iblnkn/rosetta_ws.git
cd rosetta_ws
pixi run    # clone package/library repos + install the environment
pixi run build             # colcon build -> install/
```

## An existing ROS 2 install

Import the pinned package set from
[`repos/src.repos`](https://github.com/iblnkn/rosetta_ws/blob/main/repos/src.repos)
into your own colcon workspace, then:

```bash
rosdep install --from-paths src --ignore-src -y
colcon build
```

The ML frameworks are deliberately not in `package.xml`, because `rosetta` core
imports none of them. Install `lerobot[dataset,async]==0.6.0` if you need the
porter's dataset writer, `policy_runner_node`, or the LeRobot plugins.
