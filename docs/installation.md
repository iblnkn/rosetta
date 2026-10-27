# Install

Three ways to get Rosetta. Pick by what you already have.

| You have | Use |
|---|---|
| Nothing, and you want to run the tutorial | [ros-physical-ai/demos](#the-demos-workspace) |
| Nothing, and you have your own robot | [rosetta_ws](#rosetta_ws) |
| A ROS 2 workspace already | [An existing workspace](#an-existing-workspace) |

`main` is developed and tested on ROS 2 Jazzy (Python 3.12). It moves to
Rolling, as ROS repositories usually track, once LeRobot supports Python 3.14,
the Python of Rolling on Ubuntu 26.04. Branches for released distros come with
the first ROS release.

## The demos workspace

[ros-physical-ai/demos](https://github.com/ros-physical-ai/demos) installs
ROS 2 Lyrical, Gazebo, Rosetta and LeRobot through [pixi](https://pixi.sh),
with a simulated SO-ARM101 and cameras. Linux only. You'll want a GPU for
training.

```bash
git clone https://github.com/ros-physical-ai/demos && cd demos
pixi install
pixi run install-deps    # imports the source packages, installs torch for your GPU
pixi run build
```

Run ROS commands inside `pixi shell`, or prefix them with `pixi run`. The
Zenoh router has to be up first, in its own terminal:

```bash
pixi run zenoh-router
```

## rosetta_ws

[rosetta_ws](https://github.com/iblnkn/rosetta_ws) installs ROS 2 Jazzy
through [RoboStack](https://robostack.github.io/), LeRobot 0.6.0 and every
Rosetta package with [pixi](https://pixi.sh). Linux x86_64, Linux aarch64 and
macOS arm64.

```bash
git clone https://github.com/iblnkn/rosetta_ws.git && cd rosetta_ws
pixi run --frozen setup    # clone the packages, install the environment
pixi run build             # colcon build into install/
```

Use `--frozen` the first time, before LeRobot's checkout exists. After that,
plain `pixi run` works. Run ROS commands inside `pixi shell`, or prefix them
with `pixi run`. The workspace uses `rmw_zenoh_cpp`, so start the router in
its own terminal:

```bash
pixi run start-zenoh
```

## An existing workspace

Import the packages listed in
[`rosetta_ws/repos/src.repos`](https://github.com/iblnkn/rosetta_ws/blob/main/repos/src.repos)
into your `src/`, then:

```bash
rosdep install --from-paths src --ignore-src -y --skip-keys ament_python
colcon build --packages-up-to rosetta_ros
```

On ROS 2, `rosetta_ros` is the package to build and install: it has the nodes,
launch files and `rosetta_port`, and brings `rosetta` and `rosetta_interfaces`
with it. `rosetta` alone is the core library, with no nodes. Add
`rosetta_ros_hil` for human-in-the-loop episodes, and the LeRobot packages for
LeRobot.

`rosetta` itself doesn't import a learning framework. For `rosetta_port`
with the LeRobot writer, `policy_runner_node` and the LeRobot plugins, install
into the same Python:

```bash
pip install "lerobot[dataset,async]" torch grpcio
```

CI tests the newest LeRobot release, and LeRobot's main branch as an early
warning.

Add the `training` extra if you'll run `lerobot-train`.

## Check

```bash
ros2 launch rosetta_ros episode_recorder_launch.py --show-args
python -c "from rosetta.contract.schema import load_contract; load_contract('$(ros2 pkg prefix rosetta_ros)/share/rosetta_ros/contracts/so_101.yaml'); print('OK')"
```

The second command prints `OK`.
