# Install

Three ways to get Rosetta. Pick by what you already have.

| You have | Use |
|---|---|
| Nothing, and you want to run the tutorial | [ros-physical-ai/demos](#the-demos-workspace) |
| Nothing, and you have your own robot | [rosetta_ws](#rosetta_ws) |
| A ROS 2 workspace already | [An existing workspace](#an-existing-workspace) |

Rosetta builds on ROS 2 Jazzy and Lyrical.

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
colcon build
```

`rosetta` itself doesn't import a learning framework. For `rosetta_port`
with the LeRobot writer, `policy_runner_node` and the LeRobot plugins, install
into the same Python:

```bash
pip install "lerobot[dataset,async]==0.6.0" torch grpcio
```

Add the `training` extra if you'll run `lerobot-train`.

## Check

```bash
ros2 launch rosetta episode_recorder_launch.py --show-args
python -c "from rosetta.contract.schema import load_contract; load_contract('$(ros2 pkg prefix rosetta)/share/rosetta/contracts/so_101.yaml'); print('OK')"
```

The second command prints `OK`.
