# Install

Install Rosetta one of three ways. Pick by what you already have.

| You have | Use |
|---|---|
| Nothing, and you want to run the tutorial | [ros-physical-ai/demos](#the-demos-workspace) |
| Nothing, and you have your own robot | [rosetta_ws](#rosetta_ws) |
| A ROS 2 workspace already | [An existing workspace](#an-existing-workspace) |

`main` is developed and tested on ROS 2 Jazzy (Python 3.12). CI runs no other
distro.

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

The demos pin their own Rosetta commits in `pai.repos`. Those commits
predate the `rosetta_ros` package these docs use, so commands such as
`ros2 launch rosetta_ros ...` fail there until the demos update their pins.

## rosetta_ws

[rosetta_ws](https://github.com/iblnkn/rosetta_ws) installs ROS 2 Jazzy
through [RoboStack](https://robostack.github.io/), LeRobot 0.6.1, torch and
every Rosetta package with [pixi](https://pixi.sh). It runs on Linux x86_64,
Linux aarch64 and macOS arm64.

```bash
git clone https://github.com/iblnkn/rosetta_ws.git && cd rosetta_ws
pixi run --frozen setup    # clone the packages, install the environment
pixi run build             # colcon build into install/
```

Use `--frozen` the first time. Without it, pixi checks the whole lockfile
before LeRobot's checkout under `libs/` exists, and fails. After setup, plain
`pixi run` works.

Run ROS commands inside `pixi shell`, or prefix them with `pixi run`. The
environment sets `RMW_IMPLEMENTATION=rmw_zenoh_cpp`, so start the Zenoh
router in its own terminal and leave it running:

```bash
pixi run start-zenoh
```

## An existing workspace

1. Clone the repository into your workspace's `src/`, install the ROS
   dependencies, and build `rosetta_ros` with what it depends on:

   ```bash
   git clone https://github.com/iblnkn/rosetta.git src/rosetta
   rosdep install --from-paths src --ignore-src -y --skip-keys ament_python
   colcon build --packages-up-to rosetta_ros
   source install/setup.bash
   ```

   `ament_python` has no rosdep key, so rosdep stops on it unless you skip
   it. `rosetta_ros` has the nodes, launch files and `rosetta_port`, and
   brings `rosetta` and `rosetta_interfaces` with it. To build everything,
   including `rosetta_ros_hil` and the LeRobot packages, drop
   `--packages-up-to rosetta_ros`.

2. For LeRobot, install it into the same Python that ROS uses. `rosetta_port`
   with the LeRobot writer, `policy_runner_node` with the LeRobot runner, and
   the LeRobot plugins all need it:

   ```bash
   pip install "lerobot[dataset,async]<0.6.2"
   ```

   LeRobot 0.6.2 removes `lerobot.async_inference`, which the LeRobot policy
   runner is built on, so the LeRobot packages require `lerobot<0.6.2`. CI
   tests the newest release under that cap, and LeRobot's `main` branch as an
   early warning. Add the `training` extra if you'll run `lerobot-train`.

   Ubuntu 24.04 marks its system Python as externally managed, so pip there
   stops with `error: externally-managed-environment`. Rosetta's CI passes
   `--break-system-packages`, which installs into the system Python anyway.

## Check the install

List the recorder's launch arguments, then load a bundled contract:

```bash
ros2 launch rosetta_ros episode_recorder_launch.py --show-args
python3 -c "from rosetta.contract.schema import load_contract; load_contract('$(ros2 pkg prefix rosetta_ros)/share/rosetta_ros/contracts/so_101.yaml'); print('OK')"
```

The first command lists arguments from `params_file` to `activate`. The
second prints `OK`.

With `lerobot_rosetta` built, `ros2 run rosetta_ros rosetta_port --help`
lists `lerobot` after `Installed:` under `--framework`. That list comes from
the built adapters and does not check that LeRobot itself imports.
