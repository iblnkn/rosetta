# Installation

Rosetta is a plain ROS 2 Jazzy package set. There are two ways to install it, and
neither is privileged by the code — the difference is only who resolves the
dependencies.

| Path | Best for | What you get |
|---|---|---|
| [The pixi workspace](#path-1-the-pixi-workspace-easiest) | Getting running fast; anything involving training | ROS 2 Jazzy **and** LeRobot/torch in one reproducible environment, no system ROS |
| [An existing ROS 2 install](#path-2-an-existing-ros-2-install) | You already run ROS 2 Jazzy; robot-side deployment; distro packaging | The Rosetta packages in your own colcon workspace, deps via `rosdep` |

The hard part of the first path is not Rosetta — it is that ROS 2 and the ML
frameworks disagree about numpy, opencv, and torch. `rosetta_ws` exists to
resolve that conflict once, in a committed lockfile. If you are only running
the ROS side (recording bags, deploying a checkpoint served elsewhere), you do
not have that conflict, and path 2 is lighter.

## Path 1: the pixi workspace (easiest)

[`rosetta_ws`](https://github.com/iblnkn/rosetta_ws) is a
[pixi](https://pixi.sh) workspace that installs ROS 2 Jazzy (via
[RoboStack](https://robostack.github.io/)), LeRobot, and every Rosetta package
together. No Docker, no system ROS, no venv juggling.

Install pixi ([one-liner](https://pixi.sh/latest/#installation), v0.72+), then:

```bash
git clone https://github.com/iblnkn/rosetta_ws.git
cd rosetta_ws
pixi run --frozen setup    # clone package/library repos + install the environment
pixi run build             # colcon build -> install/
```

`--frozen` is only needed for that first `setup`; every later command is plain
`pixi run ...`. `pixi shell` drops you into an activated environment (ROS
sourced, overlay sourced, Zenoh RMW selected) for interactive work.

Verify:

```bash
pixi run ros2 pkg list | grep rosetta
```

A VS Code devcontainer is also available in that repo — it is a thin wrapper
that runs the same `pixi run setup`. See the
[`rosetta_ws` README](https://github.com/iblnkn/rosetta_ws) for tasks,
environments, and tab completion.

## Path 2: an existing ROS 2 install

Works with apt-installed ROS 2 Jazzy on Ubuntu 24.04, or any colcon workspace
layout you already have.

### 1. Get the sources

Each package lives in its own repository. The pinned set is listed in
[`repos/src.repos`](https://github.com/iblnkn/rosetta_ws/blob/main/repos/src.repos):

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws
pip install vcstool   # or apt install python3-vcstool
curl -LO https://raw.githubusercontent.com/iblnkn/rosetta_ws/main/repos/src.repos
vcs import src --input src.repos --recursive
```

### 2. Install ROS dependencies

The packages declare their ROS and system dependencies in `package.xml`, so
rosdep handles them:

```bash
sudo apt update
rosdep update --rosdistro=jazzy
rosdep install --from-paths src --ignore-src -y
```

### 3. Install the python frameworks (backend-dependent)

The ML frameworks are deliberately **not** in `package.xml` — `rosetta` core
imports none of them, and the backend leaves discover them at runtime via
entry points. Install only what you need:

| You want to use | Install |
|---|---|
| `rosetta`, `rosetta_interfaces` (core bridge) | nothing extra |
| `lerobot_robot_rosetta`, `lerobot_teleoperator_rosetta` | `pip install 'lerobot[dataset,async]==0.6.0'` |

```{note}
LeRobot and the ROS python stack have overlapping numpy/opencv constraints; a
virtualenv created with `--system-site-packages` on top of the ROS python is
the least painful arrangement. If that fight is not one you want, use path 1 —
resolving it is the entire reason `rosetta_ws` exists.
```

### 4. Build

```bash
cd ~/ros2_ws
colcon build --merge-install --symlink-install
source install/setup.bash
```

## Middleware

The pixi workspace defaults to `rmw_zenoh_cpp`
(`apt install ros-jazzy-rmw-zenoh-cpp` on path 2), whose router runs as its own
process:

```bash
ros2 run rmw_zenoh_cpp rmw_zenohd     # or: pixi run start-zenoh
```

Any RMW works for the core bridge — set `RMW_IMPLEMENTATION` to whichever you
already run.

## Reporting install problems

If a `package.xml` is missing a dependency rosdep should have installed, please
file an issue on that package's repo. The pixi workspace masks that class of
bug, so path-2 reports are how it gets caught.

## Next steps

- [Train and deploy your first policy](tutorials/first-policy.md) — the full pipeline, once through.
- [Write a contract](how-to/write-a-contract.md) — the one YAML that drives every component.
