Any contribution that you make to this repository will
be under the Apache 2 License, as dictated by that
[license](http://www.apache.org/licenses/LICENSE-2.0.html):

~~~text
5. Submission of Contributions. Unless You explicitly state otherwise,
   any Contribution intentionally submitted for inclusion in the Work
   by You to the Licensor shall be under the terms and conditions of
   this License, without any additional terms or conditions.
   Notwithstanding the above, nothing herein shall supersede or modify
   the terms of any separate license agreement you may have executed
   with Licensor regarding such Contributions.
~~~

## Development environment

The repository's pixi environment is the development environment: ROS 2
Jazzy, LeRobot and the build tools, pinned in `pixi.lock`. On Linux or macOS,
[install pixi](https://pixi.sh) and work from the repository root:

```bash
pixi run build
pixi run test
```

On Windows, or to work inside a container, use the dev container. On Windows,
clone the repository inside WSL 2 rather than on a Windows drive, because the
build creates symlinks in the bind-mounted repository. Then open it in VS Code
and run Dev Containers: Reopen in Container, which offers two configurations:

| Configuration | Use |
|---|---|
| `rosetta` (`.devcontainer/devcontainer.json`) | Everything except GPU training. |
| `rosetta (NVIDIA GPU)` (`.devcontainer/gpu/devcontainer.json`) | Adds every NVIDIA GPU. Needs the NVIDIA Container Toolkit on the host. |

Both create the same pixi environment on first start and share it, so
`pixi run build` and `pixi run test` work the same as on a host. On a Linux
host the container shares the host network, so ROS nodes inside it see nodes
on the host and the LAN. Docker Desktop on Windows and macOS runs containers
inside a VM, so there the host network is the VM's.

Serial ports, cameras and displays aren't passed through, because they differ
per machine. Add the `--device` or mount you need to `runArgs` or `mounts` in
your local copy of the configuration.

The container and the host share `build/` and `install/` through the bind
mount, but the repository's path differs between them, so run
`pixi run clean` when you switch.

## Run the checks

CI runs the workflows in `.github/workflows/`. All but `lint` and `docs` run
on pull requests that touch their paths, so a core change runs `core`, `ros`,
`lerobot` and `pixi`.

| Workflow | Runs | On changes to |
|---|---|---|
| `core` | `pytest rosetta/test` and a wheel build, on plain Python 3.12 and 3.13 with no ROS | `rosetta/` |
| `ros` | Builds and tests `rosetta` and `robots/ros` with industrial_ci on ROS 2 Jazzy | `rosetta/`, `robots/` |
| `lerobot` | Builds and tests `rosetta`, `robots/ros` and `policies/lerobot` with industrial_ci on ROS 2 Jazzy, against LeRobot's newest release before 0.6.2 and against LeRobot's `main`, which is allowed to fail. Also runs nightly. | `rosetta/`, `robots/ros/`, `policies/lerobot/` |
| `lint` | The `pre-commit` hooks: ruff check and format, plus file checks for whitespace, line endings, YAML, XML, merge conflicts and large files | every pull request |
| `pixi` | `pixi run test`, which builds and tests every package together in the locked pixi environment with LeRobot 0.6.1, then `pixi run docs-build` | `pixi.toml`, `pixi.lock`, `pixi/`, `rosetta/`, `robots/`, `policies/`, `docs/` |
| `devcontainer` | Builds the default dev container and runs `pixi run build` inside it | `.devcontainer/` |
| `docs` | Builds the docs site with warnings as errors and publishes it | pushes to `main` |

To run the same checks locally from the repository root:

```bash
pixi run lint     # the same pre-commit hooks as CI
pixi run test     # every package, as in the pixi job
pixi run docs-build  # the docs site, warnings as errors
```

The core tests also run without ROS, as the `core` job runs them:

```bash
pip install ./rosetta pytest
pytest rosetta/test
```

`pre-commit install` runs the lint hooks on every commit.

The pixi environment reads each package's dependencies from its
`package.xml`. A ROS or conda-forge dependency added there needs no change to
`pixi.toml`: run `pixi install` to pick it up. A pip-only dependency, such as
LeRobot, has no `package.xml` key that pixi resolves, so it also goes in
`pixi.toml`'s `pypi-dependencies`. A new package needs a line in the `[dev]`
table.

## Where an adapter lives

Robot interfaces and framework adapters register through entry points, so
Rosetta finds them the same way wherever they live.

- In this repository, under `robots/<middleware>/` or `policies/<framework>/`,
  when the adapter is maintained together with the core. Its CI runs on
  every core change, and core changes update it in the same commit.
- In your own repository, for everything else. You choose the release
  schedule and the dependencies, register the same entry points, and nothing
  in Rosetta changes. Open a pull request adding it to the Packages page
  (`docs/reference/packages.md`) so people can find it.

The docs guides
[Add a framework](https://iblnkn.github.io/rosetta/how-to/add-a-framework.html)
and [Add a robot interface](https://iblnkn.github.io/rosetta/how-to/add-a-robot-interface.html)
cover both.

## Supported ROS distros

`main` is developed and tested on ROS 2 Jazzy (Python 3.12). It moves to
Rolling, as ROS repositories usually track, once LeRobot supports Python 3.14,
the Python of Rolling on Ubuntu 26.04. Branches for released distros come with
the first ROS release.

## Releasing

The ROS packages release through bloom, which requires every package in this
repository to have the same version. The packages under `policies/` depend on
LeRobot, which has no rosdep key, so the ROS buildfarm can't build them. List
them, one per line, in a `<track>.ignored` file on the release repository's
`master` branch. The track is normally named after the distro, as in
`jazzy.ignored`. bloom reads the file from `master` even when the release
repository's default branch is named differently. Those packages install with
pip alongside a ROS 2 workspace instead.
