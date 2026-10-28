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

## Run the checks

CI runs the workflows in `.github/workflows/`. `core`, `ros` and `lerobot`
run on pull requests that touch their paths, so a core change runs all three.

| Workflow | Runs | On changes to |
|---|---|---|
| `core` | `pytest rosetta/test` and a wheel build, on plain Python 3.12 and 3.13 with no ROS | `rosetta/` |
| `ros` | Builds and tests `rosetta` and `robots/ros` with industrial_ci on ROS 2 Jazzy | `rosetta/`, `robots/` |
| `lerobot` | Builds and tests `rosetta`, `robots/ros` and `policies/lerobot` with industrial_ci on ROS 2 Jazzy, against LeRobot's newest release before 0.6.2 and against LeRobot's `main`, which is allowed to fail. Also runs nightly. | `rosetta/`, `robots/ros/`, `policies/lerobot/` |
| `lint` | The `pre-commit` hooks: ruff check and format, plus file checks for whitespace, line endings, YAML, XML, merge conflicts and large files | every pull request |
| `docs` | Builds the docs site with warnings as errors and publishes it | pushes to `main` |

To run the same checks locally from the repository root:

```bash
pre-commit run --all-files           # lint, the same hooks as CI
pip install ./rosetta pytest         # core, in a plain Python environment
pytest rosetta/test
pip install -r docs/requirements.txt # docs
sphinx-build -W -b html docs _build/html
```

`pre-commit install` runs the lint hooks on every commit. The ROS packages
build and test with colcon in a ROS 2 workspace. In
[`rosetta_ws`](https://github.com/iblnkn/rosetta_ws), that's
`pixi run build-with-tests`, then `pixi run test`.

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
