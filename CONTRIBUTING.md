Any contribution that you make to this repository will
be under the Apache 2 License, as dictated by that
[license](http://www.apache.org/licenses/LICENSE-2.0.html):

~~~
5. Submission of Contributions. Unless You explicitly state otherwise,
   any Contribution intentionally submitted for inclusion in the Work
   by You to the Licensor shall be under the terms and conditions of
   this License, without any additional terms or conditions.
   Notwithstanding the above, nothing herein shall supersede or modify
   the terms of any separate license agreement you may have executed
   with Licensor regarding such Contributions.
~~~

## Where an adapter lives

Robot interfaces and framework adapters register through entry points, so
Rosetta finds them the same way wherever they live.

- **In this repository**, under `robots/<middleware>/` or
  `policies/<framework>/`, when the adapter is maintained together with the
  core: its CI runs on every core change, and core changes update it in the
  same commit.
- **In your own repository** for everything else: you choose the release
  schedule and the dependencies, register the same entry points, and nothing
  in Rosetta changes. Open a pull request adding it to the docs' Packages page
  so people can find it.

The how-to guides "Add a framework" and "Add a robot interface" cover both.

## Releasing

The ROS packages release through bloom, which releases every package in this
repository at one version. The packages under `policies/` depend on libraries
with no rosdep key (LeRobot, torch), so the ROS buildfarm can't build them:
list them, one per line, in a `<distro>.ignored` file (e.g. `jazzy.ignored`)
on the release repository's default branch. bloom assumes that branch is
`master`; if yours is `main`, check that the file is honored before the first
release. Those packages install with pip alongside a ROS 2 workspace instead.
