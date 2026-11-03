# Add a policy runner

Write a policy runner so `policy_runner_node` runs your framework's
policies. A runner is a Python class that implements `PolicyRunner` from
`rosetta.policies`, has a constructor that takes no arguments, and is
registered under the `rosetta.policy_runners` entry point. `lerobot_rosetta`
is the full example.

The runner imports no ROS: the robot reaches it only through frames. Where
your framework hosts a run itself, open the robot with
`load_robot_interface(contract.robot_interface).connect(...)` from
`rosetta.robots.registry` instead, as `lerobot_robot_rosetta` does. To write
your framework's dataset format as well, write a dataset writer too
([Add a dataset writer](add-a-dataset-writer.md)). The
[Python API](../reference/python-api.md) lists the names you can import.

## Write the runner

`policy_runner_node` calls `setup` at configure (after `find_contract` when
it has no `contract_path`), `run` once per goal,
`feedback` at `feedback_rate_hz` while a goal runs, and `teardown` at
cleanup or shutdown:

```python
# my_runner/runner.py
from rosetta.policies import RunnerFeedback, RunnerResult


class MyRunner:
    def find_contract(self, pretrained, *, warn):
        # Where your checkpoints record their contract. None makes the node
        # require contract_path.
        warn(f"MyRunner can't find a contract for {pretrained}")
        return None

    def setup(self, host, contract):
        host.declare("rate_hz", 10.0, "Control loop rate", read_only=True)
        self._period = 1.0 / host.get("rate_hz")
        self._actions = 0
        host.logger.info(f"MyRunner ready at {host.get('rate_hz')} Hz")

    def run(self, frames, *, task, stop_event):
        self._actions = 0
        while not stop_event.wait(self._period):
            if not frames.warmed_up:
                continue
            observation = frames.sample_frame()
            # A real runner calls its policy on observation and publishes the
            # result with frames.publish_frame({"action": ...}).
            self._actions += 1
        return RunnerResult(success=True, message="Stopped")

    def feedback(self):
        return RunnerFeedback(queue_depth=0, published_actions=self._actions)

    def request_stop(self):
        pass

    def teardown(self):
        pass
```

What the runner has to handle:

- `find_contract` runs only when the node has no `contract_path`. Return the
  path of the contract `pretrained` was trained with, or `None`. Report
  each missing link through `warn` rather than raising.
- `host` holds the runner's settings. Declare each one with a default, then
  read it with `host.get`. Under ROS 2 each setting is a parameter of
  `policy_runner_node`, so params files and `--ros-args -p` set it.
  `declare` does nothing for a name that already exists, so `setup` runs
  again cleanly after a cleanup and configure.
- `frames` is the live robot, a `FrameIO`: `sample_frame()` returns the
  current observation frame, `warmed_up` is true once every observation
  stream has delivered a value, `publish_frame()` publishes one action
  frame keyed by contract key, and `clock.simulated` is true when the robot
  runs on a simulator's clock.
- `run` blocks until `stop_event` is set or the policy finishes. A run that
  stops because it was asked to is a success. Return
  `RunnerResult(success=False, message=...)` only for a real failure, which
  the node reports as `termination_reason: error`.
- The node sets `stop_event` and calls `request_stop()` together, from
  another thread. Use `request_stop` to wake a run blocked in I/O, such as a
  socket read. It's also called with no run active, so clear anything it
  latches at the start of `run`.
- `feedback()` is also called from another thread while `run` blocks. Its
  counters are `uint32` in `RunPolicy` feedback, so keep them non-negative.
- `teardown` releases what `setup` started, such as a server process.

## Register it

Advertise the class under a framework name in your package's `setup.py`:

```python
entry_points={
    "rosetta.policy_runners": ["myfw = my_runner.runner:MyRunner"],
},
```

Rebuild or reinstall the package after adding the entry point. Check that
Rosetta finds it:

```bash
python3 -c "from rosetta.policies import available_policy_runners; print(available_policy_runners())"
```

A class missing part of the protocol is rejected when it loads:

```text
TypeError: Entry point 'myfw' in group 'rosetta.policy_runners' (my_runner) does not implement PolicyRunner (missing or non-callable: feedback, request_stop, run, teardown).
```

## Use it

The launch file has no `framework` argument. Set it, and your runner's own
settings, in a params file:

```yaml
# myfw.yaml
policy_runner:
  ros__parameters:
    framework: myfw
    rate_hz: 20.0
```

```bash
ros2 launch rosetta_ros policy_runner_launch.py params_file:=myfw.yaml contract_path:=robot.yaml
```

The node logs `Configured: contract=robot.yaml, framework=myfw` when it
configures. An unknown framework name fails with a `ValueError` that lists
the installed names.

Contracts don't name a framework, so a contract written for LeRobot runs
unchanged on yours.

## Package it

A runner is an ordinary Python package, so it can live in its own
repository with its own releases. Depend on `rosetta`. Add `rosetta_ros` only
as a test dependency, and only if your tests load `robot_interface: ros2`
contracts. To build it in a ROS 2 workspace with colcon, give it a
`package.xml` with build type `ament_python`, as
[`policies/lerobot/lerobot_rosetta`](https://github.com/iblnkn/rosetta/tree/main/policies/lerobot/lerobot_rosetta)
does.

Rosetta's
[CONTRIBUTING.md](https://github.com/iblnkn/rosetta/blob/main/CONTRIBUTING.md)
says when a package belongs in the `rosetta` repository instead.
