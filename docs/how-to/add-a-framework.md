# Add a framework

Write a framework adapter so `rosetta_port` writes your learning framework's
dataset format and `policy_runner_node` runs its policies. An adapter is a
Python package with two classes, a dataset writer and a policy runner, and
two entry points. `lerobot_rosetta` is the full example.

Both classes implement protocols from `rosetta.policies`, and both need a
constructor that takes no arguments. The adapter imports no ROS: the robot
side reaches it only through frames. The
[Python API](../reference/python-api.md) lists the names you can import.

## Write the dataset writer

`rosetta_port` calls `open` once, then for each bag a run of `add_frame`
calls ending in `save_episode`, then `finalize` once:

```python
# my_adapter/writer.py
import json
from pathlib import Path


class MyWriter:
    def open(self, *, contract, repo_id, root=None, contract_path=None, embed_contract=True, **opts):
        self._out = Path(root or ".") / repo_id
        self._episodes = []
        self._frames = []

    def add_frame(self, frame):
        self._frames.append(sorted(frame))

    def save_episode(self):
        self._episodes.append(len(self._frames))
        self._frames = []

    def discard_episode(self):
        self._frames = []

    def finalize(self):
        self._out.mkdir(parents=True, exist_ok=True)
        (self._out / "episodes.json").write_text(json.dumps(self._episodes))
```

What the writer has to handle:

- `frame` maps each contract key to a `np.ndarray`, or a `str` for text, plus
  `is_first`, `is_last` and `is_terminal` as `(1,)` bool arrays and `task` as
  a string.
- `contract` is the loaded `Contract`. `iter_specs(contract)` from
  `rosetta.contract.specs` gives every key `add_frame` receives.
  `iter_policy_specs(contract)` gives only observations and actions, for a
  format that builds state and action vectors.
- `opts` always holds `push_to_hub`, `hub_private`, `vcodec` and
  `streaming_encoding` from the `rosetta_port` command line, plus any of the
  writer-specific flags that were set. Ignore the ones you don't use.
- With `embed_contract` true and a `contract_path`, copy the contract into the
  output, so a policy trained on it can find its contract later.
- `discard_episode` drops a bag that failed partway. It's also called when
  nothing is buffered.
- `finalize` isn't called when every bag fails, so don't leave saved episodes
  depending on it.

## Write the policy runner

`policy_runner_node` calls `setup` at configure, `run` once per goal,
`feedback` at `feedback_rate_hz` while a goal runs, and `teardown` at
cleanup or shutdown:

```python
# my_adapter/runner.py
from rosetta.policies import RunnerFeedback, RunnerResult


class MyRunner:
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

- `host` holds the runner's settings. Declare each one with a default, then
  read it with `host.get`. Under ROS 2 each setting is a parameter of
  `policy_runner_node`, so params files and `--ros-args -p` set it.
  `declare` does nothing for a name that already exists, so `setup` runs
  again cleanly after a cleanup and configure.
- `frames` is the live robot, a `FrameIO`: `sample_frame()` returns the
  current observation frame, `warmed_up` is true once every observation
  stream has delivered a value, and `publish_frame()` publishes one action
  frame keyed by contract key.
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

## Register them

Advertise both classes under one framework name in your package's
`setup.py`:

```python
entry_points={
    "rosetta.dataset_writers": ["myfw = my_adapter.writer:MyWriter"],
    "rosetta.policy_runners": ["myfw = my_adapter.runner:MyRunner"],
},
```

Rebuild or reinstall the package after adding the entry points. Check that
Rosetta finds them:

```bash
python3 -c "from rosetta.policies import available_dataset_writers, available_policy_runners; print(available_dataset_writers(), available_policy_runners())"
```

A class missing part of a protocol is rejected when it loads:

```text
TypeError: Entry point 'myfw' in group 'rosetta.policy_runners' (my_adapter) does not implement PolicyRunner (missing or non-callable: feedback, request_stop, run, teardown).
```

## Use them

Port bags with your writer:

```bash
ros2 run rosetta_ros rosetta_port --framework myfw --raw-dir bags --contract robot.yaml --repo-id my_dataset
```

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

An adapter is an ordinary Python package, so it can live in its own
repository with its own releases. Depend on `rosetta`. Add `rosetta_ros` only
as a test dependency, and only if your tests load `robot_interface: ros2`
contracts. To build the adapter in a ROS 2 workspace with colcon, give it a
`package.xml` with build type `ament_python`, as
[`policies/lerobot/lerobot_rosetta`](https://github.com/iblnkn/rosetta/tree/main/policies/lerobot/lerobot_rosetta)
does.

Rosetta's
[CONTRIBUTING.md](https://github.com/iblnkn/rosetta/blob/main/CONTRIBUTING.md)
says when an adapter belongs in the `rosetta` repository instead.
