# Add a dataset writer

Write a dataset writer so `rosetta_port` writes your dataset format. A writer
is a Python class that implements `DatasetWriter` from `rosetta.data`, has a
constructor that takes no arguments, and is registered under the
`rosetta.dataset_writers` entry point. `lerobot_dataset_rosetta` is the full
example.

The writer imports no ROS: `rosetta_port` replays bags into frames and hands
them over. A format that trains straight from recordings, the way
`rerun_rosetta` reads `.rrd` bags, needs no writer at all. To run policies as
well, write a policy runner too ([Add a policy runner](add-a-policy-runner.md)).
The [Python API](../reference/python-api.md) lists the names you can import.

## Write the writer

`rosetta_port` calls `open` once, then for each bag a run of `add_frame`
calls ending in `save_episode`, then `finalize` once:

```python
# my_writer/writer.py
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

## Register it

Advertise the class under a format name in your package's `setup.py`:

```python
entry_points={
    "rosetta.dataset_writers": ["myformat = my_writer.writer:MyWriter"],
},
```

Rebuild or reinstall the package after adding the entry point. Check that
Rosetta finds it:

```bash
python3 -c "from rosetta.data import available_dataset_writers; print(available_dataset_writers())"
```

A class missing part of the protocol is rejected when it loads:

```text
TypeError: Entry point 'myformat' in group 'rosetta.dataset_writers' (my_writer) does not implement DatasetWriter (missing or non-callable: discard_episode, finalize, save_episode).
```

## Use it

Port bags with your writer:

```bash
ros2 run rosetta_ros rosetta_port --format myformat --raw-dir bags --contract robot.yaml --repo-id my_dataset
```

Contracts don't name a dataset format, so a contract written for LeRobot
ports unchanged to yours.

## Package it

A writer is an ordinary Python package, so it can live in its own
repository with its own releases. Depend on `rosetta`. Add `rosetta_ros` only
as a test dependency, and only if your tests load `robot_interface: ros2`
contracts. To build it in a ROS 2 workspace with colcon, give it a
`package.xml` with build type `ament_python`, as
[`data/lerobot/lerobot_dataset_rosetta`](https://github.com/iblnkn/rosetta/tree/main/data/lerobot/lerobot_dataset_rosetta)
does.

Rosetta's
[CONTRIBUTING.md](https://github.com/iblnkn/rosetta/blob/main/CONTRIBUTING.md)
says when a package belongs in the `rosetta` repository instead.
