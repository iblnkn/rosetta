# Add a framework

An adapter connects a learning framework to Rosetta. It's a Python package
with two classes and two entry points. `lerobot_rosetta` is the working
example.

## Implement the two protocols

From `rosetta.policies.protocols`:

```python
class DatasetWriter:
    def open(self, *, contract, repo_id, root=None, contract_path=None,
             embed_contract=True, **opts): ...
    def add_frame(self, frame): ...
    def save_episode(self): ...
    def discard_episode(self): ...
    def finalize(self): ...

class PolicyRunner:
    def setup(self, node, contract): ...
    def run(self, frames, *, task, stop_event): ...   # returns RunnerResult
    def feedback(self): ...                            # returns RunnerFeedback
    def request_stop(self): ...
    def teardown(self): ...
```

`frames` in `run` is a `FrameIO` with `sample_frame()`, `warmed_up`,
`publish_frame(frame)`, `send_safety_action()` and `reset_state()`. Both
classes need a zero-argument constructor.

## Register them

```python
entry_points={
    "rosetta.dataset_writers": ["myfw = my_adapter.writer:MyWriter"],
    "rosetta.policy_runners":  ["myfw = my_adapter.runner:MyRunner"],
},
```

## Use them

```bash
ros2 run rosetta rosetta_port --framework myfw ...
ros2 launch rosetta policy_runner_launch.py ...   # with framework: myfw in the params file
```

An unknown name lists the registered ones. A class missing part of the
protocol is rejected at load. Contracts don't name a framework, so one
written for LeRobot runs unchanged on yours.
