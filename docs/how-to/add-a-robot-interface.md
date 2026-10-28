# Add a robot interface

Write a robot interface so contracts can describe a robot on a middleware
other than ROS 2, such as MQTT. A robot interface is a package that registers
one class, and a contract selects it by name with `robot_interface:`.
`rosetta_ros` is the full example. The smallest one is
`rosetta/test/fake_interface.py` in the repository.

The examples on this page build a toy interface named `toy`, whose one
message type `toy/Vector` is a dict of named floats.

## Implement the protocol

The class implements `RobotInterface` from `rosetta.robots.protocols`, with a
constructor that takes no arguments:

```python
# rosetta_toy/interface.py
TYPES = {"toy/Vector"}


class ToyInterface:
    def register_codecs(self):
        from . import codecs  # noqa: F401

    def provided_timelines(self, msg_type):
        if msg_type not in TYPES:
            raise ValueError(f"Unknown message type '{msg_type}'")
        return frozenset({"receive"})

    def validate_qos(self, qos):
        if set(qos) - {"retain"}:
            raise ValueError(f"Unknown qos key(s) {sorted(set(qos) - {'retain'})}. Allowed: ['retain']")
```

Contract loading calls all three methods:

- `register_codecs` imports the modules that register your built-in codecs.
  It runs once per process, for every installed interface, before any
  `rosetta.codecs` plugin loads.
- `provided_timelines` returns the timeline names a channel of `msg_type`
  provides, and raises `ValueError` for a type your middleware doesn't know.
  Every channel's `align.timeline` must be in that set. Timelines are open
  names: `ros2` provides `receive` for every type and `header` for types with
  a `std_msgs/Header` field named `header`.
- `validate_qos` raises `ValueError` when a channel's `qos` mapping doesn't
  parse for your middleware.

These run wherever a contract loads, including on machines that only prepare
data. Import your middleware inside the methods, not at module level.

## Register codecs

Decoders turn a message into an array, and encoders turn an array back into a
message. Register them with `register_decoder` and `register_encoder` from
`rosetta.frames.codecs`, in the modules `register_codecs` imports:

```python
# rosetta_toy/codecs.py
import numpy as np

from rosetta.frames.codecs import register_decoder, register_encoder


@register_decoder("toy/Vector", dtype="float64", requires_select=True)
def decode_vector(msg, spec):
    return np.array([msg[name] for name in spec.names], dtype=np.float64)


@register_encoder("toy/Vector", requires_select=True)
def encode_vector(values, spec, stamp_ns=None):
    return {name: float(v) for name, v in zip(spec.names, values)}
```

[Add a message type](add-a-message-type.md) covers what a codec has to get
right.

All interfaces share one codec registry keyed by type string. Interfaces
register in name order, and one that registers a type string another
interface already registered fails. Only contracts that name the failed
interface fail to load:

```text
rosetta.contract.errors.ContractValidationError: Failed to load robot interface 'toy': Decoder already registered for 'std_msgs/msg/Float64'. Pass override=True to replace it intentionally.
```

Choose type strings in your middleware's own form, as `toy/Vector` is here.

## Register the interface

Advertise the class under the `rosetta.robot_interfaces` entry-point group.
The entry-point name is the value contracts use for `robot_interface`:

```python
entry_points={
    "rosetta.robot_interfaces": ["toy = rosetta_toy.interface:ToyInterface"],
},
```

Rebuild or reinstall the package after adding the entry point.

## Check it

Write a contract for the interface:

```yaml
# toy.yaml
robot_type: toy_arm
robot_interface: toy
fps: 10
observations:
  observation.state:
    channel: {topic: arm/state, type: toy/Vector, qos: {retain: true}}
    align: {strategy: hold, timeline: receive}
    select: [j1, j2]
actions:
  action:
    channel: {topic: arm/command, type: toy/Vector}
    align: {strategy: hold, timeline: receive}
    select: [j1, j2]
```

Load it:

```bash
python3 -c "from rosetta.contract.schema import load_contract; load_contract('toy.yaml'); print('OK')"
```

It prints `OK` when the interface accepts every channel. A channel the
interface rejects names the channel and the reason:

```text
rosetta.contract.errors.ContractValidationError: observations.observation.state.align: timeline 'header' is not provided by 'toy/Vector'. This channel provides: ['receive']
rosetta.contract.errors.ContractValidationError: observations.observation.state.qos: Unknown qos key(s) ['qos_level']. Allowed: ['retain']
```

Where the package isn't installed, the contract fails to load:

```text
rosetta.contract.errors.ContractValidationError: robot_interface 'toy' is not installed. Installed: ros2
```

## Connect it to a robot

Loading contracts is all core needs from an interface. Running a robot takes
more, all of it in your package:

- A live bridge that implements `FrameIO` from `rosetta.frames.protocols`:
  `sample_frame()` returns the current observation frame,
  `publish_frame(frame)` publishes one action frame, `warmed_up` is true once
  every observation stream has delivered a value, `send_safety_action()`
  publishes the contract's safety action, and `reset_state()` clears
  per-episode state. Any framework adapter's `PolicyRunner` runs against it
  unchanged.
- A process that hosts a `PolicyRunner` on that bridge.
- A recorder, and a tool that replays recordings as frame dicts into a
  `DatasetWriter`.

In `rosetta_ros` these are `TopicBridge`, `policy_runner_node`,
`episode_recorder_node` and `rosetta_port`.

## Package it

Depend on `rosetta`. Keep the interface in your own repository, or propose it
for `robots/<middleware>/` in the `rosetta` repository. Rosetta's
[CONTRIBUTING.md](https://github.com/iblnkn/rosetta/blob/main/CONTRIBUTING.md)
says which fits.
