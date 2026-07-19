# Add Custom Codecs

This guide shows you how to support a ROS message type beyond the [built-ins](../reference/contract.md#supported-message-types): write a decoder (ROS → numpy) and, for actions, an encoder (numpy → ROS). Codecs are keyed by message type in a registry. There are two ways to register yours.

## Method 1: Plugin via entry points (recommended)

Package your codecs and advertise the module under the `rosetta.codecs` entry-point group. Rosetta discovers and imports the module at contract load, running the `@register_*` decorators, so the contract names the type only, with no module paths in the YAML:

```python
# my_pkg/my_codecs.py
import numpy as np
from rosetta.frames.codecs import register_decoder, register_encoder

@register_decoder("my_msgs/msg/MyCustomSensor", dtype="float64")
def decode_my_sensor(msg, spec):
    return np.array([msg.field1, msg.field2], dtype=np.float64)

@register_encoder("my_msgs/msg/MyCustomCommand")
def encode_my_command(values, spec, stamp_ns=None):
    from my_msgs.msg import MyCustomCommand
    msg = MyCustomCommand()
    msg.field1, msg.field2 = float(values[0]), float(values[1])
    return msg
```

```toml
# in your plugin's pyproject.toml
[project.entry-points."rosetta.codecs"]
my_codecs = "my_pkg.my_codecs"
```

```yaml
observations:
  observation.state:
    channel: {topic: /my_sensor,
              type: my_msgs/msg/MyCustomSensor}  # codec self-registered; no path needed
    align: {strategy: hold, timeline: receive}
```

Registering a second codec for an already-covered type is an error, so two plugins never silently conflict over a type. A plugin whose import fails latches the failure: every later discovery call re-raises the original error until the process restarts. To replace a built-in (e.g. you wrote a better `sensor_msgs/msg/Image` decoder), pass `override=True`:

```python
@register_decoder("sensor_msgs/msg/Image", dtype="video", override=True)
def my_better_image_decoder(msg, spec): ...
```

Registration accepts one more flag. `requires_select=True` marks a codec unable to produce a value without a `select` list (e.g. `JointState`, which needs joint names to know which fields to extract). A contract using such a channel without `select` fails at load instead of at runtime.

## Method 2: Inline path in the contract

Point a single source directly at a codec function. The override applies to one source only and needs no packaging. Use this for a one-off, or to run a different decoder on one topic while the registry default applies elsewhere:

```yaml
actions:
  action:
    channel:
      topic: /my_command
      type: my_msgs/msg/MyCustomCommand
      decoder: my_package.codecs:decode_my_command  # module:function (reading bags)
      encoder: my_package.codecs:encode_my_command  # for publishing
    align: {strategy: hold, timeline: receive}
```

The module must be importable. Paths are validated at contract load time.

> **Trust model: a contract is code-equivalent.** Loading a contract *imports* every named `decoder:`/`encoder:` module (the import is the path validation) and, at runtime, invokes those functions on robot message data. Only load contracts you trust. This matters most for the policy runner's sidecar fallback, which fetches `rosetta_contract.yaml` from a Hugging Face Hub model or dataset repo. A contract downloaded from a third-party repo is treated as trusted input, exactly like a launch file. When a hub-resolved contract declares inline `decoder:`/`encoder:` paths, the runner logs a warning naming each one.

> **Round-trip safety:** every built-in encoder/decoder pair is round-trip tested (`decode(encode(v)) == v`) in the test suite. When you contribute a new built-in pair, add a sample message to those tests.

## Function signatures

**Decoder:** ROS message → numpy array

```python
def my_decoder(msg, spec) -> np.ndarray:
    # msg: ROS message instance
    # spec.names: list of selected field paths from the contract
    # spec.source.channel.type: ROS message type string
    return np.array([...], dtype=np.float64)
```

**Encoder:** numpy array → ROS message

```python
def my_encoder(values, spec, stamp_ns=None):
    # values: numpy array of action values (decode_value/encode_value already ran
    #         spec.operators in the serve/inverse direction, e.g. clamp/deg2rad,
    #         before your encoder sees them)
    # spec.names: list of selected field paths from the contract
    # stamp_ns: optional timestamp in nanoseconds
    msg = MyMessage()
    # ... populate msg from values ...
    return msg
```

When each is used: [inline codec fields](../reference/contract.md#inline-codec-fields). Set `dtype` on custom-decoded channels: [when dtype is required](../reference/contract.md#when-dtype-is-required).
