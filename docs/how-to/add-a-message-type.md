# Add a message type

Write a codec for a message type Rosetta doesn't ship, so a contract can name
the type in a channel. The built-in types are listed in
[Message types](../reference/message-types.md).

A contract that names a type with no codec fails to load:

```text
rosetta.contract.errors.ContractValidationError: No decoder registered for 'sensor_msgs/msg/BatteryState' in observation 'observation.battery'. Add a decoder in decoders.py, install a 'rosetta.codecs' entry-point plugin that registers one, or set a custom 'decoder:' on the channel.
```

You need a Python package installed in the same environment as Rosetta, and
the message type's Python module importable there.

## Write the codec

A decoder turns a message into a flat array. An encoder turns an array back
into a message, for the types Rosetta publishes: actions, teleop feedback,
and rewards from a reward classifier.

This module adds a decoder for `sensor_msgs/msg/BatteryState`, and an encoder
and a decoder for `geometry_msgs/msg/Point`:

```python
# my_pkg/codecs.py
import numpy as np

from rosetta.frames.codecs import register_decoder, register_encoder


@register_decoder("sensor_msgs/msg/BatteryState", dtype="float32", requires_select=True)
def decode_battery_state(msg, spec):
    return np.array([getattr(msg, name) for name in spec.names], dtype=np.float32)


@register_decoder("geometry_msgs/msg/Point", dtype="float64", requires_select=True)
def decode_point(msg, spec):
    return np.array([getattr(msg, name) for name in spec.names], dtype=np.float64)


@register_encoder("geometry_msgs/msg/Point", requires_select=True)
def encode_point(values, spec, stamp_ns=None):
    from geometry_msgs.msg import Point

    msg = Point()
    for name, value in zip(spec.names, values):
        setattr(msg, name, float(value))
    return msg
```

What the code has to get right:

- `spec.names` is the channel's `select` list as a tuple, and `()` without
  `select`. The decoder returns exactly one value per name, or one value when
  there is no `select`. A message that decodes to the wrong width, or makes
  the decoder raise, is dropped with one warning per stream until a message
  decodes again.
- `requires_select=True` makes a channel of this type without `select` fail
  at load. Pass it when the codec reads fields by name, as these do.
- `dtype` is the stream's contract dtype: `float32`, `float64`, `int32`,
  `int64`, `bool`, `video` or `string`. A channel's own `dtype:` overrides it.
- `values` in the encoder has already run back through the channel's `apply`
  list, so it holds the values in message units.
- Give an action type a decoder as well as an encoder. Serving needs only the
  encoder, but `rosetta_port` decodes every channel it records, and it fails
  each bag with `contract streams cannot be decoded for porting` when a
  channel has no decoder.
- Import message classes inside the function, as `encode_point` does. Rosetta
  imports codec modules every time a contract loads, and a module that fails
  to import fails every contract in that environment.

## Register it

Advertise the module under the `rosetta.codecs` entry-point group in your
package's `setup.py`:

```python
entry_points={
    "rosetta.codecs": ["my_pkg = my_pkg.codecs"],
},
```

Rosetta reads entry points from the installed package's metadata, so rebuild
or reinstall the package after adding one. Every contract that names the type
then loads, with nothing extra in the contract.

To use a codec in one contract only, name the function on the channel
instead, and give the module no entry point:

```yaml
observations:
  observation.battery:
    channel: {topic: /battery, type: sensor_msgs/msg/BatteryState,
              decoder: "my_pkg.codecs:decode_battery_state", dtype: float32}
    align: {strategy: hold, timeline: header}
    select: [voltage, percentage]
```

An inline `decoder:` wins over the registry for that channel. Without a
`dtype:` on the channel, an inline decoder's stream is `float64`. The
function needs no decorator, and `encoder:` works the same way on an action
channel.

## Check it

Load a contract that uses the type:

```bash
python3 -c "from rosetta.contract.schema import load_contract; load_contract('robot.yaml'); print('OK')"
```

`robot.yaml` is your contract. The command prints `OK` when every channel
resolves. Problems with the codec show up here, for example:

```text
rosetta.contract.errors.ContractValidationError: 'sensor_msgs/msg/BatteryState' in observation 'observation.battery': its decoder gathers values by selected field names; add 'select:' to declare them.
rosetta.contract.errors.ContractValidationError: Codec function not found for 'my_pkg.codecs:nope' in observations.observation.battery.channel.decoder: module 'my_pkg.codecs' has no attribute 'nope'
```

A type that already has a codec can't be registered again, and the failed
plugin fails every contract load in that environment:

```text
rosetta.contract.errors.ContractValidationError: Failed to load codec plugin 'my_pkg' (my_pkg.codecs) from entry-point group 'rosetta.codecs': Decoder already registered for 'sensor_msgs/msg/JointState'. Pass override=True to replace it intentionally.
```

To replace a built-in codec on purpose, pass `override=True` to
`register_decoder` or `register_encoder`. To replace it for one channel only,
use an inline `decoder:` or `encoder:`.
