# Add a message type

Rosetta ships decoders for the common sensor and command types. For
anything else, register your own. The API is on
[Message types](../reference/message-types.md#custom-codecs).

## Write the codec

A decoder turns a message into an array of the declared dtype with one
value per `select` entry, or one value without `select`. A message that
decodes to the wrong width, or raises, is dropped with one warning per
stream. An encoder does the reverse, and you only need one if the type is an
action.

```python
# my_pkg/codecs.py
import numpy as np
from rosetta.frames.codecs import register_decoder, register_encoder

@register_decoder("my_msgs/msg/MySensor", dtype="float64")
def decode_my_sensor(msg, spec):
    return np.array([msg.field1, msg.field2], dtype=np.float64)

@register_encoder("my_msgs/msg/MyCommand")
def encode_my_command(values, spec, stamp_ns=None):
    from my_msgs.msg import MyCommand
    out = MyCommand()
    out.a, out.b = float(values[0]), float(values[1])
    return out
```

`spec.names` is the contract's `select` list as a tuple, empty when there
was no `select`. Honor it if the type has named fields.

## Register it

Advertise the module as an entry point in your package's `setup.py`, so any
contract naming the type works:

```python
entry_points={
    "rosetta.codecs": ["my_pkg = my_pkg.codecs"],
},
```

Or name the function in the contract:

```yaml
channel:
  topic: /my_sensor
  type: my_msgs/msg/MySensor
  decoder: my_pkg.codecs:decode_my_sensor
```

## Check

```bash
python -c "from rosetta.contract.schema import load_contract; load_contract('robot.yaml'); print('OK')"
```

A bad path, a missing module or a duplicate registration shows up here.
