# Message types and operators

The message types Rosetta decodes and encodes, the operators an `apply` list
can name, and the API for registering more. The decoders and encoders ship
with the `ros2` robot interface in `rosetta_ros`. The operators and the
registration API are part of core `rosetta`. The contract keys that use them
are on [Contract](contract.md).

## Decoders

A decoder turns a message into an array or a string. Observations, record-only
sections and `teleop.input` load only for a type with a built-in decoder or a
[custom decoder](#custom-codecs). Porting decodes every source, actions
included.

| Type | dtype | `select` | Entries |
|---|---|---|---|
| `sensor_msgs/msg/Image` | `video` | not used | Whole image. Encodings `rgb8`, `bgr8`, `rgba8`, `bgra8`, `mono8`, `8UC1`, matched without case. Output is RGB uint8, alpha dropped, mono copied to three channels. |
| `sensor_msgs/msg/CompressedImage` | `video` | not used | Whole image, decoded with OpenCV (`cv2`, required). Channel order comes from the `format` string, BGR when it names none. |
| `sensor_msgs/msg/JointState` | `float64` | required | `position.<joint>`, `velocity.<joint>`, `effort.<joint>`. A bare `<joint>` means position. Joints are looked up by name. |
| `trajectory_msgs/msg/JointTrajectory` | `float64` | required | First point only. `position.<joint>`, `velocity.<joint>`, `acceleration.<joint>`, `effort.<joint>`. The plurals `positions`, `velocities` and `accelerations` work too. A bare `<joint>` means position. |
| `control_msgs/msg/MultiDOFCommand` | `float64` | required | `values.<dof>`, `values_dot.<dof>`. A bare `<dof>` means `values`. |
| `sensor_msgs/msg/Joy` | `float32` | required | `axes.<i>`, `buttons.<i>`. A bare `<i>` means `axes`, and YAML needs it quoted (`'0'`), since every `select` entry must be a string. Buttons decode as floats. |
| `sensor_msgs/msg/Imu` | `float64` | required | Dotted path from the message root, such as `orientation.x` or `angular_velocity.z`. |
| `nav_msgs/msg/Odometry` | `float64` | required | Dotted path, such as `pose.pose.position.x` or `twist.twist.angular.z`. |
| `geometry_msgs/msg/Twist` | `float64` | required | Dotted path, such as `linear.x` or `angular.z`. |
| `geometry_msgs/msg/TwistStamped` | `float64` | required | Dotted path from the root, so `twist.linear.x`, not `linear.x`. |
| `std_msgs/msg/Float32MultiArray` | `float32` | optional | The whole `data` array. `select` entries only name the elements, and their count must equal the array length. Without `select` only a one-element array decodes. |
| `std_msgs/msg/Float64MultiArray` | `float64` | optional | Same as `Float32MultiArray`. |
| `std_msgs/msg/Int32MultiArray` | `int32` | optional | Same as `Float32MultiArray`. |
| `std_msgs/msg/Float32`, `Float64`, `Int32`, `Int64`, `Bool` | `float32`, `float64`, `int32`, `int64`, `bool` | optional | One value from `data`. A `select` entry only names it. |
| `std_msgs/msg/String` | `string` | not used | `data` as text. |

A message that fails to decode is dropped. The failure is logged once per
stream, and the recovery is logged when a message decodes again. These fail
per message:

- A depth image: `Image` encodings `mono16`, `16UC1`, `32FC1`, `32FC`, and a
  `CompressedImage` whose `format` names a depth encoding or
  `compressedDepth`.
- An `Image` with an empty or unlisted encoding.
- A joint or DOF name that is not in the message.
- A `JointTrajectory` with no points.
- A `Joy` index that is negative or past the end of the array.
- A value whose width differs from the source's width.

## Encoders

An encoder turns an action vector into a message. `actions` and
`teleop.feedback` load only for a type with a built-in encoder or a custom
encoder.

| Type | `select` | Header stamp | Notes |
|---|---|---|---|
| `sensor_msgs/msg/JointState` | required | set | Each selected field must cover every named joint. Unselected fields stay empty. |
| `trajectory_msgs/msg/JointTrajectory` | required | set | One point, `time_from_start` zero. Full coverage, as for `JointState`. |
| `control_msgs/msg/MultiDOFCommand` | required | none | Full coverage, as for `JointState`. |
| `sensor_msgs/msg/Joy` | required | set | Each array is sized to its highest selected index. Gaps are zero. Buttons are rounded to the nearest integer. |
| `geometry_msgs/msg/Twist` | required | none | Dotted path, as for decoding. |
| `geometry_msgs/msg/TwistStamped` | required | set | Dotted path from the root, such as `twist.linear.x`. |
| `std_msgs/msg/Float32`, `Float64` | optional | none | The source must be one value wide. |
| `std_msgs/msg/Float32MultiArray`, `Float64MultiArray` | optional | none | `data` is the whole vector. |
| `std_msgs/msg/Int32MultiArray` | optional | none | `data` is the whole vector, rounded to the nearest integer. |

The stamp is the node clock at publish time. Coverage, width and the
one-value rule fail the encode, not the load.

Encoding runs the inverse `apply` pipeline, then a width check, then a
finiteness check. A live action frame with NaN or Inf in any value is
dropped whole and logged, and nothing from it is published. If frames keep
failing, the watchdog publishes each channel's `safety` action.

## Operators

| Operator | Form | Inverse | On actions | Behavior |
|---|---|---|---|---|
| `rad2deg` | `rad2deg` | degrees to radians | yes | Takes no arguments. Round-trip checked at load. |
| `clamp` | `clamp: {min: <lo>, max: <hi>}` | the same clip | yes | Element-wise clip to `[lo, hi]`. Exactly the keys `min` and `max`, both finite numbers, `min <= max`. Keeps the input dtype. NaN passes through. |
| `resize` | `resize: [<h>, <w>]` | none | no | Nearest-neighbour resize. Image keys only. `h` and `w` are integers from 1 to 8192. Sets the image size the dataset stores. |

Decoding runs `apply` front to back. Encoding runs it back to front through
the inverses. Each operator has one of three invertibility kinds:

| Kind | Inverse | Allowed on `actions`, `teleop.feedback` |
|---|---|---|
| `FORWARD_ONLY` | none | no |
| `BIDIRECTIONAL` | yes, need not undo `forward` | yes |
| `BIJECTIVE` | yes, must undo `forward`, checked when the contract loads | yes |

## Custom codecs

Register a decoder for a type Rosetta does not ship, and an encoder too if the
type is used as an action or feedback.

```python
import numpy as np
from geometry_msgs.msg import Point

from rosetta.frames.codecs import register_decoder, register_encoder


@register_decoder("geometry_msgs/msg/Point", dtype="float64")
def decode_point(msg, spec):
    return np.array([msg.x, msg.y, msg.z], dtype=np.float64)


@register_encoder("geometry_msgs/msg/Point")
def encode_point(values, spec, stamp_ns=None):
    return Point(x=float(values[0]), y=float(values[1]), z=float(values[2]))
```

A source of that type then declares `select: [x, y, z]` to make it three wide.

`register_decoder(type_str, dtype, *, override=False, requires_select=False)`

| Argument | Meaning |
|---|---|
| `type_str` | Message type as the contract writes it. |
| `dtype` | One of `float32`, `float64`, `int32`, `int64`, `bool`, `video`, `string`. Checked at registration. |
| `override` | Replace the codec already registered for `type_str`. Without it, a second registration is a `ValueError`. With it and nothing registered, also a `ValueError`. |
| `requires_select` | A source of this type without `select` fails at load. |

`register_encoder(type_str, *, override=False, requires_select=False)` takes
the same arguments without `dtype`.

| Function | Signature | Returns |
|---|---|---|
| Decoder | `fn(msg, spec)` | A numpy array as wide as the source, or a `str`. `None` is an error. |
| Encoder | `fn(values, spec, stamp_ns=None)` | A message. `values` is a float64 array that already ran the inverse `apply` pipeline and passed the width and finiteness checks. |

`spec.names` is the `select` list as a tuple, empty without `select`.
`spec.dim` is the source's width. `spec.source.channel` is the parsed
`channel` block.

There are two ways to make Rosetta use a codec:

- Advertise the module under the entry-point group `rosetta.codecs`, with the
  module path as the value. Rosetta imports every module in the group on the
  first contract load, and the contract names only the type. A module that
  fails to import fails every contract load in that process.
- Name the function on the channel with `decoder: my_pkg.codecs:decode_point`
  or `encoder: my_pkg.codecs:encode_point`. The module must be importable.
  The path is checked at load, and it wins over the registry for that channel.
  A channel with a custom decoder resolves to `float64` unless it sets
  `dtype`.

## Custom operators

```python
from rosetta.contract.errors import ContractValidationError
from rosetta.contract.operators import Invertibility, Operator, register_operator


@register_operator("scale", kind=Invertibility.BIJECTIVE)
class ScaleOperator(Operator):
    def __init__(self, args, ctx):
        if not isinstance(args, (int, float)) or isinstance(args, bool) or args == 0:
            raise ContractValidationError(f"scale expects a non-zero number, got {args!r}")
        self.k = float(args)

    def forward(self, arr):
        return arr * self.k

    def inverse(self, arr):
        return arr / self.k
```

A contract then writes `apply: [scale: 1000]`.

`register_operator(name, *, kind, override=False)` registers the class under
`name`. A name already registered is a `ValueError` unless `override=True`. A
class without `forward`, or without `inverse` for a kind other than
`FORWARD_ONLY`, is a `TypeError`.

| Member | Meaning |
|---|---|
| `__init__(self, args, ctx)` | `args` is the YAML value after the name, `None` for a bare name. The base class rejects any arguments. `ctx.is_image` is true on an image key. |
| `forward(self, arr)` | The decode direction. |
| `inverse(self, arr)` | The encode direction. |
| `output_hw` | `(h, w)` when the operator sets a fixed image size, else `None`. An image key needs at least one operator that sets it. |
| `sample_input(self)` | The array the `BIJECTIVE` round-trip check runs on. Override it when the operator's domain is restricted. |

Advertise the module under the entry-point group `rosetta.operators`, with
the module path as the value. Rosetta imports the group the first time a
contract names an operator. A module that fails to import fails every
contract with an `apply` list in that process.
