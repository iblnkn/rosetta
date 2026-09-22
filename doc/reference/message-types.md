# Message types and operators

Built-in decoders, encoders and operators, and how to register your own.

## Decoders

A decoder reads a message into an array. Only these types, and types with a
[custom decoder](#custom-codecs), work as observations or port from a bag. A
message that fails to decode is dropped with one warning per stream.

| Type | dtype | `select` | Field paths |
|---|---|---|---|
| `sensor_msgs/msg/Image` | `video` | no | Whole image. Encodings `rgb8`, `bgr8`, `rgba8`, `bgra8`, `mono8`, `8UC1`. Output is RGB uint8. Depth encodings are an error. |
| `sensor_msgs/msg/CompressedImage` | `video` | no | Whole image, decoded with OpenCV. `compressedDepth` is an error. |
| `sensor_msgs/msg/JointState` | `float64` | required | `position.<joint>`, `velocity.<joint>`, `effort.<joint>`. A bare `<joint>` means position. Joints are looked up by name. |
| `sensor_msgs/msg/Imu` | `float64` | required | Dotted path, e.g. `orientation.x`, `angular_velocity.z`. |
| `nav_msgs/msg/Odometry` | `float64` | required | Dotted path, e.g. `pose.pose.position.x`, `twist.twist.angular.z`. |
| `geometry_msgs/msg/Twist` | `float64` | required | `linear.x` ... `angular.z`. |
| `geometry_msgs/msg/TwistStamped` | `float64` | required | `twist.linear.x` ... |
| `control_msgs/msg/MultiDOFCommand` | `float64` | required | `values.<dof>`, `values_dot.<dof>`. A bare `<dof>` means `values`. |
| `trajectory_msgs/msg/JointTrajectory` | `float64` | required | First point only. `position.<joint>`, `velocity.<joint>`, `acceleration.<joint>`, `effort.<joint>`. `positions`, `velocities` and `accelerations` also accepted. A bare `<joint>` means position. A message with no points is dropped. |
| `sensor_msgs/msg/Joy` | `float32` | required | `axes.<i>`, `buttons.<i>`. A bare `<i>` means `axes`. |
| `std_msgs/msg/Float32MultiArray` | `float32` | optional | `data`. The `select` length has to equal the array length. Without `select` the stream is one wide, so only a one-element `data` decodes. |
| `std_msgs/msg/Float64MultiArray` | `float64` | optional | Same. |
| `std_msgs/msg/Int32MultiArray` | `int32` | optional | Same. |
| `std_msgs/msg/Float32`, `Float64`, `Int32`, `Int64`, `Bool` | matching | no | Scalar. |
| `std_msgs/msg/String` | `string` | no | Text. |

## Encoders

An encoder writes an array into a message. Only these types work as actions
or `teleop.feedback`.

| Type | `select` | Stamp | Notes |
|---|---|---|---|
| `sensor_msgs/msg/JointState` | required | `header.stamp` set | A selected field has to cover every named joint. Unselected fields stay empty. |
| `trajectory_msgs/msg/JointTrajectory` | required | `header.stamp` set | One point, `time_from_start` zero. |
| `sensor_msgs/msg/Joy` | required | `header.stamp` set | Arrays sized to the highest index. Gaps are zero. Buttons are rounded. |
| `control_msgs/msg/MultiDOFCommand` | required | none | Full coverage, as for JointState. |
| `geometry_msgs/msg/Twist` | required | none | |
| `geometry_msgs/msg/TwistStamped` | required | `header.stamp` set | |
| `std_msgs/msg/Float32`, `Float64` | no | none | The source has to be one wide. |
| `std_msgs/msg/Float32MultiArray`, `Float64MultiArray` | optional | none | One wide without `select`. |
| `std_msgs/msg/Int32MultiArray` | optional | none | One wide without `select`. Values rounded. |

On the serve path a frame runs the inverse `apply` pipeline, then a width
check, then a finiteness check. A frame with NaN or Inf in it is dropped
whole and logged. If frames keep failing, the watchdog publishes the declared
`safety` action.

## Operators

| Operator | Form | Inverse | On actions | Behavior |
|---|---|---|---|---|
| `rad2deg` | `rad2deg` | `deg2rad` | yes | Bijective. Round-trip checked at load. |
| `clamp` | `clamp: {min: lo, max: hi}` | itself | yes | Element-wise clip. Exactly the keys `min` and `max`, both finite, `min <= max`. NaN passes through. |
| `resize` | `resize: [h, w]` | none | no | Nearest-neighbour. Image sources only, an error elsewhere. `h` and `w` integers from 1 to 8192. Sets the image size the dataset stores. |

Recording runs `apply` front to back. Serving runs it back to front through
the inverses.

## Custom codecs

Register a decoder for a type Rosetta doesn't ship, and an encoder if the
type is used as an action.

```python
import numpy as np
from rosetta.frames.codecs import register_decoder, register_encoder

@register_decoder("my_msgs/msg/MySensor", dtype="float64")
def decode_my_sensor(msg, spec):
    # spec.names is the select list as a tuple, () without select
    return np.array([msg.field1, msg.field2], dtype=np.float64)

@register_encoder("my_msgs/msg/MyCommand")
def encode_my_command(values, spec, stamp_ns=None):
    # values already ran the inverse apply pipeline
    ...
```

Two ways to make Rosetta find them:

- Advertise the module under the entry-point group `rosetta.codecs`. Rosetta
  imports it when a contract loads, and the contract only names the type.
- Name the function in the contract with `decoder: my_pkg.codecs:decode_my_sensor`
  or `encoder: ...`. The module has to be importable. The path is checked at
  load.

Registering a codec for a type that already has one is an error unless you
pass `override=True`. Passing `override=True` for a type with no codec is an
error too.

## Custom operators

```python
from rosetta.contract.operators import register_operator
```

Advertise the module under the entry-point group `rosetta.operators`. An
operator registers with a `kind`: `FORWARD_ONLY` like `resize`,
`BIDIRECTIONAL` like `clamp`, or `BIJECTIVE` like `rad2deg`. Only a
`BIJECTIVE` operator is round-trip checked at load. A duplicate name is an
error unless you pass `override=True`.
