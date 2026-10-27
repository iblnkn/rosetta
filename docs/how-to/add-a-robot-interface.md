# Add a robot interface

A robot interface connects Rosetta to a robot middleware, such as ROS 2 or
MQTT. It's a package that registers one class, and contracts select it by
name with `robot_interface:`. `rosetta_ros` is the working example, and
`rosetta/test/fake_interface.py` in the repository is the smallest one.

## Implement the protocol

From `rosetta.robots.protocols`:

```python
class RobotInterface:
    def register_codecs(self): ...                   # import your built-in codecs
    def provided_timelines(self, msg_type): ...      # -> frozenset of timeline names
    def validate_qos(self, qos): ...                 # raise ValueError if it doesn't parse
```

Contract loading calls all three. `provided_timelines` raises `ValueError` for
a type your middleware doesn't know, and every channel's `align.timeline` must
be in the set it returns. Import your middleware inside the methods, not at
module level, so finding the interface stays cheap. The class needs a
zero-argument constructor.

## Register codecs

Decoders turn a message into an array and encoders turn an array back into a
message, registered with `register_decoder` and `register_encoder` from
`rosetta.frames.codecs`. `register_codecs` imports the modules that call them.
The [Message types](../reference/message-types.md#custom-codecs) page has the
API. Codecs share one registry keyed by type string, so choose type strings
that don't collide with another interface's.

## Register the interface

```python
entry_points={
    "rosetta.robot_interfaces": ["mqtt = rosetta_mqtt.interface:MqttInterface"],
},
```

A contract with `robot_interface: mqtt` now loads wherever the package is
installed, and one naming an interface that isn't installed fails to load.

## Connect it to a robot

Contract loading is all core needs. Running a robot takes a live bridge that
implements `FrameIO` from `rosetta.frames.protocols` (`sample_frame()`,
`warmed_up`, `publish_frame(frame)`, `send_safety_action()`,
`reset_state()`), plus whatever processes host it and record data in your
middleware. In `rosetta_ros` these are `TopicBridge`, the policy runner and
recorder nodes, and `rosetta_port`. A `FrameIO` can run any framework
adapter's `PolicyRunner` unchanged.

## Package it

Depend on `rosetta`. Keep the interface in your own repository, or propose it
for `robots/<middleware>/` in the `rosetta` repository: Rosetta's
[CONTRIBUTING.md](https://github.com/iblnkn/rosetta/blob/main/CONTRIBUTING.md)
says which.
