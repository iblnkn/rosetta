# Glossary

Terms as Rosetta and its docs use them, in alphabetical order.

Adjunct
: A channel the recorder records to the bag and nothing decodes, such as
  `/tf`. Declared under `adjunct` in the contract.

Align
: How a stream's samples land on ticks: the strategy `hold`, `asof` or `drop`,
  on a named timeline. Required on every source.

Apply
: The ordered list of operators run on a source's values: front to back when
  decoding, back to front through the inverses when encoding.

Channel
: The `channel` block of a source: topic, message type, QoS, and the codec
  and safety settings for that topic.

Codec
: A decoder (message to array) or an encoder (array to message), registered
  for one message type. The built-in codecs come with the robot interface.

Contract
: One YAML file per robot declaring every frame key, its sources, and how
  each source is aligned, selected and transformed. The same file drives
  recording, porting and deployment.

Dataset writer
: The part of a framework adapter that writes ported frames in the
  framework's dataset format. `rosetta_port --framework` picks it.

Episode
: One recorded demonstration. One bag directory, and one episode in the
  ported dataset.

Frame
: One sample of every frame key at one tick, as a flat `{key: value}`
  mapping. LeRobot calls it a frame too (one row of a dataset). It is not a
  tf coordinate frame.

Frame key
: The name of one entry in a frame, such as `observation.state` or `action`.
  LeRobot calls it a feature.

Framework adapter
: A package that binds a learning framework to Rosetta by registering a
  dataset writer and a policy runner under the framework's name, such as
  `lerobot` from `lerobot_rosetta`.

Operator
: A named transform an `apply` list can use, such as `rad2deg`, `clamp` or
  `resize`. Each has a forward direction and possibly an inverse.

Policy runner
: The part of a framework adapter that runs a policy against live frames.
  Also the node that hosts it, `policy_runner_node` (node name
  `policy_runner`).

Port
: To turn bags into a training dataset through a contract, with
  `rosetta_port`.

Robot interface
: The package that binds Rosetta to a robot middleware, named by
  `robot_interface` in a contract. `ros2` comes from `rosetta_ros`. Unrelated
  to `rosetta_interfaces`, the ROS 2 package of Rosetta's action, message and
  service definitions.

Safety
: What an action channel publishes when actions stop or the node
  deactivates: `none`, `zeros` or `hold`.

Select
: Which values a source takes from each message, in which order. It also
  sets the source's width.

Sidecar
: The copy of the contract stored with a dataset,
  `meta/rosetta_contract.yaml` in a LeRobot dataset.

Source
: One channel feeding a frame key, with its align, select, apply and kind. A
  key with several sources is a multi-source key.

Stream
: The messages on one topic, each arriving on its own clock. What a ROS 2
  robot publishes.

Teleop
: The contract section for a human device: `input` drives action topics,
  `events` maps buttons to named events, `feedback` sends observations back
  to the device.

Tick
: One point on the frame clock. Ticks are `1 / fps` apart.

Timeline
: A named timestamp a channel provides. In `ros2`, every type has `receive`
  (arrival time), and a type with a `std_msgs/Header` field named `header`
  also has `header`.

Warmup
: The interval before every observation source has delivered a sample.
  Porting skips ticks during warmup.

Watchdog
: The live check that publishes each action channel's safety action after
  more than two frame periods without a published action frame.
