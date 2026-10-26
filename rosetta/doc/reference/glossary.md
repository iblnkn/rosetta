# Glossary

Stream
: Messages of one type on one topic, each stamped on its own clock. What a
  ROS 2 robot publishes.

Frame
: One sample of every key at one tick, as a flat `{key: array}` mapping.
  What a policy consumes and produces.

Tick
: One point on the frame clock. Ticks are `1 / fps` apart.

Key
: The name of one entry in a frame, such as `observation.state` or `action`.

Source
: One stream feeding a key. A key with several sources is a multi-source key.

Channel
: A stream's address: topic, message type, QoS.

Timeline
: The clock a stream is read on. Every channel has `receive`. A message type
  with a `std_msgs/Header` field named `header` also has `header`.

Align
: How a stream's samples land on ticks: `hold`, `asof` or `drop`, on a
  declared timeline.

Select
: Which fields of a message, in which order.

Apply
: Operators run on the selected values, forward when recording, inverse when
  serving.

Contract
: One YAML file per robot declaring every key, its sources, and the align,
  select and apply settings for each source.

Episode
: One recorded demonstration. One bag directory.

Adapter
: The package binding a learning framework to Rosetta. It registers a dataset
  writer and a policy runner under the framework's name.

Warmup
: The interval before every observation stream has delivered a sample. No
  frames are produced during warmup.
