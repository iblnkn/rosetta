# Copyright 2026 Isaac Blankenau
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small mcap recordings for tests: a camera and joint states, written with rosbag2."""

import numpy as np
import rosbag2_py
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image, JointState

PROMPT_KEY = "lerobot.operator_prompt"


def joint_state(a, b):
    msg = JointState()
    msg.name = ["j1", "j2"]
    msg.position = [float(a), float(b)]
    return msg


def image(value, size=16):
    """A flat rgb8 image, every pixel ``value``."""
    msg = Image()
    msg.height, msg.width, msg.encoding, msg.step = size, size, "rgb8", 3 * size
    msg.data = np.full((size, size, 3), value, dtype=np.uint8).tobytes()
    return msg


def write_bag(bag_dir, events, *, prompt="grab"):
    """Write ``(topic, receive_ns, msg)`` events to an mcap bag recorded with operator prompt ``prompt``."""
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap", custom_data={PROMPT_KEY: prompt}),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topics = {}
    for topic, _, msg in events:
        topics.setdefault(topic, f"sensor_msgs/msg/{type(msg).__name__}")
    for i, (topic, type_name) in enumerate(topics.items()):
        writer.create_topic(rosbag2_py.TopicMetadata(id=i, name=topic, type=type_name, serialization_format="cdr"))
    for topic, ns, msg in sorted(events, key=lambda event: event[1]):
        writer.write(topic, serialize_message(msg), recv_timestamp=ns, send_timestamp=ns)
    del writer
