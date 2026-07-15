# Copyright 2025 Isaac Blankenau
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

"""Per-frame task labels from a task topic.

The porter stamps every frame with a task string. Precedence: the latest
message on a contract task topic at or before the tick (hold semantics —
tasks may change mid-episode) wins over the operator prompt recorded in bag
metadata, which is the fallback for frames before the first task message or
bags with none.
"""

import pytest

pytest.importorskip("rosbag2_py")

import rosbag2_py
import yaml
from rclpy.serialization import serialize_message
from rosetta.contract.schema import Align, Channel, Source
from rosetta.contract.specs import ObservationStreamSpec
from rosetta.robots.ros2.bag_metadata import BAG_METADATA_KEY, BAG_PROMPT_KEY
from rosetta.robots.ros2.offline.bag_frames import iter_bag_frames
from sensor_msgs.msg import JointState
from std_msgs.msg import String

FPS = 10
STEP_NS = int(1e9 / FPS)
T0 = 1_000_000_000

OBS_TOPIC = "/joint_states"
TASK_TOPIC = "/task_prompt"

SPECS = [
    ObservationStreamSpec(
        key="observation.state",
        names=["position.j1"],
        fps=FPS,
        source=Source(
            channel=Channel(topic=OBS_TOPIC, type="sensor_msgs/msg/JointState"),
            align=Align("hold", "receive"),
        ),
        is_image=False,
        image_resize=None,
        dtype="float64",
    )
]
WARMUP_KEYS = {"observation.state"}
TASK_TOPICS = {TASK_TOPIC: "std_msgs/msg/String"}


def _js(pos):
    msg = JointState()
    msg.name = ["j1"]
    msg.position = [float(pos)]
    return msg


def _task(text):
    msg = String()
    msg.data = text
    return msg


def _write_bag(bag_dir, events, prompt=""):
    """Write (topic, type, ns, msg) events to a bag, plus prompt metadata."""
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topic_types = {t: ty for t, ty, _, _ in events}
    for tid, (topic, type_str) in enumerate(sorted(topic_types.items())):
        try:
            meta = rosbag2_py.TopicMetadata(id=tid, name=topic, type=type_str, serialization_format="cdr")
        except TypeError:  # pre-Jazzy signature without id
            meta = rosbag2_py.TopicMetadata(topic, type_str, "cdr")
        writer.create_topic(meta)
    for topic, _, ns, msg in sorted(events, key=lambda e: e[2]):
        writer.write(topic, serialize_message(msg), ns)
    del writer  # flush + write metadata.yaml

    if prompt:
        meta_path = bag_dir / "metadata.yaml"
        meta = yaml.safe_load(meta_path.read_text())
        info = meta[BAG_METADATA_KEY]
        if not info.get("custom_data"):
            info["custom_data"] = {}
        info["custom_data"][BAG_PROMPT_KEY] = prompt
        meta_path.write_text(yaml.safe_dump(meta))


def _frames(bag_dir):
    return list(iter_bag_frames(bag_dir, SPECS, warmup_keys=WARMUP_KEYS, task_topics=TASK_TOPICS))


def _obs_events(n_ticks):
    return [(OBS_TOPIC, "sensor_msgs/msg/JointState", T0 + i * STEP_NS, _js(i)) for i in range(n_ticks)]


def test_no_task_messages_falls_back_to_prompt(tmp_path):
    bag = tmp_path / "ep0"
    _write_bag(bag, _obs_events(4), prompt="fold the towel")
    frames = _frames(bag)
    assert len(frames) == 4
    assert all(f["task"] == "fold the towel" for f in frames)


def test_task_topic_wins_over_prompt_after_first_message(tmp_path):
    bag = tmp_path / "ep0"
    events = [
        *_obs_events(4),
        # Arrives between tick 1 and tick 2: ticks 0-1 keep the prompt,
        # ticks 2-3 carry the topic task.
        (TASK_TOPIC, "std_msgs/msg/String", T0 + STEP_NS + STEP_NS // 2, _task("pick up the cube")),
    ]
    _write_bag(bag, events, prompt="fold the towel")
    tasks = [f["task"] for f in _frames(bag)]
    assert tasks == ["fold the towel", "fold the towel", "pick up the cube", "pick up the cube"]


def test_task_changes_mid_episode(tmp_path):
    bag = tmp_path / "ep0"
    events = [
        *_obs_events(4),
        (TASK_TOPIC, "std_msgs/msg/String", T0, _task("approach the cube")),
        # Exactly at tick 2: included in frame 2 (same boundary rule as
        # observations — a message at tick t lands in frame t).
        (TASK_TOPIC, "std_msgs/msg/String", T0 + 2 * STEP_NS, _task("grasp the cube")),
    ]
    _write_bag(bag, events, prompt="unused prompt")
    tasks = [f["task"] for f in _frames(bag)]
    assert tasks == ["approach the cube", "approach the cube", "grasp the cube", "grasp the cube"]


def test_without_task_topics_prompt_only(tmp_path):
    # A contract with no tasks section behaves exactly as before.
    bag = tmp_path / "ep0"
    events = [
        *_obs_events(2),
        (TASK_TOPIC, "std_msgs/msg/String", T0, _task("should be ignored")),
    ]
    _write_bag(bag, events, prompt="fold the towel")
    frames = list(iter_bag_frames(bag, SPECS, warmup_keys=WARMUP_KEYS))
    assert [f["task"] for f in frames] == ["fold the towel", "fold the towel"]
