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

"""Which topics the episode recorder records, and with what QoS.

Pure policy over topic names, types, and QoS profiles: no node and no
subscriptions. A topic is a ``(name, type, qos)`` tuple throughout.
"""

from __future__ import annotations

from typing import Callable

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosbag2_py._storage import (
    Duration as Rosbag2Duration,
)
from rosbag2_py._storage import (
    QoS as Rosbag2QoS,
)
from rosbag2_py._storage import (
    rmw_qos_durability_policy_t,
    rmw_qos_history_policy_t,
    rmw_qos_liveliness_policy_t,
    rmw_qos_reliability_policy_t,
)

from rosetta.contract.specs import iter_specs
from rosetta_ros.rclpy_utils import qos_profile_from_dict

Topic = tuple[str, str, QoSProfile]

# Safety ceiling on the widened QoS depth for per-episode latched
# subscriptions, regardless of how many publishers are discovered.
MAX_LATCHED_SUB_DEPTH = 64

# One stored stream per camera family, best-first: any compressed form beats
# storing raw frames; raw ("") is the fallback when no re-encoding is
# advertised.
_IMAGE_STREAM_PREFERENCE = ("/compressed", "/zstd", "/theora", "/compressedDepth", "")


def contract_topics(contract, *, use_sim_time: bool) -> list[Topic]:
    """Every topic the contract names: observations, actions, tasks, and adjuncts.

    ``/clock`` is added under simulation time so playback can drive sim time.
    Duplicates collapse to one entry (see :func:`dedup_topics`).
    """
    topics = [
        (spec.source.channel.topic, spec.source.channel.type, qos_profile_from_dict(spec.source.channel.qos))
        for spec in iter_specs(contract)
    ]
    topics += [(t.channel.topic, t.channel.type, qos_profile_from_dict(t.channel.qos)) for t in contract.tasks]
    topics += [(a.topic, a.type, qos_profile_from_dict(a.qos)) for a in contract.adjunct]
    if use_sim_time:
        topics.append(("/clock", "rosgraph_msgs/msg/Clock", qos_profile_from_dict(None)))
    return dedup_topics(topics)


def dedup_topics(topics: list[Topic]) -> list[Topic]:
    """Collapse duplicate topic entries, keeping the first; reject conflicting declarations.

    A topic referenced by several contract sections (e.g. one topic read as
    both an observation and a reward) must yield one subscription and one bag
    registration.

    Raises:
        ValueError: one topic is declared with different types or QoS.

    """
    deduped: dict[str, Topic] = {}
    for entry in topics:
        topic, type_str, qos = entry
        prev = deduped.get(topic)
        if prev is None:
            deduped[topic] = entry
            continue
        if prev[1] != type_str:
            raise ValueError(
                f"Topic '{topic}' appears in multiple contract sections with different types ({prev[1]} vs {type_str})"
            )
        if extract_qos_numeric_values(prev[2]) != extract_qos_numeric_values(qos):
            raise ValueError(
                f"Topic '{topic}' appears in multiple contract sections with conflicting qos; unify the entries"
            )
    return list(deduped.values())


def latched_sub_depth(configured_depth: int, publisher_count: int) -> int:
    """QoS depth for a per-episode TRANSIENT_LOCAL subscription.

    The contract's configured QoS depth is a wire/DDS matching hint
    (conventionally 1). It is not the count of distinct publishers latched
    on the topic. Several independent nodes can publish the same
    TRANSIENT_LOCAL topic (e.g. robot_state_publisher plus a
    sensor-calibration static_transform_publisher both on /tf_static),
    each redelivering its own latched sample when the subscription is
    created. A depth below the live publisher count lets one redelivered
    sample evict another in the RMW queue before the executor drains it.
    Verified against rmw_zenoh and cyclonedds.

    Args:
        configured_depth: Depth from the contract's QoS for the topic.
        publisher_count: Live publisher count from ``count_publishers``.

    Returns:
        A depth at least ``publisher_count``, capped at
        ``MAX_LATCHED_SUB_DEPTH``.

    """
    return min(max(configured_depth, publisher_count), MAX_LATCHED_SUB_DEPTH)


def image_transport_auto_skips(
    graph: dict[str, str],
    recorded: set[str],
    include: Callable[[str], bool] = lambda _t: False,
    exclude: Callable[[str], bool] = lambda _t: False,
) -> set[str]:
    """Family members to skip so each camera stores exactly one stream.

    image_transport publishers are lazy: every re-encoding (``compressed``/
    ``zstd``/``theora``/``compressedDepth``) encodes only while subscribed, so
    recording all of them makes the camera node encode each frame several ways
    at once — enough load to halve its capture rate.

    A family is a raw ``sensor_msgs/msg/Image`` topic plus whatever
    re-encodings of it are advertised. Members that are contract-``recorded``
    or match ``include`` are pinned: they record, and the rest of the family
    is skipped. With nothing pinned, the best non-``exclude``d member by
    :data:`_IMAGE_STREAM_PREFERENCE` wins — a fully excluded family records
    nothing. Topics outside any family are not this policy's business.

    Args:
        graph: Topic name -> message type for every advertised topic.
        recorded: Topics already recorded regardless of discovery (contract).
        include: Predicate pinning topics the user explicitly whitelisted.
        exclude: Predicate removing candidates (the user's blacklist).

    Returns:
        Set of topic names to exclude from auto-recording.

    """
    skips: set[str] = set()
    for base, type_str in graph.items():
        if type_str != "sensor_msgs/msg/Image":
            continue
        family = [base + s for s in _IMAGE_STREAM_PREFERENCE if base + s in graph]
        keep = {t for t in family if t in recorded or include(t)}
        if not keep:
            keep = {next((t for t in family if not exclude(t)), None)} - {None}
        skips.update(t for t in family if t not in keep)
    return skips


def adapt_publisher_qos(publishers_info, *, warn=None) -> QoSProfile:
    """
    Build a subscription QoS for a discovered topic, adapted across all its publishers.

    Mirrors ``ros2 bag record``. Request RELIABLE only if every publisher
    offers it, and TRANSIENT_LOCAL only if every publisher offers it.
    Requesting either policy when one publisher does not offer it makes that
    publisher's messages silently undeliverable. ``warn`` is a callable taking
    a message string. It fires when publishers disagree, since the adapted
    profile is then a compromise.

    Args:
        publishers_info: Publisher endpoint info from
            ``get_publishers_info_by_topic``. Empty yields the default profile.
        warn: Optional callable invoked once with a message when publishers
            offer mixed QoS.

    Returns:
        A QoSProfile safe to subscribe with against every current publisher.

    """
    if not publishers_info:
        return qos_profile_from_dict(None)
    profiles = [p.qos_profile for p in publishers_info]
    reliable = all(p.reliability == ReliabilityPolicy.RELIABLE for p in profiles)
    latched = all(p.durability == DurabilityPolicy.TRANSIENT_LOCAL for p in profiles)
    if warn is not None and len({(p.reliability, p.durability) for p in profiles}) > 1:
        warn("publishers offer mixed QoS; subscribing adapted profile")
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=max(10, *(p.depth for p in profiles)),
        reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL if latched else DurabilityPolicy.VOLATILE,
    )


def extract_qos_numeric_values(q: QoSProfile) -> dict[str, int]:
    """Flatten a QoSProfile's policy enums to their raw RMW integer codes.

    Keys are ``depth``, ``history``, ``reliability``, ``durability`` and
    ``liveliness``; rosbag2's QoS setters want these ints.
    """
    return {
        "depth": q.depth,
        "history": q.history.value,
        "reliability": q.reliability.value,
        "durability": q.durability.value,
        "liveliness": q.liveliness.value,
    }


# rosbag2_py._storage.Duration takes int32 seconds / uint32 nanoseconds.
# "Infinite"/unset durations discovered from real publishers come back as
# RMW_DURATION_INFINITE (int64 nanoseconds max), which overflows those
# bounds, so clamp to the max representable sentinel.
_MAX_DURATION_SEC = 2147483647
_MAX_DURATION_NSEC = 4294967295


def _rosbag2_duration(rclpy_duration) -> Rosbag2Duration:
    ns = int(getattr(rclpy_duration, "nanoseconds", 0) or 0)
    return Rosbag2Duration(
        min(ns // 1_000_000_000, _MAX_DURATION_SEC),
        min(ns % 1_000_000_000, _MAX_DURATION_NSEC),
    )


def qos_to_rosbag2(q: QoSProfile) -> Rosbag2QoS:
    """Convert an rclpy QoSProfile to a rosbag2_py QoS for TopicMetadata.

    The QoS setter methods require rmw_qos_*_policy_t enums, not raw ints.
    """
    vals = extract_qos_numeric_values(q)
    bag_qos = Rosbag2QoS(vals["depth"])
    bag_qos = bag_qos.history(rmw_qos_history_policy_t(vals["history"]))
    bag_qos = bag_qos.reliability(rmw_qos_reliability_policy_t(vals["reliability"]))
    bag_qos = bag_qos.durability(rmw_qos_durability_policy_t(vals["durability"]))
    bag_qos = bag_qos.liveliness(rmw_qos_liveliness_policy_t(vals["liveliness"]))
    bag_qos = bag_qos.deadline(_rosbag2_duration(q.deadline))
    bag_qos = bag_qos.lifespan(_rosbag2_duration(q.lifespan))
    return bag_qos.liveliness_lease_duration(_rosbag2_duration(q.liveliness_lease_duration))
